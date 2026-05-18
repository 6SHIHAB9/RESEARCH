import json
import logging
import random
import os
from openai import AsyncOpenAI
from core.world import World, WORLD_WIDTH, WORLD_HEIGHT, Landmark, Resource
from agents.agent import Agent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Safe parsing helpers — prevents float(None) crashes from malformed LLM output
# ──────────────────────────────────────────────────────────────────────────────

def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value, default="") -> str:
    if value is None:
        return default
    return str(value)


def safe_choice(value, allowed: list, default: str) -> str:
    if value in allowed:
        return value
    return default


# ──────────────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────────────

def _get_client() -> AsyncOpenAI:
    api_key = os.environ.get("CEREBRAS_API_KEY", "placeholder")
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )


MODEL = "llama3.1-8b"
MAX_AGENTS_PER_BATCH = 8

VALID_ACTIONS = [
    "speak", "observe", "wander", "ignore", "retreat",
    "linger", "forage", "confront", "trade", "rest",
]


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_agent_context(
    agent: Agent,
    nearby: list[Agent],
    nearby_landmarks: list[Landmark],
    nearby_resources: list[Resource],
) -> dict:
    memory_summary = [m.get("summary", "") for m in agent.memory[-4:]]

    nearby_info = []
    for other in nearby[:5]:
        rel = agent.get_relationship(other.id)
        label = rel.label()
        nearby_info.append({
            "id": other.id,
            "name": other.name,
            "mood": other.mood_label(),
            "relationship": label,
            "encounters": rel.encounters,
            "dominant_need": other.needs.dominant_need(),
        })

    landmark_info = [{"name": lm.name, "kind": lm.kind} for lm in nearby_landmarks]

    resource_info = []
    for res in nearby_resources:
        claimer_name = None
        if res.claimed_by:
            claimer = None
            # We don't have world ref here; pass name via caller-injected data
            claimer_name = res.claimed_by
        resource_info.append({
            "name": res.name,
            "kind": res.kind,
            "amount": res.amount,
            "claimed_by": claimer_name,
        })

    # Weighted hint for the LLM about likely behavior
    weights = agent.behavior_weights()
    top_drives = sorted(weights.items(), key=lambda x: -x[1])[:3]
    drive_hint = ", ".join(f"{k}({v:.2f})" for k, v in top_drives)

    return {
        "id": agent.id,
        "name": agent.name,
        "personality": agent.personality,
        "mood": agent.mood_label(),
        "needs": agent.needs.to_dict(),
        "dominant_need": agent.needs.dominant_need(),
        "behavior_drive": drive_hint,
        "resources_held": agent.resources,
        "territory_claim": agent.territory_claim,
        "nearby_agents": nearby_info,
        "nearby_landmarks": landmark_info,
        "nearby_resources": resource_info,
        "memories": memory_summary,
        "last_action": agent.last_action,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_batch_prompt(agent_contexts: list[dict]) -> str:
    agents_json = json.dumps(agent_contexts, indent=2)

    return f"""You are simulating an emergent digital society. Each agent is survival-driven,
emotionally reactive, and socially complex. They form opinions, hold grudges, make alliances,
compete over resources, and act on their dominant needs.

Agents are NOT passive observers. They have internal hunger, loneliness, fear, curiosity,
and aggression that MUST drive their choices. Their memories shape who they trust, fear, or resent.

For each agent below, decide what happens this tick.

Valid actions: speak, observe, wander, ignore, retreat, linger, forage, confront, trade, rest

Respond ONLY with a valid JSON array — one object per agent, same order as input.
Each object must have EXACTLY these fields (no extras, no missing):
- "id": string — copy from input
- "action": string — one of the valid actions above
- "phrase": string or null — spoken words if action is "speak" (max 15 words). Must reflect personality,
  emotional state, and memories. Avoid filler. Use suspicion, persuasion, bargaining, threats,
  confessions, or alliance-seeking. null for non-speak actions.
- "mood_delta": float from -0.3 to 0.3
- "target_id": string or null — id of agent being interacted with
- "trust_delta": float from -0.3 to 0.3 — change in trust toward target
- "fear_delta": float from 0.0 to 0.3 — change in fear toward target
- "affinity_delta": float from -0.3 to 0.3 — change in affinity toward target
- "hostility_delta": float from 0.0 to 0.3 — change in hostility toward target
- "dx": float from -2.0 to 2.0 — horizontal movement
- "dy": float from -2.0 to 2.0 — vertical movement
- "memory_note": string or null — one sentence under 15 words to remember, referencing agent names
- "resource_action": string or null — one of: "forage", "claim", "trade_offer", null
- "resource_target": string or null — name of the resource being acted on

BEHAVIORAL RULES:
- Agents with HIGH hunger → prefer "forage" or "trade" actions, seek food resources
- Agents with HIGH loneliness → prefer "speak", seek nearby agents
- Agents with HIGH fear → prefer "retreat", move away from perceived threats
- Agents with HIGH aggression → may "confront" rivals, claim territories
- Agents with HIGH curiosity → "observe" or "wander" toward unknown areas
- Agents with LOW energy → "rest" to recover
- Repeat positive encounters build alliances; repeat negative encounters build grudges
- Agents should reference memories naturally: mention names of those who helped or wronged them
- Territorial agents defend resource claims aggressively
- Desperate (very hungry) agents may attempt risky confrontations
- Dialogue should be driven by need: pleas, deals, threats, suspicions, opinions
- Avoid "nice weather today" filler. Every spoken phrase should mean something.
- Most agents do only 1 meaningful thing this tick. Inaction is valid.

Agents:
{agents_json}

Respond with ONLY the JSON array. No explanation. No markdown fences.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main tick runner
# ──────────────────────────────────────────────────────────────────────────────

async def run_tick_batch(world: World):
    """Run one batched LLM tick for the world."""
    # Tick needs decay for ALL agents each cycle
    for agent in world.agents.values():
        agent.needs.tick_decay(agent.personality)

    # Replenish world resources
    world.tick_resources()

    all_agents = list(world.agents.values())
    random.shuffle(all_agents)
    active_agents = all_agents[:MAX_AGENTS_PER_BATCH]

    agent_contexts = []
    for agent in active_agents:
        nearby = world.get_nearby_agents(agent)
        nearby_landmarks = world.get_nearby_landmarks(agent)
        nearby_resources = world.get_nearby_resources(agent)
        ctx = _build_agent_context(agent, nearby, nearby_landmarks, nearby_resources)
        agent_contexts.append(ctx)

    prompt = _build_batch_prompt(agent_contexts)

    logger.info(f"  → Sending batch of {len(agent_contexts)} agents to Cerebras...")

    client = _get_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.1,
        max_tokens=1600,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]

    try:
        results = json.loads(raw)
        _apply_results(world, results, active_agents)
    except json.JSONDecodeError as e:
        logger.error(f"  ✗ JSON parse error: {e}")
        for agent in active_agents:
            agent.move(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
            agent.last_action = "wandering"


# ──────────────────────────────────────────────────────────────────────────────
# Result applier
# ──────────────────────────────────────────────────────────────────────────────

def _apply_results(world: World, results: list[dict], active_agents: list[Agent]):
    agent_map = {a.id: a for a in active_agents}

    for result in results:
        agent_id = safe_str(result.get("id"))
        agent = agent_map.get(agent_id)
        if not agent:
            continue

        action = safe_choice(result.get("action"), VALID_ACTIONS, "wander")
        phrase = result.get("phrase")
        mood_delta = safe_float(result.get("mood_delta"), 0.0)
        target_id = result.get("target_id")

        # Rich relationship deltas (with safe_float to prevent None crashes)
        trust_delta = safe_float(result.get("trust_delta"), 0.0)
        fear_delta = safe_float(result.get("fear_delta"), 0.0)
        affinity_delta = safe_float(result.get("affinity_delta"), 0.0)
        hostility_delta = safe_float(result.get("hostility_delta"), 0.0)

        # Movement — safe_float prevents the float(None) crash
        dx = safe_float(result.get("dx"), random.uniform(-0.8, 0.8))
        dy = safe_float(result.get("dy"), random.uniform(-0.8, 0.8))

        memory_note = result.get("memory_note")
        resource_action = result.get("resource_action")
        resource_target_name = result.get("resource_target")

        old_x, old_y = agent.x, agent.y

        # ── Movement ─────────────────────────────────────────────────────────
        # Fear pulls away from threats; loneliness pulls toward others
        if action == "retreat" and target_id:
            target = world.agents.get(target_id)
            if target:
                flee_dx = agent.x - target.x
                flee_dy = agent.y - target.y
                dist = max(0.1, (flee_dx**2 + flee_dy**2)**0.5)
                dx = (flee_dx / dist) * 2.0
                dy = (flee_dy / dist) * 2.0

        agent.move(dx, dy)

        # ── Needs updates from actions ────────────────────────────────────────
        if action == "rest":
            agent.needs.energy = min(1.0, agent.needs.energy + 0.15)
            agent.needs.loneliness = min(1.0, agent.needs.loneliness + 0.05)
        elif action == "speak" and target_id:
            agent.needs.loneliness = max(0.0, agent.needs.loneliness - 0.12)
        elif action == "forage":
            # Try to consume from a nearby food resource
            _handle_forage(world, agent)
        elif action == "confront" and target_id:
            agent.needs.aggression = max(0.0, agent.needs.aggression - 0.1)
            agent.needs.fear = min(1.0, agent.needs.fear + 0.05)

        # ── Mood ─────────────────────────────────────────────────────────────
        old_mood = agent.mood
        agent.nudge_mood(mood_delta)

        # ── Phrases ──────────────────────────────────────────────────────────
        if phrase and action == "speak":
            agent.recent_phrases.append(phrase)
            if len(agent.recent_phrases) > 5:
                agent.recent_phrases = agent.recent_phrases[-5:]

        # ── Relationships ────────────────────────────────────────────────────
        if target_id and (trust_delta or fear_delta or affinity_delta or hostility_delta):
            agent.update_relationship(
                target_id,
                trust_delta=trust_delta,
                fear_delta=fear_delta,
                affinity_delta=affinity_delta,
                hostility_delta=hostility_delta,
            )
            target = world.agents.get(target_id)
            if target:
                # Reciprocal effect (weaker)
                target.update_relationship(
                    agent_id,
                    trust_delta=trust_delta * 0.5,
                    fear_delta=fear_delta * 0.3,
                    affinity_delta=affinity_delta * 0.5,
                    hostility_delta=hostility_delta * 0.3,
                )
                # Target fear spikes if confronted
                if action == "confront":
                    target.needs.fear = min(1.0, target.needs.fear + 0.15)

        # ── Resource interactions ─────────────────────────────────────────────
        if resource_action and resource_target_name:
            _handle_resource_action(world, agent, resource_action, resource_target_name)

        # ── Memory ───────────────────────────────────────────────────────────
        if memory_note:
            agent.add_memory({"tick": world.tick_number, "summary": memory_note})

        agent.last_action = action
        agent.last_interaction_tick = world.tick_number

        # ── Logging ──────────────────────────────────────────────────────────
        _log_action(world, agent, action, phrase, target_id, dx, dy,
                    old_mood, old_x, old_y, trust_delta, hostility_delta)


def _handle_forage(world: World, agent: Agent):
    """Agent attempts to gather food from a nearby resource."""
    for res in world.get_nearby_resources(agent):
        if res.kind == "food" and res.amount > 0:
            # Check if resource is claimed by a rival
            if res.claimed_by and res.claimed_by != agent.id:
                claimer = world.agents.get(res.claimed_by)
                if claimer:
                    agent.update_relationship(res.claimed_by, hostility_delta=0.05)
                    claimer.update_relationship(agent.id, hostility_delta=0.08, fear_delta=-0.02)
                    logger.info(f"  ⚡ {agent.name} poached from {claimer.name}'s resource ({res.name})")
                    world.log_event({
                        "type": "resource_conflict",
                        "agent": agent.name,
                        "agent_id": agent.id,
                        "victim": claimer.name if claimer else "unknown",
                        "resource": res.name,
                    })

            res.amount -= 1
            agent.resources["food"] = agent.resources.get("food", 0) + 1
            agent.needs.hunger = max(0.0, agent.needs.hunger - 0.25)
            agent.nudge_mood(0.1)
            logger.info(f"  🍖 {agent.name} foraged from {res.name} (remaining: {res.amount})")
            world.log_event({
                "type": "forage",
                "agent": agent.name,
                "agent_id": agent.id,
                "resource": res.name,
            })
            break


def _handle_resource_action(world: World, agent: Agent, action: str, resource_name: str):
    """Handle claim and trade resource actions."""
    target_res = next((r for r in world.resources if r.name == resource_name), None)
    if not target_res:
        return

    if action == "claim":
        dist = ((agent.x - target_res.x)**2 + (agent.y - target_res.y)**2)**0.5
        if dist <= 8.0:
            old_claimer_id = target_res.claimed_by
            target_res.claimed_by = agent.id
            agent.territory_claim = resource_name
            if old_claimer_id and old_claimer_id != agent.id:
                old_claimer = world.agents.get(old_claimer_id)
                agent.update_relationship(old_claimer_id, hostility_delta=0.05)
                if old_claimer:
                    old_claimer.update_relationship(agent.id, hostility_delta=0.12, fear_delta=0.05)
                    old_claimer.territory_claim = None
                    logger.info(f"  🏴 {agent.name} seized {resource_name} from {old_claimer.name}!")
                    world.log_event({
                        "type": "territory_seized",
                        "agent": agent.name,
                        "agent_id": agent.id,
                        "from": old_claimer.name if old_claimer else "unknown",
                        "resource": resource_name,
                    })
            else:
                logger.info(f"  🚩 {agent.name} claimed {resource_name}")


def _log_action(world, agent, action, phrase, target_id, dx, dy,
                old_mood, old_x, old_y, trust_delta, hostility_delta):
    """Consolidated logging for all action types."""
    target = world.agents.get(target_id) if target_id else None
    tname = target.name if target else ""

    # Landmark proximity changes
    for lm in world.landmarks:
        dist = ((agent.x - lm.x)**2 + (agent.y - lm.y)**2)**0.5
        old_dist = ((old_x - lm.x)**2 + (old_y - lm.y)**2)**0.5
        if dist <= 5.0 and old_dist > 5.0:
            logger.info(f"  📍 {agent.name} drifted toward the {lm.name}")
        elif dist <= 5.0 and action == "linger":
            logger.info(f"  🏕️  {agent.name} is lingering near the {lm.name}")

    # Relationship milestone logging
    if target_id and target:
        rel = agent.get_relationship(target_id)
        if rel.net_bond() > 0.5 and rel.encounters % 5 == 0:
            logger.info(f"  🤝 {agent.name} ↔ {tname}: strong alliance (bond {rel.net_bond():.2f})")
        elif rel.net_bond() < -0.4 and rel.encounters % 5 == 0:
            logger.info(f"  ⚔️  {agent.name} ↔ {tname}: deep rivalry (bond {rel.net_bond():.2f})")

    if action == "speak" and phrase:
        logger.info(f"  💬 {agent.name} → {tname}: \"{phrase}\"")
        world.log_event({
            "type": "speech",
            "agent": agent.name,
            "agent_id": agent.id,
            "target": tname,
            "target_id": target_id,
            "phrase": phrase,
        })
    elif action == "confront":
        logger.info(f"  ⚡ {agent.name} confronted {tname}")
        world.log_event({
            "type": "confrontation",
            "agent": agent.name,
            "agent_id": agent.id,
            "target": tname,
            "target_id": target_id,
        })
    elif action == "trade":
        logger.info(f"  🔄 {agent.name} offered trade to {tname}")
        world.log_event({
            "type": "trade",
            "agent": agent.name,
            "agent_id": agent.id,
            "target": tname,
            "target_id": target_id,
        })
    elif action == "retreat":
        logger.info(f"  ↩  {agent.name} retreated from {tname}")
        world.log_event({"type": "retreat", "agent": agent.name, "agent_id": agent.id})
    elif action == "rest":
        logger.info(f"  😴 {agent.name} is resting")
    elif action == "observe":
        logger.info(f"  👁  {agent.name} observing {tname or '...'}")
    elif action == "wander":
        logger.info(f"  〰  {agent.name} wandering ({dx:+.1f}, {dy:+.1f})")
    elif action == "ignore":
        logger.info(f"  —  {agent.name} ignoring {tname}")

    mood_change = abs(agent.mood - old_mood)
    if mood_change > 0.15:
        direction = "↑" if agent.mood > old_mood else "↓"
        logger.info(f"  {direction} {agent.name} mood: {old_mood:.2f} → {agent.mood:.2f}")