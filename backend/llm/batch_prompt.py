import json
import logging
import random
import os
from openai import AsyncOpenAI
from core.world import World, WORLD_WIDTH, WORLD_HEIGHT, Landmark
from agents.agent import Agent

logger = logging.getLogger(__name__)

def _get_client() -> AsyncOpenAI:
    api_key = os.environ.get("CEREBRAS_API_KEY", "placeholder")
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )

MODEL = "llama3.1-8b"
MAX_AGENTS_PER_BATCH = 8   


def _build_agent_context(agent: Agent, nearby: list[Agent], nearby_landmarks: list[Landmark]) -> dict:
    """Build a compact context object for one agent, including local geography."""
    memory_summary = []
    for m in agent.memory[-3:]:
        memory_summary.append(m.get("summary", ""))

    nearby_info = []
    for other in nearby[:4]: 
        bond = agent.social_bonds.get(other.id, 0.0)
        bond_label = "familiar" if bond > 0.3 else ("tense" if bond < -0.2 else "unknown")
        nearby_info.append(f"{other.name} ({other.mood_label()}, {bond_label})")

    landmark_info = [{"name": lm.name, "kind": lm.kind} for lm in nearby_landmarks]

    return {
        "id": agent.id,
        "name": agent.name,
        "personality": agent.personality,
        "mood": agent.mood_label(),
        "nearby_agents": nearby_info,
        "nearby_landmarks": landmark_info,
        "memories": memory_summary,
        "last_action": agent.last_action,
    }


def _build_batch_prompt(agent_contexts: list[dict]) -> str:
    agents_json = json.dumps(agent_contexts, indent=2)

    return f"""You are simulating a small group of beings in a persistent world with local geography.
Each being simply exists. They have no goals. They are not performing for anyone.

For each being below, decide what happens in this moment.
Consider their personality, mood, local geography (landmarks), nearby beings, and memories.

Respond ONLY with a valid JSON array. One object per being, same order as input.
Each object must have exactly these fields:
- "id": the being's id (copy from input)
- "action": one of: "speak", "observe", "wander", "ignore", "retreat", "linger"
- "phrase": a short utterance if action is "speak", otherwise null. Max 12 words. Natural, not poetic.
- "mood_delta": float from -0.3 to 0.3, how this moment shifts their mood
- "target_id": id of nearby being they interact with (if any), otherwise null
- "bond_delta": float -0.2 to 0.2, change in bond with target (0 if no target)
- "move_dx": float -1.5 to 1.5, horizontal drift this tick (smaller movements to prevent spreading)
- "move_dy": float -1.5 to 1.5, vertical drift this tick
- "memory_note": one short sentence (under 15 words) to remember, or null

Rules:
- Most beings should do very little. Silence and inactivity are normal.
- Movement Ecology: Movement is stochastic, but beings occasionally drift toward nearby landmarks or familiar agents with stronger social bonds.
- Gatherings emerge naturally over time through repeated proximity, not forced coordination.
- Beings only perceive their immediate surroundings. They possess incomplete information.
- Do not invent drama, excessive dialogue, or assistant-like behavior. The world should feel like an ambient, sparse ecosystem.

Beings:
{agents_json}

Respond with ONLY the JSON array. No explanation. No markdown.
"""


async def run_tick_batch(world: World):
    """Run one batched LLM tick for the world."""
    all_agents = list(world.agents.values())
    random.shuffle(all_agents)
    active_agents = all_agents[:MAX_AGENTS_PER_BATCH]

    agent_contexts = []
    for agent in active_agents:
        nearby = world.get_nearby_agents(agent)
        nearby_landmarks = world.get_nearby_landmarks(agent)
        ctx = _build_agent_context(agent, nearby, nearby_landmarks)
        agent_contexts.append(ctx)

    prompt = _build_batch_prompt(agent_contexts)

    logger.info(f"  → Sending batch of {len(agent_contexts)} agents to Cerebras...")

    client = _get_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.15,
        max_tokens=1200,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    try:
        results = json.loads(raw)
        _apply_results(world, results, active_agents)
    except json.JSONDecodeError as e:
        logger.error(f"  ✗ JSON parse error: {e}")
        for agent in active_agents:
            agent.move(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
            agent.last_action = "wandering"


def _apply_results(world: World, results: list[dict], active_agents: list[Agent]):
    """Apply LLM decisions back to world state."""
    agent_map = {a.id: a for a in active_agents}

    for result in results:
        agent_id = result.get("id")
        agent = agent_map.get(agent_id)
        if not agent:
            continue

        action = result.get("action", "wander")
        phrase = result.get("phrase")
        mood_delta = float(result.get("mood_delta", 0))
        target_id = result.get("target_id")
        bond_delta = float(result.get("bond_delta", 0))
        move_dx = float(result.get("move_dx", random.uniform(-1, 1)))
        move_dy = float(result.get("move_dy", random.uniform(-1, 1)))
        memory_note = result.get("memory_note")

        # Capture old coordinates for landmark delta calculation
        old_x, old_y = agent.x, agent.y

        # Apply movement
        agent.move(move_dx, move_dy)

        # Apply mood shift
        old_mood = agent.mood
        agent.nudge_mood(mood_delta)

        # Store phrase
        if phrase:
            agent.recent_phrases.append(phrase)
            if len(agent.recent_phrases) > 5:
                agent.recent_phrases = agent.recent_phrases[-5:]

        # Apply bond change
        if target_id and bond_delta != 0:
            agent.update_bond(target_id, bond_delta)
            target = world.agents.get(target_id)
            if target:
                target.update_bond(agent_id, bond_delta * 0.6)

        # Store memory
        if memory_note:
            agent.add_memory({
                "tick": world.tick_number,
                "summary": memory_note,
            })

        agent.last_action = action
        agent.last_interaction_tick = world.tick_number

        mood_change = abs(agent.mood - old_mood)

        # ==================================================
        # 8. LIGHTWEIGHT TERMINAL LOGGING (Ecology & Geography)
        # ==================================================
        for lm in world.landmarks:
            dist = ((agent.x - lm.x)**2 + (agent.y - lm.y)**2)**0.5
            old_dist = ((old_x - lm.x)**2 + (old_y - lm.y)**2)**0.5
            
            if dist <= 5.0 and old_dist > 5.0:
                logger.info(f"  📍 {agent.name} drifted toward the {lm.name}")
            elif dist <= 5.0 and action == "linger":
                logger.info(f"  🏕️  {agent.name} is lingering near the {lm.name}")

        if target_id:
            target = world.agents.get(target_id)
            if target and agent.social_bonds.get(target_id, 0.0) > 0.4:
                logger.info(f"  👥 {agent.name} and {target.name} share a familiar recurring encounter")

        # Standard Logging
        if action == "speak" and phrase:
            logger.info(f"  💬 {agent.name}: \"{phrase}\"")
            world.log_event({
                "type": "speech",
                "agent": agent.name,
                "agent_id": agent.id,
                "target_id": target_id,
                "phrase": phrase,
            })
        elif action == "retreat":
            logger.info(f"  ↩  {agent.name} retreated")
            world.log_event({"type": "retreat", "agent": agent.name, "agent_id": agent.id})
        elif action == "observe":
            target = world.agents.get(target_id) if target_id else None
            t = f" → {target.name}" if target else ""
            logger.info(f"  👁  {agent.name} observing{t}")
        elif action == "wander":
            logger.info(f"  〰  {agent.name} wandering ({move_dx:+.1f}, {move_dy:+.1f})")
        elif action == "ignore":
            target = world.agents.get(target_id) if target_id else None
            t = f" {target.name}" if target else ""
            logger.info(f"  —  {agent.name} ignoring{t}")
        elif action == "linger":
            pass # Pre-handled by landmark log to prevent double-logging
        elif mood_change > 0.15:
            direction = "↑" if mood_delta > 0 else "↓"
            logger.info(f"  {direction} {agent.name} mood: {old_mood:.2f} → {agent.mood:.2f}")

        if bond_delta > 0.1 and target_id:
            target = world.agents.get(target_id)
            if target:
                logger.info(f"  🤝 {agent.name} → {target.name} bond: +{bond_delta:.2f}")