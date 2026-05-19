"""
router.py — Dynamic priority queue dispatch.

Every tick:
1. Score all agents by urgency
2. Fire top N in parallel (N = safe call count for tick interval)
3. Agents not called get smart rule-based fallback
4. Crisis agents always jump the queue
5. Rotation ensures no agent is starved
"""

import asyncio
import logging
import random
import math
from openai import AsyncOpenAI
from agents.agent import Agent
from core.world import World
from llm.prompt import build_agent_prompt
from llm.parser import parse_agent_response
from config import CEREBRAS_API_KEYS, CEREBRAS_BASE_URL, MODEL, TICK_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

MAX_RETRIES = 1
RETRY_DELAY = 2.0

# Rate limit: 5 req/min per key
REQUESTS_PER_MIN_PER_KEY = 5


def _max_calls_this_tick() -> int:
    """
    How many individual LLM calls can we safely fire this tick?
    Based on tick interval and number of keys.
    """
    total_req_per_min = REQUESTS_PER_MIN_PER_KEY * len(CEREBRAS_API_KEYS)
    safe_per_tick = math.floor(total_req_per_min * (TICK_INTERVAL_SECONDS / 60))
    # Cap at total agent count, floor at 1
    return max(1, min(safe_per_tick, 20))


# ── Client pool ───────────────────────────────────────────────────────────────

_clients: list[AsyncOpenAI] = []

def _get_clients() -> list[AsyncOpenAI]:
    global _clients
    if not _clients:
        _clients = [
            AsyncOpenAI(api_key=key, base_url=CEREBRAS_BASE_URL)
            for key in CEREBRAS_API_KEYS
        ]
        logger.info(f"  Initialized {len(_clients)} API clients")
    return _clients


# ── Priority scoring ──────────────────────────────────────────────────────────

def _urgency_score(agent: Agent, world: World) -> float:
    score = 0.0
    needs = agent.needs

    # Crisis needs — always jump the queue
    if needs.hunger > 0.8:     score += 10.0
    if needs.thirst > 0.8:     score += 10.0
    if needs.energy < 0.15:    score += 8.0
    if needs.health < 0.3:     score += 9.0

    # High needs (not crisis but urgent)
    score += needs.hunger * 2.0
    score += needs.thirst * 2.0
    score += needs.loneliness * 1.0
    score += needs.anger * 1.5

    # Known people nearby → richer decisions possible
    nearby = world.nearby_agents(agent)
    for other in nearby:
        rel = agent.get_rel(other.id)
        if rel.encounters > 0:
            # Strong relationships (positive or negative) = more interesting
            score += 0.5 + abs(rel.bond_score()) * 1.5
        if other.needs.dominant() in ("hunger", "thirst"):
            # Someone nearby is suffering → empathic agents care
            score += agent.traits.get("empathy", 0.5) * 0.8

    # Starvation penalty — agents not called recently rise in priority
    last_llm = getattr(agent, "last_llm_tick", 0)
    ticks_since = world.tick_number - last_llm
    score += ticks_since * 1.2   # linear growth ensures rotation

    # Small jitter to break ties
    score += random.uniform(0, 0.5)

    return score


def _select_agents(world: World) -> tuple[list[Agent], list[Agent]]:
    """
    Returns (llm_agents, fallback_agents).
    llm_agents are the top N by urgency score.
    """
    max_calls = _max_calls_this_tick()
    alive = [a for a in world.agents.values() if a.alive]

    scored = sorted(alive, key=lambda a: _urgency_score(a, world), reverse=True)
    llm_agents = scored[:max_calls]
    fallback_agents = scored[max_calls:]

    return llm_agents, fallback_agents


# ── Fallback behavior (rule-based, instant) ───────────────────────────────────

def _fallback_action(agent: Agent, world: World) -> dict:
    needs = agent.needs
    nearby_resources = world.nearby_resources(agent)
    nearby_agents = world.nearby_agents(agent)

    def toward(tx, ty):
        dx, dy = tx - agent.x, ty - agent.y
        dist = max(0.1, (dx**2 + dy**2) ** 0.5)
        return (dx / dist) * 1.5, (dy / dist) * 1.5

    base = {
        "target_id": None, "phrase": None, "mood_delta": 0,
        "resource_name": None, "give_items": {}, "receive_items": {},
        "craft_a": None, "craft_b": None, "memory_note": None, "rel_updates": {}
    }

    # Crisis: starving → seek food
    if needs.hunger > 0.75:
        food = next((r for r in nearby_resources if r.kind in ("berries", "fish") and r.amount > 0), None)
        if food:
            dx, dy = toward(food.x, food.y)
            return {**base, "action": "forage", "resource_name": food.name, "dx": dx, "dy": dy}
        # No food nearby — move toward nearest food source
        nearest = world.nearest_resource(agent, "berries") or world.nearest_resource(agent, "fish")
        if nearest:
            dx, dy = toward(nearest.x, nearest.y)
            return {**base, "action": "wander", "dx": dx, "dy": dy}

    # Crisis: dehydrated → seek water
    if needs.thirst > 0.75:
        water = next((r for r in nearby_resources if r.kind == "water" and r.amount > 0), None)
        if water:
            dx, dy = toward(water.x, water.y)
            return {**base, "action": "forage", "resource_name": water.name, "dx": dx, "dy": dy}
        nearest = world.nearest_resource(agent, "water")
        if nearest:
            dx, dy = toward(nearest.x, nearest.y)
            return {**base, "action": "wander", "dx": dx, "dy": dy}

    # Exhausted → rest
    if needs.energy < 0.2:
        return {**base, "action": "rest", "dx": 0, "dy": 0, "mood_delta": 0.05}

    # Lonely → drift toward nearest agent
    if needs.loneliness > 0.7 and nearby_agents:
        closest = min(nearby_agents, key=lambda a: (a.x - agent.x)**2 + (a.y - agent.y)**2)
        dx, dy = toward(closest.x, closest.y)
        return {**base, "action": "wander", "dx": dx, "dy": dy}

    # Default: wander
    return {**base, "action": "wander",
            "dx": random.uniform(-1.0, 1.0),
            "dy": random.uniform(-1.0, 1.0)}


# ── Single agent LLM call ─────────────────────────────────────────────────────

async def _call_agent(agent: Agent, world: World, client: AsyncOpenAI) -> tuple[str, dict | None]:
    prompt = build_agent_prompt(agent, world)
    nearby = world.nearby_agents(agent)
    valid_target_ids = [a.id for a in nearby]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=350,
                response_format={"type": "json_object"},
            )
            raw = (response.choices[0].message.content or "").strip()

            if not raw:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return agent.id, None

            result = parse_agent_response(raw, agent.id, valid_target_ids)
            agent.last_llm_tick = world.tick_number
            return agent.id, result

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"  Rate limit for {agent.name}, waiting {wait:.0f}s")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(wait)
                    continue
            else:
                logger.error(f"  LLM error for {agent.name}: {e}")
            return agent.id, None

    return agent.id, None


# ── Main tick entry point ─────────────────────────────────────────────────────

async def run_tick(world: World) -> list[tuple[str, dict | None]]:
    """
    Dynamically select top N agents by urgency.
    Fire all N in parallel, rotating across API keys.
    Remaining agents get instant rule-based fallback.
    """
    clients = _get_clients()
    max_calls = _max_calls_this_tick()
    llm_agents, fallback_agents = _select_agents(world)

    logger.info(f"  → LLM calls: {max_calls} (tick={TICK_INTERVAL_SECONDS}s, keys={len(clients)})")
    logger.info(f"  → LLM: {[a.name for a in llm_agents]}")
    if fallback_agents:
        logger.info(f"  → Fallback: {[a.name for a in fallback_agents]}")

    # Fire all LLM agents in parallel, rotating keys
    tasks = [
        _call_agent(agent, world, clients[i % len(clients)])
        for i, agent in enumerate(llm_agents)
    ]
    llm_results = await asyncio.gather(*tasks)

    # Instant fallback for the rest
    fallback_results = [
        (agent.id, _fallback_action(agent, world))
        for agent in fallback_agents
    ]

    return list(llm_results) + fallback_results