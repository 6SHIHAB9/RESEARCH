"""
router.py — Token bucket dispatch. Zero 429s.

Every tick:
1. Score all agents by urgency
2. Each API key has its own token bucket (5 req/min)
3. Calls are queued and dispatched only when a bucket has a token
4. All agents get LLM every tick — no fallback needed unless keys missing
5. Rotation ensures no key is overloaded
"""

import asyncio
import logging
import random
import time
from openai import AsyncOpenAI
from agents.agent import Agent
from core.world import World
from llm.prompt import build_agent_prompt
from llm.parser import parse_agent_response
from config import CEREBRAS_API_KEYS, CEREBRAS_BASE_URL, MODEL, TICK_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

# Rate limit per key — 5 req/min = 1 req per 12s
REQUESTS_PER_MIN_PER_KEY = 5
MIN_INTERVAL_PER_KEY = 60.0 / REQUESTS_PER_MIN_PER_KEY  # 12.0 seconds


# ── Token bucket (one per API key) ────────────────────────────────────────────

class KeyBucket:
    """
    Strictly throttles one API key to REQUESTS_PER_MIN_PER_KEY req/min.
    Callers await acquire() which resolves only when a slot is available.
    """
    def __init__(self, client: AsyncOpenAI, key_index: int):
        self.client = client
        self.key_index = key_index
        self._lock = asyncio.Lock()
        self._last_call_time: float = 0.0

    async def acquire(self):
        """Wait until this key can safely fire another request."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            wait = MIN_INTERVAL_PER_KEY - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_time = time.monotonic()


# ── Client + bucket pool ──────────────────────────────────────────────────────

_buckets: list[KeyBucket] = []

def _get_buckets() -> list[KeyBucket]:
    global _buckets
    if not CEREBRAS_API_KEYS:
        return []
    if not _buckets:
        _buckets = [
            KeyBucket(
                client=AsyncOpenAI(api_key=key, base_url=CEREBRAS_BASE_URL),
                key_index=i,
            )
            for i, key in enumerate(CEREBRAS_API_KEYS)
        ]
        logger.info(f"  Initialized {len(_buckets)} API clients with token buckets "
                    f"({REQUESTS_PER_MIN_PER_KEY} req/min each)")
    return _buckets


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
            score += 0.5 + abs(rel.bond_score()) * 1.5
        if other.needs.dominant() in ("hunger", "thirst"):
            score += agent.traits.get("empathy", 0.5) * 0.8

    # Starvation penalty — agents not called recently rise in priority
    last_llm = getattr(agent, "last_llm_tick", 0)
    ticks_since = world.tick_number - last_llm
    score += ticks_since * 1.2

    # Small jitter to break ties
    score += random.uniform(0, 0.5)

    return score


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
        "craft_a": None, "craft_b": None, "project_id": None,
        "memory_note": None, "rel_updates": {}
    }

    if needs.hunger > 0.75:
        food = next((r for r in nearby_resources if r.kind in ("berries", "fish") and r.amount > 0), None)
        if food:
            dx, dy = toward(food.x, food.y)
            return {**base, "action": "forage", "resource_name": food.name, "dx": dx, "dy": dy}
        nearest = world.nearest_resource(agent, "berries") or world.nearest_resource(agent, "fish")
        if nearest:
            dx, dy = toward(nearest.x, nearest.y)
            return {**base, "action": "wander", "dx": dx, "dy": dy}

    if needs.thirst > 0.75:
        water = next((r for r in nearby_resources if r.kind == "water" and r.amount > 0), None)
        if water:
            dx, dy = toward(water.x, water.y)
            return {**base, "action": "forage", "resource_name": water.name, "dx": dx, "dy": dy}
        nearest = world.nearest_resource(agent, "water")
        if nearest:
            dx, dy = toward(nearest.x, nearest.y)
            return {**base, "action": "wander", "dx": dx, "dy": dy}

    if needs.energy < 0.2:
        return {**base, "action": "rest", "dx": 0, "dy": 0, "mood_delta": 0.05}

    if (
        agent.home_group
        and agent.repeated_action_count("build", window=4) < 2
        and (agent.inventory.get("wood", 0) > 0 or agent.inventory.get("stone", 0) > 0)
    ):
        project = next(
            (p for p in world.projects if p.group_id == agent.home_group and not p.complete),
            None,
        )
        if project:
            return {**base, "action": "build", "project_id": project.id, "dx": 0, "dy": 0}

    if needs.loneliness > 0.7 and nearby_agents:
        closest = min(nearby_agents, key=lambda a: (a.x - agent.x)**2 + (a.y - agent.y)**2)
        dx, dy = toward(closest.x, closest.y)
        if agent.repeated_action_count("wander", window=4) >= 3:
            return {**base, "action": "observe", "target_id": closest.id, "dx": 0, "dy": 0}
        return {**base, "action": "wander", "target_id": closest.id, "dx": dx, "dy": dy}

    if agent.repeated_action_count("wander", window=4) >= 3 and nearby_agents:
        closest = min(nearby_agents, key=lambda a: (a.x - agent.x)**2 + (a.y - agent.y)**2)
        return {**base, "action": "observe", "target_id": closest.id, "dx": 0, "dy": 0}
    return {**base, "action": "wander",
            "dx": random.uniform(-1.0, 1.0),
            "dy": random.uniform(-1.0, 1.0)}


# ── Single agent LLM call ─────────────────────────────────────────────────────

async def _call_agent(agent: Agent, world: World, bucket: KeyBucket) -> tuple[str, dict | None]:
    prompt = build_agent_prompt(agent, world)
    nearby = world.nearby_agents(agent)
    valid_target_ids = [a.id for a in nearby]

    # Acquire a slot from the token bucket — waits if needed, never 429s
    await bucket.acquire()

    try:
        response = await bucket.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=350,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return agent.id, None

        result = parse_agent_response(raw, agent.id, valid_target_ids)
        agent.last_llm_tick = world.tick_number
        return agent.id, result

    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            # Shouldn't happen with token bucket, but log if it does
            logger.warning(f"  Unexpected 429 for {agent.name} (bucket leak?)")
        else:
            logger.error(f"  LLM error for {agent.name}: {e}")
        return agent.id, None


# ── Main tick entry point ─────────────────────────────────────────────────────

async def run_tick(world: World) -> list[tuple[str, dict | None]]:
    """
    All agents get an LLM call every tick.
    Token buckets ensure we never exceed 5 req/min per key.
    Calls are distributed round-robin across keys and execute
    as soon as their assigned bucket has a slot — no bursting, no 429s.
    """
    buckets = _get_buckets()

    if not buckets:
        logger.info("  No API keys configured; using deterministic fallback for all agents")
        alive = [a for a in world.agents.values() if a.alive]
        return [(agent.id, _fallback_action(agent, world)) for agent in alive]

    alive = sorted(
        [a for a in world.agents.values() if a.alive],
        key=lambda a: _urgency_score(a, world),
        reverse=True,
    )

    logger.info(f"  → LLM calls: {len(alive)} agents across {len(buckets)} keys (token bucket)")

    # Distribute agents round-robin across buckets
    tasks = [
        _call_agent(agent, world, buckets[i % len(buckets)])
        for i, agent in enumerate(alive)
    ]

    results = await asyncio.gather(*tasks)
    return list(results)