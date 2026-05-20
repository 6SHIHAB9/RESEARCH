"""
router.py — Global rate limiter for Cerebras free tier.

KEY INSIGHT (learned from 429 logs):
  The 5 req/min limit is enforced GLOBALLY across your Cerebras account,
  not per API key independently. 3 keys × 5 req/min does NOT give you
  15 req/min — all 3 keys share one rate limit bucket server-side.
  Using multiple keys only helps if they are on SEPARATE Cerebras accounts.

  Evidence: Key 0 fires at 12:03:39, Key 1 fires at 12:03:42 (3s later)
  and immediately 429s — even though Key 1 had never been used.

Correct strategy:
  - Treat ALL keys as drawing from ONE shared pool: 5 req/min = 1 per 12s
  - Use a single global asyncio.Lock + timestamp — one request in-flight
    at a time, globally, with 12s between completions
  - Keys are rotated round-robin only for load-balancing / redundancy,
    not for throughput
  - A factory function (make_coro) is passed to the global gate instead
    of a pre-built coroutine, so each attempt creates a fresh coroutine
    (fixes "coroutine never awaited" RuntimeWarning on retry)

With 12 agents at 1 req/12s, a full tick takes ~144s minimum.
Set TICK_INTERVAL_SECONDS=150 or higher, or reduce agent count.
"""

import asyncio
import collections
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

# ── Rate limit constants ──────────────────────────────────────────────────────

REQUESTS_PER_MIN  = 5    # advertised limit — but Cerebras enforces stricter in practice
REQUESTS_PER_HOUR = 150  # global account limit
WINDOW_SECONDS    = 60.0 # rolling window
PER_MIN_CAP       = 2    # hard limit in practice: never send a 4th req within 60s of the 1st
HOURLY_CAP        = 140  # leave 10 req headroom on the 150/hr limit


# ── Global gate (one request in-flight at a time, account-wide) ──────────────
#
# Strategy: rolling-window tracking, not fixed-interval spacing.
#
# Fixed spacing (every 12s) fails because:
#   - Response latency varies (1-4s), so "12s between responses" means
#     "8-11s between dispatches" — requests accumulate inside the 60s window.
#   - After 4 requests the 5th fires while all 4 are still in the window.
#
# Rolling-window fix: before firing, count how many requests landed in the
# last 60s. If that count is >= PER_MIN_CAP, sleep until the oldest one
# is >60s old. This guarantees we never exceed the cap regardless of latency.

_global_lock: asyncio.Lock | None = None
_request_dispatch_times: collections.deque = collections.deque()  # all request timestamps


def _get_global_lock() -> asyncio.Lock:
    global _global_lock
    if _global_lock is None:
        _global_lock = asyncio.Lock()
    return _global_lock


def _count_in_window(window: float) -> int:
    """Count requests dispatched within the last `window` seconds."""
    cutoff = time.monotonic() - window
    while _request_dispatch_times and _request_dispatch_times[0] < cutoff:
        _request_dispatch_times.popleft()
    return len(_request_dispatch_times)


# Minimum spacing between dispatches.
# 60s / 3 req = 20s. This is the primary throttle — keeps us well under
# any window interpretation Cerebras uses (fixed-minute or rolling).
MIN_DISPATCH_GAP = 31.0  # 60s / 3 = 20s + 1s buffer

_last_dispatch_time: float = 0.0   # monotonic; set to now() on first init


async def _global_call(make_coro):
    """
    Serialise every API request through one global gate.

    Two-layer protection:
      1. Rolling window: never have >= PER_MIN_CAP requests in last 60s.
      2. Minimum gap: never dispatch two requests < MIN_DISPATCH_GAP apart.

    Both are needed — the rolling count alone allows bursts of 4 rapid
    requests (all under cap) which exhaust the quota instantly.

    `make_coro` is a zero-argument callable returning a fresh coroutine
    each invocation (coroutines can only be awaited once).
    """
    global _last_dispatch_time

    lock = _get_global_lock()
    async with lock:
        # ── Minimum inter-dispatch gap ───────────────────────────────────
        gap_wait = (_last_dispatch_time + MIN_DISPATCH_GAP) - time.monotonic()
        if gap_wait > 0:
            await asyncio.sleep(gap_wait)

        # ── Rolling per-minute guard ─────────────────────────────────────
        while _count_in_window(WINDOW_SECONDS) >= PER_MIN_CAP:
            oldest = _request_dispatch_times[0]
            sleep_for = (oldest + WINDOW_SECONDS + 0.5) - time.monotonic()
            if sleep_for > 0:
                logger.debug(
                    f"  Global gate: {_count_in_window(WINDOW_SECONDS)}/{PER_MIN_CAP} "
                    f"req in window — waiting {sleep_for:.1f}s"
                )
                await asyncio.sleep(sleep_for)

        # ── Rolling hourly guard ─────────────────────────────────────────
        while _count_in_window(3600.0) >= HOURLY_CAP:
            oldest = _request_dispatch_times[0]
            sleep_for = (oldest + 3600.0 + 1.0) - time.monotonic()
            if sleep_for > 0:
                logger.warning(
                    f"  Global hourly cap ({_count_in_window(3600.0)}/{HOURLY_CAP}) — "
                    f"sleeping {sleep_for:.0f}s"
                )
                await asyncio.sleep(sleep_for)

        # ── Fire — hold lock for entire round-trip ───────────────────────
        _last_dispatch_time = time.monotonic()
        _request_dispatch_times.append(_last_dispatch_time)
        try:
            return await make_coro()
        except Exception:
            raise


def _penalise():
    """
    After a 429, block the gate for a full minute by:
      1. Inserting PER_MIN_CAP synthetic future timestamps so the rolling
         window stays full for WINDOW_SECONDS.
      2. Pushing _last_dispatch_time forward so the gap guard also waits.
    """
    global _last_dispatch_time
    future = time.monotonic() + WINDOW_SECONDS
    # Fill the window with synthetic entries so count stays at cap for 60s.
    for _ in range(PER_MIN_CAP):
        _request_dispatch_times.append(future)
    _last_dispatch_time = future
    logger.warning(
        f"  Global gate: 429 — full cooldown, gate blocked ~{WINDOW_SECONDS:.0f}s"
    )


# ── Client pool (round-robin for redundancy, not throughput) ─────────────────

_clients: list[AsyncOpenAI] = []
_client_index: int = 0


def _get_clients() -> list[AsyncOpenAI]:
    global _clients
    if not _clients and CEREBRAS_API_KEYS:
        global _last_dispatch_time
        # Treat server start as if a request just fired — first real request
        # will wait the full MIN_DISPATCH_GAP, preventing a cold-start burst.
        _last_dispatch_time = time.monotonic()
        _clients = [
            AsyncOpenAI(api_key=key, base_url=CEREBRAS_BASE_URL, max_retries=0)
            for key in CEREBRAS_API_KEYS
        ]
        logger.info(
            f"  Initialized {len(_clients)} API clients | "
            f"GLOBAL gate: {REQUESTS_PER_MIN} req/min, {HOURLY_CAP} req/hr | "
            f"rolling window: {PER_MIN_CAP} req/{WINDOW_SECONDS:.0f}s | global lock per round-trip"
        )
    return _clients


def _next_client() -> AsyncOpenAI:
    """Round-robin across keys for redundancy."""
    global _client_index
    clients = _get_clients()
    client = clients[_client_index % len(clients)]
    _client_index += 1
    return client


# ── Priority scoring ──────────────────────────────────────────────────────────

def _urgency_score(agent: Agent, world: World) -> float:
    score = 0.0
    needs = agent.needs

    if needs.hunger > 0.8:   score += 10.0
    if needs.thirst > 0.8:   score += 10.0
    if needs.energy < 0.15:  score += 8.0
    if needs.health < 0.3:   score += 9.0

    score += needs.hunger    * 2.0
    score += needs.thirst    * 2.0
    score += needs.loneliness * 1.0
    score += needs.anger     * 1.5

    nearby = world.nearby_agents(agent)
    for other in nearby:
        rel = agent.get_rel(other.id)
        if rel.encounters > 0:
            score += 0.5 + abs(rel.bond_score()) * 1.5
        if other.needs.dominant() in ("hunger", "thirst"):
            score += agent.traits.get("empathy", 0.5) * 0.8

    last_llm = getattr(agent, "last_llm_tick", 0)
    score += (world.tick_number - last_llm) * 1.2
    score += random.uniform(0, 0.5)
    return score


# ── Fallback behavior ─────────────────────────────────────────────────────────

def _fallback_action(agent: Agent, world: World) -> dict:
    needs = agent.needs
    nearby_resources = world.nearby_resources(agent)
    nearby_agents    = world.nearby_agents(agent)

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

async def _call_agent(agent: Agent, world: World) -> tuple[str, dict | None]:
    prompt = build_agent_prompt(agent, world)
    nearby = world.nearby_agents(agent)
    valid_target_ids = [a.id for a in nearby]

    for attempt in range(2):
        client = _next_client()

        # Pass a factory lambda — _global_call invokes it fresh each time,
        # so no "coroutine never awaited" warnings on retry.
        def make_coro(c=client, p=prompt):
            return c.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": p}],
                temperature=1.0,
                max_tokens=350,
                response_format={"type": "json_object"},
            )

        try:
            response = await _global_call(make_coro)
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                return agent.id, None

            result = parse_agent_response(raw, agent.id, valid_target_ids)
            agent.last_llm_tick = world.tick_number
            return agent.id, result

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                _penalise()
                if attempt == 0:
                    logger.warning(f"  {agent.name}: 429 on attempt 1 — retrying after cooldown")
                    continue
                logger.warning(f"  {agent.name}: 429 after retry — using fallback")
            else:
                logger.error(f"  LLM error for {agent.name}: {e}")
            return agent.id, None

    return agent.id, None


# ── Main tick entry point ─────────────────────────────────────────────────────

async def run_tick(world: World):
    """
    Async generator — yields (agent_id, result) one at a time as each
    LLM call completes, so the caller can apply + broadcast immediately
    rather than waiting for all 12 agents to finish first.

    Throughput: 1 req / 12s = 5 req/min (true account-wide limit).
    12 agents × 12s = ~144s minimum per tick.
    12 agents × 20s = ~240s minimum per tick.
    Recommended TICK_INTERVAL_SECONDS = 250.
    """
    clients = _get_clients()

    if not clients:
        logger.info("  No API keys — using deterministic fallback for all agents")
        for agent in world.agents.values():
            if agent.alive:
                yield agent.id, _fallback_action(agent, world)
        return

    alive = sorted(
        [a for a in world.agents.values() if a.alive],
        key=lambda a: _urgency_score(a, world),
        reverse=True,
    )

    logger.info(
        f"  → LLM calls: {len(alive)} agents | "
        f"rolling window: {PER_MIN_CAP} req/{WINDOW_SECONDS:.0f}s | "
        f"est. {len(alive) * (WINDOW_SECONDS / PER_MIN_CAP):.0f}s"
    )

    for agent in alive:
        result = await _call_agent(agent, world)
        yield result