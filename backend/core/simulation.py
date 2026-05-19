import asyncio
import logging
import time
from core.world import World
from llm.router import run_tick
from agents.agent import Agent
from config import TICK_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

_broadcast_callback = None

def set_broadcast(callback):
    global _broadcast_callback
    _broadcast_callback = callback


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default



def _sanitize_items(items) -> dict:
    """Ensure items is a flat {str: int} dict. LLM sometimes nests dicts."""
    if not isinstance(items, dict):
        return {}
    result = {}
    for k, v in items.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            result[str(k)] = int(v)
        # skip nested dicts/lists
    return result

def apply_results(world: World, results: list):
    for agent_id, result in results:
        agent = world.agents.get(agent_id)
        if not agent or not result:
            if agent:
                agent.move(0.5, 0.5)
                agent.last_action = "wandering"
            continue

        action        = result.get("action", "wander")
        target_id     = result.get("target_id")
        phrase        = result.get("phrase")
        dx            = _safe_float(result.get("dx"), 0.0)
        dy            = _safe_float(result.get("dy"), 0.0)
        mood_delta    = _safe_float(result.get("mood_delta"), 0.0)
        resource_name = result.get("resource_name")
        give_items    = result.get("give_items") or {}
        receive_items = result.get("receive_items") or {}
        craft_a       = result.get("craft_a")
        craft_b       = result.get("craft_b")
        memory_note   = result.get("memory_note")
        rel_updates   = result.get("rel_updates") or {}

        target = world.agents.get(target_id) if target_id else None

        # ── Movement ──────────────────────────────────────────────────────────
        if action == "retreat" and target:
            flee_dx = agent.x - target.x
            flee_dy = agent.y - target.y
            dist = max(0.1, (flee_dx**2 + flee_dy**2)**0.5)
            dx = (flee_dx / dist) * 2.0
            dy = (flee_dy / dist) * 2.0
        elif action == "speak" and target:
            toward_dx = target.x - agent.x
            toward_dy = target.y - agent.y
            dist = max(0.1, (toward_dx**2 + toward_dy**2)**0.5)
            if dist > 2:
                dx = (toward_dx / dist) * 1.0
                dy = (toward_dy / dist) * 1.0

        agent.move(dx, dy)

        # ── Needs from action ─────────────────────────────────────────────────
        if action == "rest":
            agent.needs.energy = min(1.0, agent.needs.energy + 0.2)
            agent.needs.loneliness = min(1.0, agent.needs.loneliness + 0.05)
        elif action == "speak" and target:
            agent.needs.loneliness = max(0.0, agent.needs.loneliness - 0.15)
        elif action == "forage" and resource_name:
            world.harvest(agent, resource_name)
        elif action == "claim" and resource_name:
            world.claim_resource(agent, resource_name)
        elif action == "craft" and craft_a and craft_b:
            result_item = world.attempt_craft(agent, craft_a, craft_b)
            if result_item:
                logger.info(f"  🔨 {agent.name} crafted {result_item} from {craft_a}+{craft_b}!")
        elif action == "trade" and target and give_items and receive_items:
            give_items = _sanitize_items(give_items)
            receive_items = _sanitize_items(receive_items)
            if give_items and receive_items:
                success = world.execute_trade(agent, target, give_items, receive_items)
                if success:
                    logger.info(f"  🔄 {agent.name} traded {give_items} to {target.name} for {receive_items}")
        elif action == "give" and target and give_items:
            give_items = _sanitize_items(give_items)
            success = world.execute_trade(agent, target, give_items, {}) if give_items else False
            if success:
                agent.update_rel(target_id, trust=0.08, love=0.05)
                target.update_rel(agent_id, trust=0.1, love=0.08, debt=-0.1)
                target.nudge_mood(0.15)
                logger.info(f"  🎁 {agent.name} gave {give_items} to {target.name}")
        elif action == "confront" and target:
            agent.needs.anger = max(0.0, agent.needs.anger - 0.1)
            target.needs.fear = min(1.0, target.needs.fear + 0.2)
            agent.update_rel(target_id, rivalry=0.05)
            target.update_rel(agent_id, fear=0.1, trust=-0.05)

        # ── Mood ──────────────────────────────────────────────────────────────
        agent.nudge_mood(mood_delta)

        # ── Phrase ────────────────────────────────────────────────────────────
        if phrase and action == "speak":
            agent.last_phrase = phrase

        # ── Relationships — safe_float every value ────────────────────────────
        if isinstance(rel_updates, dict):
            for rel_target_id, updates in rel_updates.items():
                if rel_target_id in world.agents and isinstance(updates, dict):
                    agent.update_rel(rel_target_id,
                        trust   = _safe_float(updates.get("trust"),   0.0),
                        love    = _safe_float(updates.get("love"),    0.0),
                        rivalry = _safe_float(updates.get("rivalry"), 0.0),
                        fear    = _safe_float(updates.get("fear"),    0.0),
                        debt    = _safe_float(updates.get("debt"),    0.0),
                    )

        # ── Memory ────────────────────────────────────────────────────────────
        if memory_note:
            agent.remember(world.tick_number, str(memory_note))

        agent.last_action = action
        _log_action(world, agent, action, phrase, target, dx, dy)


def _log_action(world, agent, action, phrase, target, dx, dy):
    tname = target.name if target else ""

    if action == "speak" and phrase:
        logger.info(f"  💬 {agent.name} → {tname or '(alone)'}: \"{phrase}\"")
        world.log("speech", {"agent": agent.name, "agent_id": agent.id,
                              "target": tname, "target_id": target.id if target else None,
                              "phrase": phrase})
    elif action == "confront":
        logger.info(f"  ⚡ {agent.name} confronted {tname}")
        world.log("confrontation", {"agent": agent.name, "agent_id": agent.id, "target": tname})
    elif action == "retreat":
        logger.info(f"  ↩  {agent.name} retreated from {tname or 'threat'}")
    elif action == "rest":
        logger.info(f"  😴 {agent.name} resting")
    elif action == "observe":
        logger.info(f"  👁  {agent.name} watching {tname or '...'}")
    elif action == "forage":
        logger.info(f"  🌿 {agent.name} foraging")
    elif action == "wander":
        logger.info(f"  〰  {agent.name} wandering ({dx:+.1f}, {dy:+.1f})")
    elif action == "ignore":
        logger.info(f"  —  {agent.name} ignoring {tname}")
    elif action == "give":
        logger.info(f"  🎁 {agent.name} giving to {tname}")
    elif action == "trade":
        logger.info(f"  🔄 {agent.name} trading with {tname}")
    elif action == "claim":
        logger.info(f"  🚩 {agent.name} claiming territory")
    elif action == "craft":
        logger.info(f"  🔨 {agent.name} crafting")

    if target:
        rel = agent.get_rel(target.id)
        if rel.encounters > 0 and rel.encounters % 5 == 0:
            bond = rel.bond_score()
            if abs(bond) > 0.3:
                emoji = "🤝" if bond > 0 else "⚔️"
                logger.info(f"  {emoji} {agent.name} ↔ {tname}: {rel.label()} (bond={bond:.2f}, enc={rel.encounters})")


async def simulation_loop(world: World):
    logger.info("🌍 Civilization awakening...")
    logger.info(f"⚙️  Tick interval: {TICK_INTERVAL_SECONDS}s | Agents: {len(world.agents)}")

    while True:
        tick_start = time.time()
        world.tick_number += 1
        logger.info(f"━━━ TICK {world.tick_number} ━━━")

        world.tick_needs()
        world.tick_resources()
        world.tick_social_status()

        try:
            results = await run_tick(world)
            apply_results(world, results)
        except Exception as e:
            logger.error(f"  Tick error: {e}", exc_info=True)

        if _broadcast_callback:
            try:
                await _broadcast_callback(world.to_snapshot())
            except Exception as e:
                logger.error(f"  Broadcast error: {e}")

        elapsed = time.time() - tick_start
        logger.info(f"✓ Tick {world.tick_number} complete ({elapsed:.1f}s)")
        await asyncio.sleep(max(0, TICK_INTERVAL_SECONDS - elapsed))