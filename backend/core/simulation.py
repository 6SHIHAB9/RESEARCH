import asyncio
import logging
import time
from core.world import World
from llm.router import run_tick
from agents.agent import Agent
from config import TICK_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

# WebSocket broadcast callback (set by api/websocket.py at startup)
_broadcast_callback = None

def set_broadcast(callback):
    global _broadcast_callback
    _broadcast_callback = callback


def apply_results(world: World, results: list[tuple[str, dict | None]]):
    """Apply all agent decisions to world state."""
    for agent_id, result in results:
        agent = world.agents.get(agent_id)
        if not agent or not result:
            if agent:
                # Fallback: just wander
                agent.move(0.5, 0.5)
                agent.last_action = "wandering"
            continue

        action       = result["action"]
        target_id    = result["target_id"]
        phrase       = result.get("phrase")
        dx           = result["dx"]
        dy           = result["dy"]
        mood_delta   = result["mood_delta"]
        resource_name= result.get("resource_name")
        give_items   = result.get("give_items", {})
        receive_items= result.get("receive_items", {})
        craft_a      = result.get("craft_a")
        craft_b      = result.get("craft_b")
        memory_note  = result.get("memory_note")
        rel_updates  = result.get("rel_updates", {})

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
            success = world.execute_trade(agent, target, give_items, receive_items)
            if success:
                logger.info(f"  🔄 {agent.name} traded {give_items} to {target.name} for {receive_items}")
        elif action == "give" and target and give_items:
            success = world.execute_trade(agent, target, give_items, {})
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
            agent.recent_phrases = getattr(agent, 'recent_phrases', [])
            agent.recent_phrases.append(phrase)

        # ── Relationships ─────────────────────────────────────────────────────
        for rel_target_id, updates in rel_updates.items():
            if rel_target_id in world.agents:
                agent.update_rel(rel_target_id,
                    trust   = float(updates.get("trust", 0)),
                    love    = float(updates.get("love", 0)),
                    rivalry = float(updates.get("rivalry", 0)),
                    fear    = float(updates.get("fear", 0)),
                    debt    = float(updates.get("debt", 0)),
                )

        # ── Memory ────────────────────────────────────────────────────────────
        if memory_note:
            agent.remember(world.tick_number, memory_note)

        agent.last_action = action

        # ── Logging ───────────────────────────────────────────────────────────
        _log_action(world, agent, action, phrase, target, dx, dy)


def _log_action(world, agent, action, phrase, target, dx, dy):
    tname = target.name if target else ""
    tick = world.tick_number

    if action == "speak" and phrase:
        logger.info(f"  💬 {agent.name} → {tname or '(alone)'}: \"{phrase}\"")
        world.log("speech", {"agent": agent.name, "agent_id": agent.id,
                              "target": tname, "target_id": target.id if target else None,
                              "phrase": phrase})
    elif action == "confront":
        logger.info(f"  ⚡ {agent.name} confronted {tname}")
        world.log("confrontation", {"agent": agent.name, "agent_id": agent.id,
                                     "target": tname})
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

    # Relationship milestones
    if target:
        rel = agent.get_rel(target.id)
        if rel.encounters > 0 and rel.encounters % 5 == 0:
            bond = rel.bond_score()
            label = rel.label()
            if abs(bond) > 0.3:
                emoji = "🤝" if bond > 0 else "⚔️"
                logger.info(f"  {emoji} {agent.name} ↔ {tname}: {label} (bond={bond:.2f}, encounters={rel.encounters})")


async def simulation_loop(world: World):
    """Main tick loop."""
    logger.info("🌍 Civilization awakening...")
    logger.info(f"⚙️  Tick interval: {TICK_INTERVAL_SECONDS}s | Agents: {len(world.agents)}")

    while True:
        tick_start = time.time()
        world.tick_number += 1

        logger.info(f"━━━ TICK {world.tick_number} ━━━")

        # 1. Passive decay
        world.tick_needs()
        world.tick_resources()
        world.tick_social_status()

        # 2. LLM decisions (all agents, parallel groups)
        try:
            results = await run_tick(world)
            apply_results(world, results)
        except Exception as e:
            logger.error(f"  Tick error: {e}")

        # 3. Broadcast to UI
        if _broadcast_callback:
            try:
                snapshot = world.to_snapshot()
                await _broadcast_callback(snapshot)
            except Exception as e:
                logger.error(f"  Broadcast error: {e}")

        elapsed = time.time() - tick_start
        logger.info(f"✓ Tick {world.tick_number} complete ({elapsed:.1f}s)")

        # Wait for next tick
        sleep_time = max(0, TICK_INTERVAL_SECONDS - elapsed)
        await asyncio.sleep(sleep_time)
