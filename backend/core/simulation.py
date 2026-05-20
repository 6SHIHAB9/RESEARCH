import asyncio
import logging
import random
import time

from agents.agent import Agent
from config import TICK_INTERVAL_SECONDS
from core.world import World
from llm.router import run_tick

logger = logging.getLogger(__name__)

_broadcast_callback = None


def set_broadcast(callback):
    global _broadcast_callback
    _broadcast_callback = callback


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_items(items) -> dict:
    """Keep only positive integer item amounts."""
    if not isinstance(items, dict):
        return {}
    result = {}
    for key, value in items.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        amount = int(value)
        if amount > 0:
            result[str(key)] = amount
    return result


def _should_force_variety(agent: Agent, action: str) -> bool:
    # Only throttle actions that cause real problems when spammed
    if action in {"forage", "wander", "trade", "give"}:
        return agent.repeated_action_count(action, window=5) >= 4
    return False


def _ground_phrase(agent: Agent, phrase: str | None, target: Agent | None) -> str | None:
    if not phrase:
        return None

    phrase = " ".join(str(phrase).split())[:180]
    lower = phrase.lower()
    risky_claims = ("stole", "betrayed", "lied", "attacked", "cheated", "took")
    if not any(word in lower for word in risky_claims):
        return phrase

    evidence = " ".join(m.get("event", "") for m in agent.memory[-8:]).lower()
    target_name = target.name.lower() if target else ""
    has_evidence = any(word in evidence for word in risky_claims)
    if target_name:
        has_evidence = has_evidence and target_name in evidence
    if has_evidence:
        return phrase

    return "I am suspicious, but I need proof."


def _center_seeking_wander(agent: Agent) -> tuple[float, float]:
    """
    FIX: When forced to wander (repetition cooldown or boundary),
    steer toward world center (30, 30) with some randomness.
    Prevents agents from getting stuck against boundaries.
    """
    center_x, center_y = 30.0, 30.0
    dx = (center_x - agent.x) * 0.15 + random.uniform(-0.6, 0.6)
    dy = (center_y - agent.y) * 0.15 + random.uniform(-0.6, 0.6)
    return dx, dy


def apply_results(world: World, results: list):
    for agent_id, result in results:
        agent = world.agents.get(agent_id)
        if not agent:
            continue
        if not result:
            # FIX: use center-seeking wander for empty results too
            dx, dy = _center_seeking_wander(agent)
            agent.move(dx, dy)
            agent.last_action = "wander"
            # FIX: don't record fallback wanders in action_history
            # so they don't trigger the repeat cooldown
            world.increment_metric("fallback_empty_result")
            continue

        action = result.get("action", "wander")
        target_id = result.get("target_id")
        phrase = result.get("phrase")
        dx = max(-2.0, min(2.0, _safe_float(result.get("dx"), 0.0)))
        dy = max(-2.0, min(2.0, _safe_float(result.get("dy"), 0.0)))
        mood_delta = _safe_float(result.get("mood_delta"), 0.0)
        resource_name = result.get("resource_name")
        give_items = result.get("give_items") or {}
        receive_items = result.get("receive_items") or {}
        craft_a = result.get("craft_a")
        craft_b = result.get("craft_b")
        project_id = result.get("project_id")
        memory_note = result.get("memory_note")
        rel_updates = result.get("rel_updates") or {}
        target = world.agents.get(target_id) if target_id else None

        if _should_force_variety(agent, action):
            # FIX: silent redirect — don't log as failure, just change the action
            if action == "observe" and target:
                # Try a different target instead of same one again
                other_targets = [
                    a for a in world.nearby_agents(agent)
                    if a.id != target_id
                ]
                if other_targets:
                    target = random.choice(other_targets)
                    target_id = target.id
                    action = "observe"
                else:
                    # No other target — wander toward nearest resource instead
                    action = "wander"
                    dx, dy = _center_seeking_wander(agent)
            elif action == "wander":
                # FIX: keep moving but pick a smarter direction
                # Head toward nearest resource if hungry/thirsty, else toward center
                nearby_resources = world.nearby_resources(agent)
                nearest = None
                if agent.needs.hunger > 0.5:
                    nearest = world.nearest_resource(agent, "berries") or world.nearest_resource(agent, "fish")
                elif agent.needs.thirst > 0.5:
                    nearest = world.nearest_resource(agent, "water")
                if nearest:
                    dist = max(0.1, ((nearest.x - agent.x)**2 + (nearest.y - agent.y)**2)**0.5)
                    dx = ((nearest.x - agent.x) / dist) * 1.5
                    dy = ((nearest.y - agent.y) / dist) * 1.5
                else:
                    dx, dy = _center_seeking_wander(agent)
            else:
                action = "wander"
                dx, dy = _center_seeking_wander(agent)

        if action == "retreat" and target:
            flee_dx = agent.x - target.x
            flee_dy = agent.y - target.y
            dist = max(0.1, (flee_dx**2 + flee_dy**2) ** 0.5)
            dx = (flee_dx / dist) * 2.0
            dy = (flee_dy / dist) * 2.0
        elif action == "speak" and target:
            toward_dx = target.x - agent.x
            toward_dy = target.y - agent.y
            dist = max(0.1, (toward_dx**2 + toward_dy**2) ** 0.5)
            if dist > 2:
                dx = (toward_dx / dist) * 1.0
                dy = (toward_dy / dist) * 1.0

        agent.move(dx, dy)

        success = True
        failure_reason = ""
        logged_success = False

        if action == "rest":
            agent.needs.energy = min(1.0, agent.needs.energy + 0.2)
            agent.needs.loneliness = min(1.0, agent.needs.loneliness + 0.05)
        elif action == "speak":
            if not target:
                success = False
                failure_reason = "missing_or_invalid_target"
            else:
                phrase = _ground_phrase(agent, phrase, target)
                agent.needs.loneliness = max(0.0, agent.needs.loneliness - 0.15)
        elif action == "forage":
            success = bool(resource_name) and world.harvest(agent, resource_name)
            failure_reason = "resource_unavailable_or_too_far"
        elif action == "claim":
            success = bool(resource_name) and world.claim_resource(agent, resource_name)
            failure_reason = "resource_unavailable_or_too_far"
            logged_success = success
        elif action == "craft":
            success = bool(craft_a and craft_b)
            result_item = world.attempt_craft(agent, craft_a, craft_b) if success else None
            success = bool(result_item)
            failure_reason = "missing_items_or_unknown_recipe"
            if success:
                logger.info(f"  [craft] {agent.name} crafted {result_item} from {craft_a}+{craft_b}")
        elif action == "build":
            success = world.contribute_to_project(agent, project_id)
            failure_reason = "no_project_or_no_materials"
            if success:
                logger.info(f"  [build] {agent.name} contributed to a camp project")
        elif action == "trade":
            give_items = _sanitize_items(give_items)
            receive_items = _sanitize_items(receive_items)
            success = bool(target and give_items and receive_items)
            if success:
                success = world.execute_trade(agent, target, give_items, receive_items)
            failure_reason = "invalid_trade_or_missing_inventory"
            logged_success = success
            if success:
                logger.info(f"  [trade] {agent.name} traded {give_items} to {target.name} for {receive_items}")
        elif action == "give":
            give_items = _sanitize_items(give_items)
            success = bool(target and give_items)
            if success:
                success = world.give_items(agent, target, give_items)
            failure_reason = "invalid_gift_or_missing_inventory"
            logged_success = success
            if success:
                logger.info(f"  [give] {agent.name} gave {give_items} to {target.name}")
        elif action == "confront":
            if not target:
                success = False
                failure_reason = "missing_or_invalid_target"
            else:
                agent.needs.anger = max(0.0, agent.needs.anger - 0.1)
                target.needs.fear = min(1.0, target.needs.fear + 0.2)
                agent.update_rel(target_id, rivalry=0.05)
                target.update_rel(agent_id, fear=0.1, trust=-0.05)
        elif action in {"retreat", "observe", "wander", "ignore"}:
            # FIX: observe with no target is meaningless — redirect to wander toward nearest agent
            if action == "observe" and not target:
                nearby = world.nearby_agents(agent)
                if nearby:
                    closest = min(nearby, key=lambda a: (a.x - agent.x)**2 + (a.y - agent.y)**2)
                    target = closest
                    target_id = closest.id
                else:
                    action = "wander"
                    dx, dy = _center_seeking_wander(agent)
            success = True
        else:
            success = False
            failure_reason = "unknown_action"

        agent.nudge_mood(mood_delta)

        if phrase and action == "speak" and success:
            agent.last_phrase = phrase
            agent.last_target = target.name if target else ""
        elif action in {"wander", "forage", "rest", "craft", "build", "claim"}:
            agent.last_phrase = ""
            agent.last_target = ""

        if success and isinstance(rel_updates, dict):
            for rel_target_id, updates in rel_updates.items():
                if rel_target_id in world.agents and isinstance(updates, dict):
                    agent.update_rel(
                        rel_target_id,
                        trust=_safe_float(updates.get("trust"), 0.0),
                        love=_safe_float(updates.get("love"), 0.0),
                        rivalry=_safe_float(updates.get("rivalry"), 0.0),
                        fear=_safe_float(updates.get("fear"), 0.0),
                        debt=_safe_float(updates.get("debt"), 0.0),
                    )

        if success and memory_note:
            agent.remember(world.tick_number, str(memory_note))

        original_action = action  # save before potential redirect

        if not success:
            world.log_failed_action(agent, original_action, failure_reason or "rejected", target)
            agent.nudge_mood(-0.03)
            # Inject a memory note so the LLM sees the failure next tick and stops repeating
            if original_action in {"trade", "give"}:
                tname = target.name if target else "someone"
                agent.remember(world.tick_number, f"My {original_action} with {tname} failed — I lacked items or they refused.")
            elif original_action == "forage":
                agent.remember(world.tick_number, "I tried to forage but found nothing here.")
            if original_action in {"trade", "give", "craft", "build", "forage", "claim"}:
                action = "observe" if target else "wander"

        agent.last_action = original_action if success else action
        agent.record_action(world.tick_number, original_action, success)
        world.increment_metric(f"action_{original_action}")
        if success:
            world.increment_metric("successful_actions")
        else:
            world.increment_metric("failed_actions")
        _log_action(world, agent, original_action, phrase, target, dx, dy, success, logged_success)


def _log_action(world, agent, action, phrase, target, dx, dy, success=True, already_logged=False):
    if not success:
        return

    tname = target.name if target else ""

    # UNIVERSAL EVENT LOGGING
    world.log(action, {
        "agent": agent.name,
        "agent_id": agent.id,
        "target": tname,
        "target_id": target.id if target else None,
        "phrase": phrase,
        "dx": dx,
        "dy": dy,
    })

    if action == "speak" and phrase:
        logger.info(f'  [say] {agent.name} -> {tname or "(alone)"}: "{phrase}"')

    elif action == "confront":
        logger.info(f"  [fight] {agent.name} confronted {tname}")

    elif action == "retreat":
        logger.info(f"  [back] {agent.name} retreated from {tname or 'threat'}")

    elif action == "rest":
        logger.info(f"  [rest] {agent.name} resting")

    elif action == "observe":
        logger.info(f"  [watch] {agent.name} watching {tname or '...'}")

    elif action == "forage":
        logger.info(f"  [forage] {agent.name} foraging")

    elif action == "wander":
        logger.info(f"  [move] {agent.name} wandering ({dx:+.1f}, {dy:+.1f})")

    elif action == "ignore":
        logger.info(f"  [ignore] {agent.name} ignoring {tname}")

    elif action == "give" and not already_logged:
        logger.info(f"  [give] {agent.name} gave to {tname}")

    elif action == "trade" and not already_logged:
        logger.info(f"  [trade] {agent.name} traded with {tname}")

    elif action == "claim":
        logger.info(f"  [claim] {agent.name} claiming territory")

    elif action == "craft":
        logger.info(f"  [craft] {agent.name} crafting")

    elif action == "build":
        logger.info(f"  [build] {agent.name} building")

    if target:
        rel = agent.get_rel(target.id)
        if rel.encounters > 0 and rel.encounters % 5 == 0:
            bond = rel.bond_score()
            if abs(bond) > 0.3:
                logger.info(
                    f"  [bond] {agent.name} <-> {tname}: "
                    f"{rel.label()} (bond={bond:.2f}, enc={rel.encounters})"
                )

async def simulation_loop(world: World):
    logger.info("Civilization awakening...")
    logger.info(f"Tick interval: {TICK_INTERVAL_SECONDS}s | Agents: {len(world.agents)}")

    while True:
        tick_start = time.time()
        world.tick_number += 1
        logger.info(f"--- TICK {world.tick_number} ---")

        world.tick_environment()
        world.tick_needs()
        world.tick_resources()
        world.tick_society()
        world.tick_social_status()

        try:
            async for agent_id, result in run_tick(world):
                apply_results(world, [(agent_id, result)])
                if _broadcast_callback:
                    try:
                        await _broadcast_callback(world.to_snapshot())
                    except Exception as e:
                        logger.error(f"  Broadcast error: {e}")
        except Exception as e:
            logger.error(f"  Tick error: {e}", exc_info=True)

        elapsed = time.time() - tick_start
        logger.info(f"Tick {world.tick_number} complete ({elapsed:.1f}s)")
        await asyncio.sleep(max(0, TICK_INTERVAL_SECONDS - elapsed))