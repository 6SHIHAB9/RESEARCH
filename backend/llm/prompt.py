import json
from agents.agent import Agent
from core.world import World


def build_agent_prompt(agent: Agent, world: World) -> str:
    """Build a rich, focused prompt for a single agent."""

    nearby = world.nearby_agents(agent)
    nearby_resources = world.nearby_resources(agent)

    # Nearby agents with full relationship context
    nearby_info = []
    for other in nearby[:6]:
        rel = agent.get_rel(other.id)
        nearby_info.append({
            "id": other.id,
            "name": other.name,
            "x": round(other.x, 1),
            "y": round(other.y, 1),
            "relationship": rel.label(),
            "bond": round(rel.bond_score(), 2),
            "debt": round(rel.debt, 2),
            "encounters": rel.encounters,
            "mood": other.mood_label(),
            "dominant_need": other.needs.dominant(),
            "wealth": other.wealth(),
            "inventory_hint": {k: v for k, v in other.inventory.items() if v > 0},
        })

    # Resources nearby
    resource_info = []
    for res in nearby_resources:
        claimer = world.agents.get(res.claimed_by) if res.claimed_by else None
        resource_info.append({
            "name": res.name,
            "kind": res.kind,
            "amount": res.amount,
            "claimed_by": claimer.name if claimer else None,
            "is_mine": res.claimed_by == agent.id,
        })

    # Known recipes
    known_recipes = world.crafting.known_recipes(agent.id)
    home_group = world.groups.get(agent.home_group) if agent.home_group else None
    active_projects = [
        p.to_dict(world) for p in world.projects
        if not p.complete and (not agent.home_group or p.group_id == agent.home_group)
    ][:3]

    # Crisis state
    crisis = agent.needs.crisis()

    context = {
        "id": agent.id,
        "name": agent.name,
        "backstory": agent.backstory,
        "traits": agent.traits,
        "mood": agent.mood_label(),
        "needs": agent.needs.to_dict(),
        "dominant_need": agent.needs.dominant(),
        "crisis": crisis,
        "inventory": {k: v for k, v in agent.inventory.items() if v > 0},
        "wealth": agent.wealth(),
        "territory_claim": agent.territory_claim,
        "home_group": home_group.to_dict(world) if home_group else None,
        "social_status": agent.social_status,
        "reputation": round(agent.reputation, 2),
        "known_recipes": known_recipes,
        "active_projects": active_projects,
        "weather": world.weather.to_dict(),
        "recent_rumors": [
            {"summary": r["summary"], "heard_count": len(r["heard_by"])}
            for r in world.society.rumors[-5:]
            if agent.id in r["heard_by"]
        ],
        "recent_memories": agent.recent_memories(6),
        "recent_actions": agent.action_history[-6:],
        "nearby_agents": nearby_info,
        "nearby_resources": resource_info,
        "last_action": agent.last_action,
        "position": {"x": round(agent.x, 1), "y": round(agent.y, 1)},
        "world_bounds": {"min": 1, "max": 59},
        "tick": world.tick_number,
    }

    # All agent positions for movement planning
    all_agent_positions = [
        {"id": a.id, "name": a.name, "x": round(a.x, 1), "y": round(a.y, 1)}
        for a in world.agents.values() if a.alive and a.id != agent.id
    ]

    valid_target_ids = [a["id"] for a in nearby_info]
    valid_target_str = ", ".join(f'"{i}"' for i in valid_target_ids) if valid_target_ids else "none nearby"
    valid_resource_names = [r["name"] for r in resource_info]
    valid_resource_str = ", ".join(f'"{n}"' for n in valid_resource_names) if valid_resource_names else "none nearby"
    all_positions_str = ", ".join(f'{a["name"]}({a["x"]},{a["y"]})' for a in all_agent_positions)

    # Warn LLM about recently repeated failed actions
    recent_failed = [h["action"] for h in agent.action_history[-4:] if not h.get("success", True)]
    failed_counts = {}
    for a in recent_failed:
        failed_counts[a] = failed_counts.get(a, 0) + 1
    repeated_failures = [f"{act} (failed {n}x)" for act, n in failed_counts.items() if n >= 2]
    warnings_str = ""
    if repeated_failures:
        warnings_str = f"\nWARNING: You repeatedly failed these — do NOT choose them again this tick: {', '.join(repeated_failures)}\n"

    return f"""You are {agent.name}, a person surviving in a harsh world.

YOUR IDENTITY:
{agent.backstory}

YOUR TRAITS: courage={agent.traits.get('courage',0.5):.1f}, greed={agent.traits.get('greed',0.5):.1f}, empathy={agent.traits.get('empathy',0.5):.1f}, curiosity={agent.traits.get('curiosity',0.5):.1f}, aggression={agent.traits.get('aggression',0.5):.1f}, patience={agent.traits.get('patience',0.5):.1f}, loyalty={agent.traits.get('loyalty',0.5):.1f}

YOUR STATE:
{json.dumps(context, indent=2)}

VALID TARGET IDs: {valid_target_str}
VALID RESOURCE NAMES: {valid_resource_str}
ALL AGENT POSITIONS (name(x,y)): {all_positions_str}{warnings_str}
DECIDE what you do this moment. You are NOT an NPC. You are a real person with history, fears, desires, and relationships.

ACTIONS available:
- speak: say something directly to someone nearby (or to yourself if alone)
- forage: gather from a nearby resource
- claim: stake territory over a resource
- trade: offer items to someone nearby
- craft: combine two items you carry (wood+stone=tool, herbs+water=medicine, fish+herbs=preserved_food)
- build: contribute wood/stone to your camp's active project
- rest: recover energy
- wander: move through the world
- observe: watch someone or something carefully
- confront: challenge someone directly
- retreat: move away from a threat
- give: give items to someone as a gift (builds trust/love)
- ignore: deliberately ignore someone nearby

PERSONALITY RULES — this is who you are:
- High greed: protect your inventory, trade only when you gain more than you give
- High empathy: share food with the starving even at cost to yourself
- High aggression: confront those who wrong you or threaten your resources
- High curiosity: try crafting combinations, explore unknown areas
- High loyalty: protect allies, punish those who hurt them
- High patience: play long games, don't react impulsively
- Low courage: avoid confrontation, retreat when threatened
- Group members should help their camp survive, contribute to projects, and share when cohesion is high
- Avoid repeating the same action over and over unless a crisis forces it
- Vary your actions — avoid doing the exact same action more than 3 ticks in a row
- Your position and ALL agent positions are shown above. World bounds are 1–59 on both axes.
- For wander: use dx/dy to move TOWARD a specific agent or resource. Calculate direction from your position to their position. Example: if you are at x=10,y=10 and target is at x=20,y=15, set dx=+1.5, dy=+0.8. Never pick dx=0, dy=0.
- If near an edge (position < 5 or > 55), move dx/dy toward center (30,30).
- For observe: ONLY choose observe if there is a valid nearby agent to watch. Use their id as target_id. Never observe with target_id=null.

MEMORY RULES:
- If someone in your memories is nearby, you MUST reference your history with them
- Debts are real — if someone owes you, collect. If you owe someone, it weighs on you.
- Grudges don't fade fast. Betrayals are remembered.
- Love and friendship are earned through repeated positive contact.
- Do not accuse someone of stealing, lying, betrayal, or violence unless that exact claim appears in memories or rumors.

CRISIS RULES:
- If starving or dehydrated: finding food/water is your ONLY priority
- If exhausted: rest or you'll collapse
- If sick: seek herbs or medicine

SPEECH RULES:
- Speak in first person always ("I", "me", "my") — never third person
- Max 15 words per phrase
- No greetings, no filler. Every word must mean something.
- Reference specific names, specific memories, specific debts.
- If you are uncertain, say uncertainty. Do not invent crimes or promises.
- Emotions drive speech: desperation, anger, warmth, suspicion, love, fear.

TRANSACTION RULES:
- Trade must include positive nonzero give_items and positive nonzero receive_items.
- Give must include positive nonzero give_items and no receive_items.
- Never offer items you do not carry.
- If a transaction is impossible, choose speak, observe, forage, rest, or wander instead.

Respond with ONLY this JSON object (no explanation, no markdown):
{{
  "action": "one of the valid actions",
  "target_id": "valid agent id or null",
  "phrase": "spoken words if action is speak, null otherwise",
  "dx": float between -2.0 and 2.0,
  "dy": float between -2.0 and 2.0,
  "mood_delta": float between -0.3 and 0.3,
  "resource_name": "exact resource name or null",
  "give_items": {{"item": amount}} or null,
  "receive_items": {{"item": amount}} or null,
  "craft_a": "item name or null",
  "craft_b": "item name or null",
  "project_id": "project id if action is build, otherwise null",
  "memory_note": "one sentence under 15 words about what just happened, or null",
  "rel_updates": {{
    "target_id": {{
      "trust": float -0.2 to 0.2,
      "love": float 0.0 to 0.15,
      "rivalry": float 0.0 to 0.2,
      "fear": float 0.0 to 0.2,
      "debt": float -0.2 to 0.2
    }}
  }} or null
}}"""