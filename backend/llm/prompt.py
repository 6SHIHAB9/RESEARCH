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
        "social_status": agent.social_status,
        "known_recipes": known_recipes,
        "recent_memories": agent.recent_memories(6),
        "nearby_agents": nearby_info,
        "nearby_resources": resource_info,
        "last_action": agent.last_action,
        "tick": world.tick_number,
    }

    valid_target_ids = [a["id"] for a in nearby_info]
    valid_target_str = ", ".join(f'"{i}"' for i in valid_target_ids) if valid_target_ids else "none nearby"
    valid_resource_names = [r["name"] for r in resource_info]
    valid_resource_str = ", ".join(f'"{n}"' for n in valid_resource_names) if valid_resource_names else "none nearby"

    return f"""You are {agent.name}, a person surviving in a harsh world.

YOUR IDENTITY:
{agent.backstory}

YOUR TRAITS: courage={agent.traits.get('courage',0.5):.1f}, greed={agent.traits.get('greed',0.5):.1f}, empathy={agent.traits.get('empathy',0.5):.1f}, curiosity={agent.traits.get('curiosity',0.5):.1f}, aggression={agent.traits.get('aggression',0.5):.1f}, patience={agent.traits.get('patience',0.5):.1f}, loyalty={agent.traits.get('loyalty',0.5):.1f}

YOUR STATE:
{json.dumps(context, indent=2)}

VALID TARGET IDs: {valid_target_str}
VALID RESOURCE NAMES: {valid_resource_str}

DECIDE what you do this moment. You are NOT an NPC. You are a real person with history, fears, desires, and relationships.

ACTIONS available:
- speak: say something directly to someone nearby (or to yourself if alone)
- forage: gather from a nearby resource
- claim: stake territory over a resource
- trade: offer items to someone nearby
- craft: combine two items you carry (wood+stone=tool, herbs+water=medicine, fish+herbs=preserved_food)
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

MEMORY RULES:
- If someone in your memories is nearby, you MUST reference your history with them
- Debts are real — if someone owes you, collect. If you owe someone, it weighs on you.
- Grudges don't fade fast. Betrayals are remembered.
- Love and friendship are earned through repeated positive contact.

CRISIS RULES:
- If starving or dehydrated: finding food/water is your ONLY priority
- If exhausted: rest or you'll collapse
- If sick: seek herbs or medicine

SPEECH RULES:
- Speak in first person always ("I", "me", "my") — never third person
- Max 15 words per phrase
- No greetings, no filler. Every word must mean something.
- Reference specific names, specific memories, specific debts.
- Emotions drive speech: desperation, anger, warmth, suspicion, love, fear.

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
