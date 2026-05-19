import random
from .agent import Agent

NAMES = [
    "Aela", "Bram", "Cass", "Dune", "Elin", "Fenn",
    "Gara", "Holt", "Ira", "Jade", "Kael", "Lira"
]

# Rich backstories that shape personality
BACKSTORIES = [
    "Grew up alone after losing family early. Self-reliant, trusts no one easily, but fiercely loyal once trust is earned.",
    "Former leader of a small group that collapsed due to betrayal. Now calculating and suspicious, always watching for signs of deception.",
    "Naturally charismatic, draws people in effortlessly. Uses charm as currency — everything is a negotiation.",
    "Quiet and observant, speaks rarely but remembers everything. Holds grudges for a long time.",
    "Raised in scarcity, hoards resources instinctively. Will trade only when desperate or when profit is obvious.",
    "Deeply empathetic, feels others' pain acutely. Will sacrifice personal comfort to help those suffering.",
    "Impulsive risk-taker, acts on emotion rather than logic. Brave to the point of recklessness.",
    "Patient and strategic, plays long games. Appears harmless but always has an ulterior motive.",
    "Deeply curious, obsessed with understanding how things work. Will experiment, explore, and discover at the cost of safety.",
    "Craves social connection above all else. Loneliness is their greatest fear, they will do almost anything to belong.",
    "Hardened survivor, ruthless when threatened. Civil and even kind in times of abundance, dangerous when cornered.",
    "Nostalgic and melancholy, haunted by a past they rarely speak of. Generous to a fault, seeking meaning through giving.",
]

def _generate_traits(backstory: str) -> dict:
    """Derive personality traits from backstory."""
    b = backstory.lower()
    return {
        "courage":    round(random.uniform(0.3, 0.9), 2),
        "greed":      round(random.uniform(0.1, 0.8), 2),
        "empathy":    round(0.8 if "empathetic" in b or "giving" in b else random.uniform(0.1, 0.7), 2),
        "curiosity":  round(0.9 if "curious" in b else random.uniform(0.2, 0.7), 2),
        "aggression": round(0.8 if "ruthless" in b or "dangerous" in b else random.uniform(0.1, 0.6), 2),
        "patience":   round(0.9 if "patient" in b or "strategic" in b else random.uniform(0.2, 0.7), 2),
        "loyalty":    round(0.9 if "loyal" in b else random.uniform(0.2, 0.8), 2),
    }


def spawn_agents(count: int = 12) -> dict:
    """Spawn count agents with unique names, backstories, traits."""
    names = random.sample(NAMES, min(count, len(NAMES)))
    backstories = random.sample(BACKSTORIES, min(count, len(BACKSTORIES)))

    agents = {}
    for i, (name, backstory) in enumerate(zip(names, backstories)):
        traits = _generate_traits(backstory)
        agent = Agent(
            id=f"agent_{i:02d}",
            name=name,
            backstory=backstory,
            traits=traits,
            x=random.uniform(5, 55),
            y=random.uniform(5, 55),
        )
        agents[agent.id] = agent

    return agents
