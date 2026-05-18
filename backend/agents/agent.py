import random
import time
from dataclasses import dataclass, field


MOOD_RANGE = (-1.0, 1.0)


@dataclass
class Relationship:
    """Tracks multi-dimensional relationship between two agents."""
    trust: float = 0.0
    fear: float = 0.0
    affinity: float = 0.0
    hostility: float = 0.0
    encounters: int = 0

    def net_bond(self) -> float:
        return (self.trust * 0.4 + self.affinity * 0.4 - self.fear * 0.1 - self.hostility * 0.5)

    def label(self) -> str:
        net = self.net_bond()
        if net > 0.5:
            return "ally"
        elif net > 0.2:
            return "friendly"
        elif net < -0.5:
            return "rival"
        elif net < -0.2:
            return "tense"
        elif self.fear > 0.4:
            return "fearful"
        return "neutral"


@dataclass
class NeedsState:
    """Internal survival and social needs that drive behavior."""
    hunger: float = field(default_factory=lambda: random.uniform(0.1, 0.4))
    energy: float = field(default_factory=lambda: random.uniform(0.5, 0.9))
    loneliness: float = field(default_factory=lambda: random.uniform(0.0, 0.4))
    curiosity: float = field(default_factory=lambda: random.uniform(0.2, 0.7))
    fear: float = field(default_factory=lambda: random.uniform(0.0, 0.2))
    aggression: float = field(default_factory=lambda: random.uniform(0.0, 0.3))

    def tick_decay(self, personality: str):
        is_aggressive = any(t in personality.lower() for t in ("aggressive", "dominant", "ruthless"))
        is_curious = any(t in personality.lower() for t in ("curious", "explorer", "wanderer"))
        is_social = any(t in personality.lower() for t in ("social", "empathic", "gregarious"))

        self.hunger = min(1.0, self.hunger + random.uniform(0.02, 0.06))
        self.energy = max(0.0, self.energy - random.uniform(0.01, 0.04))
        self.loneliness = min(1.0, self.loneliness + (0.04 if is_social else 0.015))
        self.curiosity = min(1.0, self.curiosity + (0.05 if is_curious else 0.01))
        self.aggression = max(0.0, self.aggression + (0.02 if is_aggressive else -0.01))
        self.fear = max(0.0, self.fear - 0.01)

    def dominant_need(self) -> str:
        scores = {
            "hunger": self.hunger,
            "loneliness": self.loneliness * 0.8,
            "fear": self.fear,
            "curiosity": self.curiosity * 0.6,
            "aggression": self.aggression * 0.7,
        }
        return max(scores, key=scores.get)

    def to_dict(self) -> dict:
        return {
            "hunger": round(self.hunger, 2),
            "energy": round(self.energy, 2),
            "loneliness": round(self.loneliness, 2),
            "curiosity": round(self.curiosity, 2),
            "fear": round(self.fear, 2),
            "aggression": round(self.aggression, 2),
        }


@dataclass
class Agent:
    id: str
    name: str
    personality: str
    x: float
    y: float
    mood: float = field(default_factory=lambda: random.uniform(-0.2, 0.4))
    memory: list = field(default_factory=list)
    recent_phrases: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)
    last_action: str = "wandering"
    last_interaction_tick: int = 0
    created_at: float = field(default_factory=time.time)
    needs: NeedsState = field(default_factory=NeedsState)
    resources: dict = field(default_factory=lambda: {"food": 0, "tools": 0})
    territory_claim: str = None

    @property
    def social_bonds(self) -> dict:
        return {aid: rel.net_bond() for aid, rel in self.relationships.items()}

    def get_relationship(self, other_id: str) -> Relationship:
        if other_id not in self.relationships:
            self.relationships[other_id] = Relationship()
        return self.relationships[other_id]

    def update_relationship(self, other_id: str, trust_delta=0.0, fear_delta=0.0,
                            affinity_delta=0.0, hostility_delta=0.0):
        rel = self.get_relationship(other_id)
        rel.trust = max(-1.0, min(1.0, rel.trust + trust_delta))
        rel.fear = max(0.0, min(1.0, rel.fear + fear_delta))
        rel.affinity = max(-1.0, min(1.0, rel.affinity + affinity_delta))
        rel.hostility = max(0.0, min(1.0, rel.hostility + hostility_delta))
        rel.encounters += 1

    def update_bond(self, other_id: str, delta: float):
        if delta >= 0:
            self.update_relationship(other_id, trust_delta=delta * 0.5, affinity_delta=delta * 0.5)
        else:
            self.update_relationship(other_id, hostility_delta=abs(delta) * 0.5, affinity_delta=delta * 0.5)

    def add_memory(self, entry: dict):
        self.memory.append(entry)
        if len(self.memory) > 12:
            self.memory = self.memory[-12:]

    def nudge_mood(self, delta: float):
        self.mood = max(MOOD_RANGE[0], min(MOOD_RANGE[1], self.mood + delta))

    def move(self, dx: float, dy: float, world_w: float = 100, world_h: float = 100):
        self.x = max(1, min(world_w - 1, self.x + dx))
        self.y = max(1, min(world_h - 1, self.y + dy))

    def mood_label(self) -> str:
        if self.mood > 0.5:
            return "elated"
        elif self.mood > 0.2:
            return "content"
        elif self.mood > -0.1:
            return "neutral"
        elif self.mood > -0.4:
            return "uneasy"
        else:
            return "withdrawn"

    def behavior_weights(self) -> dict:
        n = self.needs
        w = {
            "speak":    0.10 + n.loneliness * 0.40,
            "forage":   0.05 + n.hunger * 0.50,
            "observe":  0.10 + n.curiosity * 0.30,
            "wander":   0.20,
            "retreat":  0.05 + n.fear * 0.45,
            "confront": 0.02 + n.aggression * 0.35,
            "trade":    0.03 + (n.hunger * 0.2 if self.resources.get("food", 0) > 0 else 0),
            "rest":     0.05 + (0.30 if n.energy < 0.3 else 0),
            "linger":   0.10,
            "ignore":   0.10,
        }
        return w

    def to_dict(self) -> dict:
        top_rels = sorted(
            [(aid, rel.net_bond(), rel.label()) for aid, rel in self.relationships.items()],
            key=lambda x: -abs(x[1])
        )[:3]
        return {
            "id": self.id,
            "name": self.name,
            "personality": self.personality,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "mood": round(self.mood, 3),
            "mood_label": self.mood_label(),
            "last_action": self.last_action,
            "recent_phrase": self.recent_phrases[-1] if self.recent_phrases else None,
            "bond_count": len([r for r in self.relationships.values() if r.net_bond() > 0.2]),
            "top_bonds": [(aid, round(score, 2), label) for aid, score, label in top_rels],
            "needs": self.needs.to_dict(),
            "resources": self.resources,
            "territory_claim": self.territory_claim,
            "dominant_need": self.needs.dominant_need(),
        }