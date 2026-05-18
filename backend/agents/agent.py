import random
import time
from dataclasses import dataclass, field


MOOD_RANGE = (-1.0, 1.0)   # -1 = very negative, 0 = neutral, 1 = very positive


@dataclass
class Agent:
    id: str
    name: str
    personality: str
    x: float
    y: float
    mood: float = field(default_factory=lambda: random.uniform(-0.2, 0.4))
    memory: list[dict] = field(default_factory=list)
    recent_phrases: list[str] = field(default_factory=list)
    social_bonds: dict[str, float] = field(default_factory=dict)  # agent_id -> bond strength
    last_action: str = "wandering"
    last_interaction_tick: int = 0
    created_at: float = field(default_factory=time.time)

    def add_memory(self, entry: dict):
        """Add a memory fragment. Keep only most recent 8."""
        self.memory.append(entry)
        if len(self.memory) > 8:
            self.memory = self.memory[-8:]

    def update_bond(self, other_id: str, delta: float):
        """Strengthen or weaken bond with another agent."""
        current = self.social_bonds.get(other_id, 0.0)
        updated = max(-1.0, min(1.0, current + delta))
        self.social_bonds[other_id] = updated

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

    def to_dict(self) -> dict:
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
            "bond_count": len([v for v in self.social_bonds.values() if v > 0.2]),
            "top_bonds": sorted(self.social_bonds.items(), key=lambda x: -x[1])[:3],
        }
