import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Relationship between two agents ──────────────────────────────────────────

@dataclass
class Relationship:
    trust: float = 0.0        # -1 deep distrust → 1 full trust
    love: float = 0.0         # 0 indifferent → 1 deeply bonded
    rivalry: float = 0.0      # 0 none → 1 bitter enemy
    fear: float = 0.0         # 0 unafraid → 1 terrified
    debt: float = 0.0         # negative = they owe me, positive = I owe them
    encounters: int = 0

    def bond_score(self) -> float:
        return self.trust * 0.4 + self.love * 0.4 - self.rivalry * 0.5 - self.fear * 0.1

    def label(self) -> str:
        s = self.bond_score()
        if self.love > 0.5:   return "lover"
        if s > 0.6:           return "ally"
        if s > 0.3:           return "friend"
        if self.rivalry > 0.5:return "enemy"
        if s < -0.3:          return "hostile"
        if self.fear > 0.4:   return "fears"
        if self.debt < -0.3:  return "owes_me"
        if self.debt > 0.3:   return "i_owe"
        return "neutral"

    def update(self, trust=0.0, love=0.0, rivalry=0.0, fear=0.0, debt=0.0):
        self.trust   = max(-1.0, min(1.0, self.trust + trust))
        self.love    = max(0.0,  min(1.0, self.love + love))
        self.rivalry = max(0.0,  min(1.0, self.rivalry + rivalry))
        self.fear    = max(0.0,  min(1.0, self.fear + fear))
        self.debt    = max(-1.0, min(1.0, self.debt + debt))
        self.encounters += 1


# ── Agent needs (dynamic, change every tick) ──────────────────────────────────

@dataclass
class Needs:
    hunger: float    = field(default_factory=lambda: random.uniform(0.1, 0.4))
    thirst: float    = field(default_factory=lambda: random.uniform(0.1, 0.4))
    energy: float    = field(default_factory=lambda: random.uniform(0.5, 0.9))
    health: float    = field(default_factory=lambda: random.uniform(0.7, 1.0))
    loneliness: float= field(default_factory=lambda: random.uniform(0.0, 0.3))
    happiness: float = field(default_factory=lambda: random.uniform(0.3, 0.7))
    fear: float      = field(default_factory=lambda: random.uniform(0.0, 0.2))
    anger: float     = field(default_factory=lambda: random.uniform(0.0, 0.2))

    def tick_decay(self, traits: dict):
        """Passive decay each tick, skewed by personality traits.
        Tuned for long ticks (300s) — slow decay so agents socialise
        rather than spending every tick in survival mode.
        """
        self.hunger    = min(1.0, self.hunger + random.uniform(0.002, 0.004))
        self.thirst    = min(1.0, self.thirst + random.uniform(0.002, 0.005))
        self.energy    = max(0.0, self.energy - random.uniform(0.001, 0.003))
        self.loneliness= min(1.0, self.loneliness + (0.010 if traits.get("empathy", 0.5) > 0.6 else 0.006))
        self.fear      = max(0.0, self.fear - 0.01)
        self.anger     = max(0.0, self.anger - 0.01)

        # Health degrades only in true crisis
        if self.hunger > 0.9 or self.thirst > 0.9:
            self.health = max(0.0, self.health - 0.01)

    def dominant(self) -> str:
        # Survival needs only dominate when genuinely urgent (>0.6)
        # otherwise social/emotional needs can win
        scores = {
            "hunger":    self.hunger     if self.hunger > 0.75    else self.hunger * 0.15,
            "thirst":    self.thirst     if self.thirst > 0.75    else self.thirst * 0.15,
            "loneliness":self.loneliness * 0.9,
            "fear":      self.fear,
            "anger":     self.anger * 0.8,
        }
        return max(scores, key=scores.get)

    def crisis(self) -> list[str]:
        """Return list of needs in critical state."""
        c = []
        if self.hunger > 0.92:   c.append("starving")
        if self.thirst > 0.92:   c.append("dehydrated")
        if self.energy < 0.08:   c.append("exhausted")
        if self.health < 0.3:    c.append("sick")
        if self.loneliness > 0.8:c.append("desperate_for_contact")
        return c

    def to_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}

    def to_prompt_dict(self) -> dict:
        """Human-readable needs for LLM prompt — no raw numbers."""
        def label(v, invert=False):
            v = 1.0 - v if invert else v
            if v < 0.25: return "fine"
            if v < 0.50: return "mild"
            if v < 0.75: return "high"
            return "critical"
        return {
            "hunger":     label(self.hunger),
            "thirst":     label(self.thirst),
            "energy":     label(self.energy, invert=True),
            "loneliness": label(self.loneliness),
            "fear":       label(self.fear),
            "anger":      label(self.anger),
        }


# ── Core Agent ────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id: str
    name: str

    # Permanent personality (set at birth, never changes)
    backstory: str        = ""
    traits: dict          = field(default_factory=dict)
    # traits: courage, greed, empathy, curiosity, aggression (all 0-1)

    # Position
    x: float = field(default_factory=lambda: random.uniform(5, 55))
    y: float = field(default_factory=lambda: random.uniform(5, 55))

    # Dynamic state
    needs: Needs = field(default_factory=Needs)
    mood: float  = field(default_factory=lambda: random.uniform(0.2, 0.6))

    # Inventory: what this agent holds
    inventory: dict = field(default_factory=lambda: {
        "berries": 0, "fish": 0, "water": 0,
        "wood": 0, "stone": 0, "herbs": 0,
        "tool": 0, "medicine": 0, "preserved_food": 0
    })

    # Social
    relationships: dict = field(default_factory=dict)  # agent_id → Relationship

    # Memory: list of {tick, event} dicts
    memory: list = field(default_factory=list)
    action_history: list = field(default_factory=list)

    # Status
    last_action: str     = "wandering"
    last_phrase: str      = ""
    last_target: str      = ""
    social_status: float  = 0.5   # emergent, 0=outcast 1=leader
    territory_claim: str  = None
    home_group: str       = None
    reputation: float     = 0.0
    alive: bool           = True
    born_at_tick: int     = 0

    def get_rel(self, other_id: str) -> Relationship:
        if other_id not in self.relationships:
            self.relationships[other_id] = Relationship()
        return self.relationships[other_id]

    def update_rel(self, other_id: str, **kwargs):
        self.get_rel(other_id).update(**kwargs)

    def remember(self, tick: int, event: str):
        self.memory.append({"tick": tick, "event": event})
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]

    def record_action(self, tick: int, action: str, success: bool = True):
        self.action_history.append({"tick": tick, "action": action, "success": success})
        if len(self.action_history) > 12:
            self.action_history = self.action_history[-12:]

    def repeated_action_count(self, action: str, window: int = 5) -> int:
        return sum(1 for item in self.action_history[-window:] if item["action"] == action)

    def recent_memories(self, n: int = 5) -> list[str]:
        return [m["event"] for m in self.memory[-n:]]

    def wealth(self) -> int:
        return sum(self.inventory.values())

    def mood_label(self) -> str:
        if self.mood > 0.7:  return "joyful"
        if self.mood > 0.5:  return "content"
        if self.mood > 0.3:  return "neutral"
        if self.mood > 0.1:  return "uneasy"
        return "miserable"

    def nudge_mood(self, delta: float):
        self.mood = max(0.0, min(1.0, self.mood + delta))

    def move(self, dx: float, dy: float, w: int = 60, h: int = 60):
        self.x = max(1, min(w - 1, self.x + dx))
        self.y = max(1, min(h - 1, self.y + dy))

    def top_relationships(self, n: int = 4) -> list[dict]:
        rels = []
        for aid, rel in self.relationships.items():
            rels.append({
                "id": aid,
                "label": rel.label(),
                "bond": round(rel.bond_score(), 2),
                "encounters": rel.encounters,
                "debt": round(rel.debt, 2),
            })
        rels.sort(key=lambda r: -abs(r["bond"]))
        return rels[:n]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "mood": round(self.mood, 2),
            "mood_label": self.mood_label(),
            "needs": self.needs.to_dict(),
            "dominant_need": self.needs.dominant(),
            "crisis": self.needs.crisis(),
            "inventory": self.inventory,
            "wealth": self.wealth(),
            "social_status": round(self.social_status, 2),
            "territory_claim": self.territory_claim,
            "home_group": self.home_group,
            "reputation": round(self.reputation, 2),
            "last_action": self.last_action,
            "last_phrase": self.last_phrase,
            "phrase": self.last_phrase,
            "target": self.last_target,
            "action_history": self.action_history[-6:],
            "traits": self.traits,
            "backstory": self.backstory,
            "top_relationships": self.top_relationships(),
            "alive": self.alive,
        }