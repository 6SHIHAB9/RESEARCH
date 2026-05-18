import random
import time
from dataclasses import dataclass
from agents.agent import Agent
from agents.names import AGENT_NAMES, PERSONALITIES

WORLD_WIDTH = 100
WORLD_HEIGHT = 100
PERCEPTION_RADIUS = 12  # Tightened from 18 to reduce spatial spread and increase recurring encounters
INITIAL_AGENT_COUNT = 25

@dataclass
class Landmark:
    name: str
    x: float
    y: float
    kind: str

class World:
    def __init__(self):
        self.tick_number = 0
        self.created_at = time.time()
        self.agents: dict[str, Agent] = {}
        self.event_log: list[dict] = []
        
        # 1. LANDMARK SYSTEM: Permanent geography
        self.landmarks = [
            Landmark("Silent Lake", 30, 70, "water"),
            Landmark("Old Firepit", 50, 50, "ruin"),
            Landmark("Hollow Tree", 70, 30, "nature"),
            Landmark("Glass Tower", 20, 20, "structure"),
        ]
        
        self._initialize_agents()

    def _initialize_agents(self):
        names = random.sample(AGENT_NAMES, INITIAL_AGENT_COUNT)
        personalities = random.choices(PERSONALITIES, k=INITIAL_AGENT_COUNT)

        for i, (name, personality) in enumerate(zip(names, personalities)):
            agent = Agent(
                id=f"agent_{i:03d}",
                name=name,
                personality=personality,
                x=random.uniform(5, WORLD_WIDTH - 5),
                y=random.uniform(5, WORLD_HEIGHT - 5),
            )
            self.agents[agent.id] = agent

    def get_nearby_agents(self, agent: Agent) -> list[Agent]:
        """Return agents within perception radius, excluding self."""
        nearby = []
        for other in self.agents.values():
            if other.id == agent.id:
                continue
            dist = ((agent.x - other.x) ** 2 + (agent.y - other.y) ** 2) ** 0.5
            if dist <= PERCEPTION_RADIUS:
                nearby.append(other)
        return nearby

    def get_nearby_landmarks(self, agent: Agent, radius: float = PERCEPTION_RADIUS) -> list[Landmark]:
        """2. LOCAL LANDMARK PERCEPTION: Return landmarks within perception radius."""
        nearby = []
        for lm in self.landmarks:
            dist = ((agent.x - lm.x) ** 2 + (agent.y - lm.y) ** 2) ** 0.5
            if dist <= radius:
                nearby.append(lm)
        return nearby

    def log_event(self, event: dict):
        self.event_log.append({**event, "tick": self.tick_number})
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

    def to_snapshot(self) -> dict:
        return {
            "tick": self.tick_number,
            "timestamp": time.time(),
            "agents": [a.to_dict() for a in self.agents.values()],
            "landmarks": [{"name": lm.name, "x": lm.x, "y": lm.y, "kind": lm.kind} for lm in self.landmarks],
            "recent_events": self.event_log[-30:],
        }