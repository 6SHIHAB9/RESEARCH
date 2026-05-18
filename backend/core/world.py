import random
import time
from dataclasses import dataclass, field
from agents.agent import Agent
from agents.names import AGENT_NAMES, PERSONALITIES

WORLD_WIDTH = 100
WORLD_HEIGHT = 100
PERCEPTION_RADIUS = 12
INITIAL_AGENT_COUNT = 25


@dataclass
class Resource:
    name: str
    x: float
    y: float
    kind: str        # "food", "shelter", "tools", "territory"
    amount: int = 10
    max_amount: int = 10
    claimed_by: str = None   # agent_id or None

    replenish_every: int = 8   # ticks between each +1 unit
    _replenish_counter: int = 0

    def replenish(self):
        """Only add 1 unit every replenish_every ticks, not every tick."""
        if self.amount < self.max_amount:
            self._replenish_counter += 1
            if self._replenish_counter >= self.replenish_every:
                self.amount = min(self.max_amount, self.amount + 1)
                self._replenish_counter = 0


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

        self.landmarks = [
            Landmark("Silent Lake", 30, 70, "water"),
            Landmark("Old Firepit", 50, 50, "ruin"),
            Landmark("Hollow Tree", 70, 30, "nature"),
            Landmark("Glass Tower", 20, 20, "structure"),
        ]

        # World resources that agents compete / cooperate over
        # Low amounts + slow replenish = real scarcity and competition
        self.resources: list[Resource] = [
            Resource("Fishing Spot", 28, 72, "food",      amount=4, max_amount=6,  replenish_every=10),
            Resource("Berry Grove",  55, 45, "food",      amount=5, max_amount=8,  replenish_every=8),
            Resource("Cave Shelter", 18, 22, "shelter",   amount=3, max_amount=3,  replenish_every=20),
            Resource("Tool Cache",   72, 28, "tools",     amount=2, max_amount=4,  replenish_every=15),
            Resource("High Ground",  80, 80, "territory", amount=2, max_amount=2,  replenish_every=25),
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
        nearby = []
        for other in self.agents.values():
            if other.id == agent.id:
                continue
            dist = ((agent.x - other.x) ** 2 + (agent.y - other.y) ** 2) ** 0.5
            if dist <= PERCEPTION_RADIUS:
                nearby.append(other)
        return nearby

    def get_nearby_landmarks(self, agent: Agent, radius: float = PERCEPTION_RADIUS) -> list[Landmark]:
        nearby = []
        for lm in self.landmarks:
            dist = ((agent.x - lm.x) ** 2 + (agent.y - lm.y) ** 2) ** 0.5
            if dist <= radius:
                nearby.append(lm)
        return nearby

    def get_nearby_resources(self, agent: Agent, radius: float = PERCEPTION_RADIUS) -> list[Resource]:
        nearby = []
        for res in self.resources:
            dist = ((agent.x - res.x) ** 2 + (agent.y - res.y) ** 2) ** 0.5
            if dist <= radius:
                nearby.append(res)
        return nearby

    def tick_resources(self):
        """Replenish resources and expire stale claims."""
        import logging
        logger = logging.getLogger(__name__)
        for res in self.resources:
            was_empty = res.amount == 0
            res.replenish()
            if res.amount == 0 and not was_empty:
                logger.info(f"  💀 {res.name} is fully depleted!")
                self.log_event({"type": "resource_depleted", "resource": res.name})
            # Claim expires if claimant drifted far away
            if res.claimed_by:
                agent = self.agents.get(res.claimed_by)
                if agent:
                    dist = ((agent.x - res.x) ** 2 + (agent.y - res.y) ** 2) ** 0.5
                    if dist > PERCEPTION_RADIUS * 1.5:
                        res.claimed_by = None
                else:
                    res.claimed_by = None

    def log_event(self, event: dict):
        self.event_log.append({**event, "tick": self.tick_number})
        if len(self.event_log) > 300:
            self.event_log = self.event_log[-300:]

    def to_snapshot(self) -> dict:
        return {
            "tick": self.tick_number,
            "timestamp": time.time(),
            "agents": [a.to_dict() for a in self.agents.values()],
            "landmarks": [{"name": lm.name, "x": lm.x, "y": lm.y, "kind": lm.kind}
                          for lm in self.landmarks],
            "resources": [
                {
                    "name": r.name, "x": r.x, "y": r.y, "kind": r.kind,
                    "amount": r.amount, "max_amount": r.max_amount,
                    "claimed_by": r.claimed_by,
                }
                for r in self.resources
            ],
            "recent_events": self.event_log[-30:],
        }