import time
from agents.agent import Agent
from agents.spawner import spawn_agents
from economy.resources import ResourceNode, create_world_resources, CraftingSystem
from economy.market import Market
from config import WORLD_WIDTH, WORLD_HEIGHT, PERCEPTION_RADIUS, MAX_EVENTS


class World:
    def __init__(self):
        self.tick_number = 0
        self.created_at = time.time()

        # Agents
        self.agents: dict[str, Agent] = spawn_agents(12)

        # Resources
        self.resources: list[ResourceNode] = create_world_resources()

        # Economy
        self.crafting = CraftingSystem()
        self.market = Market()

        # Event log
        self.events: list[dict] = []

        # Landmarks (for UI reference)
        self.landmarks = [
            {"name": "The Spring",      "x": 30, "y": 30, "kind": "water"},
            {"name": "Old Ruins",       "x": 45, "y": 15, "kind": "ruin"},
            {"name": "Tall Pine",       "x": 10, "y": 50, "kind": "nature"},
            {"name": "Clifftop",        "x": 55, "y": 55, "kind": "high_ground"},
        ]

    # ── Spatial queries ───────────────────────────────────────────────────────

    def nearby_agents(self, agent: Agent, radius: float = PERCEPTION_RADIUS) -> list[Agent]:
        return [
            a for a in self.agents.values()
            if a.id != agent.id and a.alive
            and ((a.x - agent.x)**2 + (a.y - agent.y)**2)**0.5 <= radius
        ]

    def nearby_resources(self, agent: Agent, radius: float = PERCEPTION_RADIUS) -> list[ResourceNode]:
        return [
            r for r in self.resources
            if ((r.x - agent.x)**2 + (r.y - agent.y)**2)**0.5 <= radius
        ]

    def nearest_resource(self, agent: Agent, kind: str) -> ResourceNode | None:
        candidates = [r for r in self.resources if r.kind == kind and r.amount > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (r.x - agent.x)**2 + (r.y - agent.y)**2)

    # ── Tick updates ──────────────────────────────────────────────────────────

    def tick_needs(self):
        """Decay all agent needs every tick."""
        for agent in self.agents.values():
            if agent.alive:
                agent.needs.tick_decay(agent.traits)

    def tick_resources(self):
        """Replenish resources, expire stale claims."""
        for res in self.resources:
            res.tick()
            if res.claimed_by:
                claimer = self.agents.get(res.claimed_by)
                if not claimer or not claimer.alive:
                    res.claimed_by = None
                    continue
                dist = ((claimer.x - res.x)**2 + (claimer.y - res.y)**2)**0.5
                if dist > PERCEPTION_RADIUS * 2:
                    res.claimed_by = None

    def tick_social_status(self):
        """Recalculate social status based on wealth and relationships."""
        for agent in self.agents.values():
            if not agent.alive:
                continue
            wealth_score = min(1.0, agent.wealth() / 20)
            ally_score = len([r for r in agent.relationships.values() if r.bond_score() > 0.3]) / 11
            agent.social_status = round(wealth_score * 0.4 + ally_score * 0.6, 2)

    # ── Resource actions ──────────────────────────────────────────────────────

    def harvest(self, agent: Agent, resource_name: str, amount: int = 1) -> bool:
        res = next((r for r in self.resources if r.name == resource_name), None)
        if not res or res.amount <= 0:
            return False

        dist = ((agent.x - res.x)**2 + (agent.y - res.y)**2)**0.5
        if dist > PERCEPTION_RADIUS:
            return False

        # If claimed by someone else, create conflict
        if res.claimed_by and res.claimed_by != agent.id:
            claimer = self.agents.get(res.claimed_by)
            if claimer:
                agent.update_rel(res.claimed_by, rivalry=0.05)
                claimer.update_rel(agent.id, rivalry=0.08, trust=-0.05)
                self.log("resource_conflict", {
                    "agent": agent.name, "agent_id": agent.id,
                    "claimer": claimer.name, "resource": resource_name
                })

        taken = res.harvest(amount)
        agent.inventory[res.kind] = agent.inventory.get(res.kind, 0) + taken
        agent.needs.hunger = max(0.0, agent.needs.hunger - 0.2) if res.kind in ("berries", "fish") else agent.needs.hunger
        agent.needs.thirst = max(0.0, agent.needs.thirst - 0.25) if res.kind == "water" else agent.needs.thirst
        agent.nudge_mood(0.1)
        return True

    def claim_resource(self, agent: Agent, resource_name: str) -> bool:
        res = next((r for r in self.resources if r.name == resource_name), None)
        if not res:
            return False
        dist = ((agent.x - res.x)**2 + (agent.y - res.y)**2)**0.5
        if dist > PERCEPTION_RADIUS:
            return False

        old_claimer_id = res.claimed_by
        res.claimed_by = agent.id
        agent.territory_claim = resource_name

        if old_claimer_id and old_claimer_id != agent.id:
            old_claimer = self.agents.get(old_claimer_id)
            if old_claimer:
                old_claimer.territory_claim = None
                agent.update_rel(old_claimer_id, rivalry=0.05)
                old_claimer.update_rel(agent.id, rivalry=0.12, fear=0.05, trust=-0.1)
                self.log("territory_seized", {
                    "agent": agent.name, "agent_id": agent.id,
                    "from": old_claimer.name, "resource": resource_name
                })
                return True

        self.log("territory_claimed", {
            "agent": agent.name, "agent_id": agent.id, "resource": resource_name
        })
        return True

    def execute_trade(self, from_agent: Agent, to_agent: Agent,
                      give: dict, receive: dict) -> bool:
        """Execute a trade if both agents have the items."""
        for item, amount in give.items():
            if from_agent.inventory.get(item, 0) < amount:
                return False
        for item, amount in receive.items():
            if to_agent.inventory.get(item, 0) < amount:
                return False

        for item, amount in give.items():
            from_agent.inventory[item] -= amount
            to_agent.inventory[item] = to_agent.inventory.get(item, 0) + amount

        for item, amount in receive.items():
            to_agent.inventory[item] -= amount
            from_agent.inventory[item] = from_agent.inventory.get(item, 0) + amount

        from_agent.update_rel(to_agent.id, trust=0.05, debt=0.1)
        to_agent.update_rel(from_agent.id, trust=0.05, debt=-0.1)

        self.market.record_trade(
            self.tick_number,
            from_agent.id, from_agent.name,
            to_agent.id, to_agent.name,
            give, receive
        )
        self.log("trade", {
            "from": from_agent.name, "from_id": from_agent.id,
            "to": to_agent.name, "to_id": to_agent.id,
            "gave": give, "received": receive
        })
        return True

    def attempt_craft(self, agent: Agent, item_a: str, item_b: str) -> str | None:
        if agent.inventory.get(item_a, 0) < 1 or agent.inventory.get(item_b, 0) < 1:
            return None
        result, is_new = self.crafting.attempt_craft(agent.id, item_a, item_b)
        if result:
            agent.inventory[item_a] -= 1
            agent.inventory[item_b] -= 1
            agent.inventory[result] = agent.inventory.get(result, 0) + 1
            agent.nudge_mood(0.15)
            if is_new:
                agent.remember(self.tick_number, f"Discovered that {item_a}+{item_b}={result}!")
                self.log("discovery", {
                    "agent": agent.name, "agent_id": agent.id,
                    "recipe": f"{item_a}+{item_b}={result}", "is_new": is_new
                })
        return result

    # ── Event log ─────────────────────────────────────────────────────────────

    def log(self, event_type: str, data: dict):
        self.events.append({"tick": self.tick_number, "type": event_type, **data})
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]

    # ── Snapshot for UI/WebSocket ─────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "tick": self.tick_number,
            "timestamp": time.time(),
            "agents": [a.to_dict() for a in self.agents.values() if a.alive],
            "resources": [r.to_dict() for r in self.resources],
            "landmarks": self.landmarks,
            "events": self.events[-50:],
            "economy": self.market.to_dict(),
            "known_recipes": {
                aid: self.crafting.known_recipes(aid)
                for aid in self.agents
            },
        }
