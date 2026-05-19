import random
from dataclasses import dataclass, field
from config import RESOURCE_REPLENISH, CRAFTING_RECIPES


@dataclass
class ResourceNode:
    name: str
    kind: str           # berries, fish, water, wood, stone, herbs
    x: float
    y: float
    amount: int
    max_amount: int
    replenish_every: int
    claimed_by: str = None      # agent_id or None
    _counter: int = 0

    def tick(self):
        if self.amount < self.max_amount:
            self._counter += 1
            if self._counter >= self.replenish_every:
                self.amount = min(self.max_amount, self.amount + 1)
                self._counter = 0

    def harvest(self, amount: int = 1) -> int:
        """Take up to amount, return how much was taken."""
        taken = min(amount, self.amount)
        self.amount -= taken
        return taken

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "amount": self.amount,
            "max_amount": self.max_amount,
            "claimed_by": self.claimed_by,
        }


def create_world_resources() -> list[ResourceNode]:
    """Place resource nodes around the 60x60 world."""
    return [
        # Food
        ResourceNode("Berry Bushes",    "berries", 15, 15, 6, 8,  RESOURCE_REPLENISH["berries"]),
        ResourceNode("Berry Patch",     "berries", 45, 45, 5, 8,  RESOURCE_REPLENISH["berries"]),
        ResourceNode("Fishing Hole",    "fish",    30, 10, 4, 6,  RESOURCE_REPLENISH["fish"]),
        ResourceNode("River Bend",      "fish",    10, 40, 4, 6,  RESOURCE_REPLENISH["fish"]),
        # Water
        ResourceNode("Spring",          "water",   30, 30, 8, 10, RESOURCE_REPLENISH["water"]),
        ResourceNode("Muddy Puddle",    "water",   50, 20, 3, 5,  RESOURCE_REPLENISH["water"]),
        # Crafting materials
        ResourceNode("Dead Forest",     "wood",    20, 50, 5, 7,  RESOURCE_REPLENISH["wood"]),
        ResourceNode("Rocky Outcrop",   "stone",   50, 10, 4, 5,  RESOURCE_REPLENISH["stone"]),
        # Herbs (medicine crafting)
        ResourceNode("Herb Garden",     "herbs",   40, 35, 3, 5,  RESOURCE_REPLENISH["herbs"]),
    ]


class CraftingSystem:
    """Agents discover recipes through experimentation."""

    def __init__(self):
        self.discovered_recipes: dict = {}  # agent_id → set of recipe keys
        self.global_knowledge: set = set()  # recipes any agent knows (can spread)

    def attempt_craft(self, agent_id: str, item_a: str, item_b: str) -> str | None:
        """Try to combine two items. Returns result name or None."""
        key = tuple(sorted([item_a, item_b]))
        result = CRAFTING_RECIPES.get(key)
        if result:
            if agent_id not in self.discovered_recipes:
                self.discovered_recipes[agent_id] = set()
            is_new = key not in self.discovered_recipes[agent_id]
            self.discovered_recipes[agent_id].add(key)
            self.global_knowledge.add(key)
            return result, is_new
        return None, False

    def agent_knows(self, agent_id: str, item_a: str, item_b: str) -> bool:
        key = tuple(sorted([item_a, item_b]))
        return key in self.discovered_recipes.get(agent_id, set())

    def share_recipe(self, from_id: str, to_id: str, item_a: str, item_b: str):
        """Agent shares knowledge of a recipe with another."""
        key = tuple(sorted([item_a, item_b]))
        if to_id not in self.discovered_recipes:
            self.discovered_recipes[to_id] = set()
        self.discovered_recipes[to_id].add(key)

    def known_recipes(self, agent_id: str) -> list[str]:
        recipes = self.discovered_recipes.get(agent_id, set())
        return [f"{a}+{b}={CRAFTING_RECIPES[k]}" for k in recipes
                for a, b in [k] if k in CRAFTING_RECIPES]
