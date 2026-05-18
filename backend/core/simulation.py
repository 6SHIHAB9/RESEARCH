import asyncio
import logging
import time
import random
from core.world import World
from llm.batch_prompt import run_tick_batch

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 45       # seconds between ticks
ACTIVE_HOURS_PER_DAY = 20        # hours civilization is "awake"
SLEEP_HOURS_PER_DAY = 4
DAY_SECONDS = (ACTIVE_HOURS_PER_DAY + SLEEP_HOURS_PER_DAY) * 3600


class SimulationLoop:
    def __init__(self, world: World):
        self.world = world
        self.broadcast_fn = None
        self.is_sleeping = False
        self._start_time = time.time()

    def _should_sleep(self) -> bool:
        elapsed = (time.time() - self._start_time) % DAY_SECONDS
        active_seconds = ACTIVE_HOURS_PER_DAY * 3600
        return elapsed >= active_seconds

    async def run(self):
        logger.info(f"⚙️  Simulation loop started. Tick interval: {TICK_INTERVAL_SECONDS}s")
        while True:
            if self._should_sleep():
                if not self.is_sleeping:
                    self.is_sleeping = True
                    logger.info("🌙 Sleep phase began. Agents resting.")
                    await self._compress_memories()
                await asyncio.sleep(60)
                continue

            if self.is_sleeping:
                self.is_sleeping = False
                logger.info("☀️  Active phase resumed.")

            await self._tick()
            await asyncio.sleep(TICK_INTERVAL_SECONDS)

    async def _tick(self):
        self.world.tick_number += 1
        tick = self.world.tick_number
        logger.info(f"━━━ TICK {tick} ━━━")

        try:
            await run_tick_batch(self.world)
        except Exception as e:
            logger.error(f"LLM batch error on tick {tick}: {e}")

        # Broadcast to all observers
        if self.broadcast_fn:
            snapshot = self.world.to_snapshot()
            await self.broadcast_fn(snapshot)

        logger.info(f"✓ Tick {tick} complete. Agents: {len(self.world.agents)}")

    async def _compress_memories(self):
        """During sleep: summarize long memory chains to save tokens."""
        compressed = 0
        for agent in self.world.agents.values():
            if len(agent.memory) > 6:
                # Keep first 2 (oldest context) + last 4 (recent)
                agent.memory = agent.memory[:2] + agent.memory[-4:]
                compressed += 1
        if compressed:
            logger.info(f"💭 Memory compressed for {compressed} agents.")
