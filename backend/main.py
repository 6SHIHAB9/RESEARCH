import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.world import World
from core.simulation import simulation_loop, set_broadcast
from api.routes import router, set_world
from api.websocket import broadcast_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [TICK] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

world = World()


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_world(world)
    set_broadcast(broadcast_snapshot)
    task = asyncio.create_task(simulation_loop(world))
    yield
    task.cancel()
    logger.info("🌙 Civilization shutting down...")


app = FastAPI(
    title="Civilization Simulation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # UI can connect from anywhere
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
