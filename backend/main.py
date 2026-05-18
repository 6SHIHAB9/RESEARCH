import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core.world import World
from core.simulation import SimulationLoop
from api.routes import router, set_world
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TICK] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

world = World()
simulation = SimulationLoop(world)
set_world(world)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌍 Civilization awakening...")
    task = asyncio.create_task(simulation.run())
    yield
    task.cancel()
    logger.info("🌙 Civilization entering sleep...")

app = FastAPI(title="Civilization API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

connected_observers: list[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_observers.append(websocket)
    logger.info(f"👁  Observer connected. Total: {len(connected_observers)}")
    try:
        # Send current world state immediately on connect
        await websocket.send_json(world.to_snapshot())
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        connected_observers.remove(websocket)
        logger.info(f"👁  Observer left. Total: {len(connected_observers)}")

# Inject observer broadcast into simulation
simulation.broadcast_fn = lambda snapshot: asyncio.gather(
    *[ws.send_json(snapshot) for ws in connected_observers],
    return_exceptions=True
)
