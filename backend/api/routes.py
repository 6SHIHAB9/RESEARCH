from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# These are injected by main.py
_world_ref = None

def set_world(world):
    global _world_ref
    _world_ref = world


@router.get("/")
async def root():
    return {"status": "civilization running", "version": "0.1.0"}


@router.get("/world")
async def get_world():
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    return _world_ref.to_snapshot()


@router.get("/agents")
async def get_agents():
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    return [a.to_dict() for a in _world_ref.agents.values()]


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    agent = _world_ref.agents.get(agent_id)
    if not agent:
        return JSONResponse({"error": "agent not found"}, status_code=404)
    return {
        **agent.to_dict(),
        "memory": agent.memory,
        "social_bonds": agent.social_bonds,
        "recent_phrases": agent.recent_phrases,
    }


@router.get("/events")
async def get_events(limit: int = 50):
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    return _world_ref.event_log[-limit:]
