from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
_world_ref = None

def set_world(world):
    global _world_ref
    _world_ref = world


@router.get("/")
async def root():
    return {"status": "civilization running", "version": "0.2.0"}


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
    
    # Serialize relationships with full detail
    relationships_detail = {}
    for other_id, rel in agent.relationships.items():
        other = _world_ref.agents.get(other_id)
        relationships_detail[other_id] = {
            "name": other.name if other else other_id,
            "trust": round(rel.trust, 3),
            "fear": round(rel.fear, 3),
            "affinity": round(rel.affinity, 3),
            "hostility": round(rel.hostility, 3),
            "net_bond": round(rel.net_bond(), 3),
            "label": rel.label(),
            "encounters": rel.encounters,
        }

    return {
        **agent.to_dict(),
        "memory": agent.memory,
        "relationships": relationships_detail,
        "recent_phrases": agent.recent_phrases,
    }


@router.get("/events")
async def get_events(limit: int = 50):
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    return _world_ref.event_log[-limit:]


@router.get("/resources")
async def get_resources():
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    result = []
    for r in _world_ref.resources:
        claimer_name = None
        if r.claimed_by:
            agent = _world_ref.agents.get(r.claimed_by)
            claimer_name = agent.name if agent else r.claimed_by
        result.append({
            "name": r.name,
            "x": r.x,
            "y": r.y,
            "kind": r.kind,
            "amount": r.amount,
            "max_amount": r.max_amount,
            "claimed_by": r.claimed_by,
            "claimer_name": claimer_name,
        })
    return result


@router.get("/relationships")
async def get_all_relationships():
    """Return a graph of significant relationships for visualization."""
    if not _world_ref:
        return JSONResponse({"error": "world not initialized"}, status_code=503)
    edges = []
    seen = set()
    for agent in _world_ref.agents.values():
        for other_id, rel in agent.relationships.items():
            pair = tuple(sorted([agent.id, other_id]))
            if pair in seen or abs(rel.net_bond()) < 0.1:
                continue
            seen.add(pair)
            other = _world_ref.agents.get(other_id)
            edges.append({
                "source": agent.id,
                "source_name": agent.name,
                "target": other_id,
                "target_name": other.name if other else other_id,
                "bond": round(rel.net_bond(), 3),
                "label": rel.label(),
                "encounters": rel.encounters,
            })
    edges.sort(key=lambda e: -abs(e["bond"]))
    return edges[:100]