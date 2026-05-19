from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from api.websocket import manager

router = APIRouter()
_world_ref = None

def set_world(world):
    global _world_ref
    _world_ref = world


@router.get("/")
async def root():
    if not _world_ref:
        return {"status": "starting"}
    return {
        "status": "running",
        "tick": _world_ref.tick_number,
        "agents": len([a for a in _world_ref.agents.values() if a.alive]),
    }


@router.get("/world")
async def get_world():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return _world_ref.to_snapshot()


@router.get("/agents")
async def get_agents():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return [a.to_dict() for a in _world_ref.agents.values() if a.alive]


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    agent = _world_ref.agents.get(agent_id)
    if not agent:
        return JSONResponse({"error": "not found"}, status_code=404)

    rels = {}
    for other_id, rel in agent.relationships.items():
        other = _world_ref.agents.get(other_id)
        rels[other_id] = {
            "name": other.name if other else other_id,
            "label": rel.label(),
            "bond": round(rel.bond_score(), 3),
            "trust": round(rel.trust, 3),
            "love": round(rel.love, 3),
            "rivalry": round(rel.rivalry, 3),
            "fear": round(rel.fear, 3),
            "debt": round(rel.debt, 3),
            "encounters": rel.encounters,
        }

    return {
        **agent.to_dict(),
        "memory": agent.memory,
        "relationships": rels,
        "known_recipes": _world_ref.crafting.known_recipes(agent_id),
    }


@router.get("/resources")
async def get_resources():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    result = []
    for r in _world_ref.resources:
        claimer = _world_ref.agents.get(r.claimed_by) if r.claimed_by else None
        result.append({**r.to_dict(), "claimer_name": claimer.name if claimer else None})
    return result


@router.get("/relationships")
async def get_relationships():
    """Social graph for UI visualization."""
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    edges = []
    seen = set()
    for agent in _world_ref.agents.values():
        for other_id, rel in agent.relationships.items():
            pair = tuple(sorted([agent.id, other_id]))
            if pair in seen:
                continue
            seen.add(pair)
            other = _world_ref.agents.get(other_id)
            bond = rel.bond_score()
            if abs(bond) < 0.05:
                continue
            edges.append({
                "source": agent.id, "source_name": agent.name,
                "target": other_id, "target_name": other.name if other else other_id,
                "bond": round(bond, 3),
                "label": rel.label(),
                "trust": round(rel.trust, 3),
                "love": round(rel.love, 3),
                "rivalry": round(rel.rivalry, 3),
                "encounters": rel.encounters,
            })
    return sorted(edges, key=lambda e: -abs(e["bond"]))


@router.get("/economy")
async def get_economy():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return {
        "market": _world_ref.market.to_dict(),
        "wealth_distribution": {
            a.name: a.wealth()
            for a in sorted(_world_ref.agents.values(), key=lambda x: -x.wealth())
            if a.alive
        },
        "inventory_totals": _get_inventory_totals(),
        "known_recipes": {
            aid: _world_ref.crafting.known_recipes(aid)
            for aid in _world_ref.agents
        },
    }


@router.get("/society")
async def get_society():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return {
        "weather": _world_ref.weather.to_dict(),
        "groups": [g.to_dict(_world_ref) for g in _world_ref.groups.values()],
        "projects": [p.to_dict(_world_ref) for p in _world_ref.projects],
        "rumors": [
            {
                "tick": r["tick"],
                "type": r["type"],
                "summary": r["summary"],
                "heard_count": len(r["heard_by"]),
                "heat": r["heat"],
            }
            for r in _world_ref.society.rumors[-50:]
        ],
    }


@router.get("/groups")
async def get_groups():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return [g.to_dict(_world_ref) for g in _world_ref.groups.values()]


@router.get("/projects")
async def get_projects():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return [p.to_dict(_world_ref) for p in _world_ref.projects]


@router.get("/metrics")
async def get_metrics():
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return {
        "tick": _world_ref.tick_number,
        "metrics": dict(sorted(_world_ref.metrics.items())),
        "alive_agents": len([a for a in _world_ref.agents.values() if a.alive]),
        "event_count": len(_world_ref.events),
        "groups": len(_world_ref.groups),
        "projects": len(_world_ref.projects),
        "weather": _world_ref.weather.to_dict(),
    }


def _get_inventory_totals() -> dict:
    totals = {}
    for agent in _world_ref.agents.values():
        if agent.alive:
            for item, amount in agent.inventory.items():
                totals[item] = totals.get(item, 0) + amount
    return totals


@router.get("/events")
async def get_events(limit: int = 100):
    if not _world_ref:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return _world_ref.events[-limit:]


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    if _world_ref:
        await websocket.send_json(_world_ref.to_snapshot())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
