import json
import re
import logging

logger = logging.getLogger(__name__)

VALID_ACTIONS = [
    "speak", "forage", "claim", "trade", "craft",
    "build", "rest", "wander", "observe", "confront", "retreat", "give", "ignore"
]


def safe_float(value, default=0.0, lo=-10.0, hi=10.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def safe_str(value, default="") -> str:
    return str(value) if value is not None else default


def safe_choice(value, allowed: list, default: str) -> str:
    return value if value in allowed else default


def sanitize(raw: str) -> str:
    """Fix common LLM JSON mistakes."""
    # Strip markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()
    # Extract first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    # Fix lone backslashes
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    # Fix trailing commas
    raw = re.sub(r',\s*([\]}])', r'\1', raw)
    return raw


def parse_agent_response(raw: str, agent_id: str, valid_target_ids: list) -> dict | None:
    """Parse a single agent LLM response into a clean action dict."""
    if not raw or not raw.strip():
        logger.warning(f"  Empty response for {agent_id}")
        return None

    try:
        clean = sanitize(raw)
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"  Parse error for {agent_id}: {e}")
        logger.debug(f"  Raw: {raw[:200]}")
        return None

    # Validate target
    raw_target = data.get("target_id")
    target_id = raw_target if (raw_target in valid_target_ids and raw_target != agent_id) else None

    # Clamp movement — prevent teleporting
    dx = safe_float(data.get("dx"), 0.0, -2.0, 2.0)
    dy = safe_float(data.get("dy"), 0.0, -2.0, 2.0)

    return {
        "action":        safe_choice(data.get("action"), VALID_ACTIONS, "wander"),
        "target_id":     target_id,
        "phrase":        data.get("phrase"),
        "dx":            dx,
        "dy":            dy,
        "mood_delta":    safe_float(data.get("mood_delta"), 0.0, -0.3, 0.3),
        "resource_name": data.get("resource_name"),
        "give_items":    data.get("give_items") or {},
        "receive_items": data.get("receive_items") or {},
        "craft_a":       data.get("craft_a"),
        "craft_b":       data.get("craft_b"),
        "project_id":    data.get("project_id"),
        "memory_note":   data.get("memory_note"),
        "rel_updates":   data.get("rel_updates") or {},
    }
