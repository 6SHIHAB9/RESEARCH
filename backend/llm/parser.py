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
    # FIX: repair unescaped double quotes inside string values
    # e.g. "phrase": "he said "hello" to me" → "phrase": "he said \"hello\" to me"
    def fix_inner_quotes(m):
        key = m.group(1)
        value = m.group(2)
        # Escape any unescaped quotes inside the value
        value = re.sub(r'(?<!\\)"', '\\"', value)
        return f'"{key}": "{value}"'
    raw = re.sub(r'"(\w+)":\s*"(.*?)"(?=\s*[,\}])', fix_inner_quotes, raw, flags=re.DOTALL)
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
        # FIX: last resort — try extracting just the fields we need with regex
        logger.warning(f"  Parse error for {agent_id}: {e} — trying field extraction")
        data = _extract_fields(raw)
        if not data:
            logger.error(f"  Could not recover JSON for {agent_id}")
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


def _extract_fields(raw: str) -> dict | None:
    """
    FIX: Last-resort field extractor when JSON is broken beyond sanitization.
    Pulls scalar fields out with regex — good enough for action/dx/dy/phrase.
    """
    def get_str(key):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', raw)
        return m.group(1) if m else None

    def get_num(key):
        m = re.search(rf'"{key}"\s*:\s*(-?[\d.]+)', raw)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None

    action = get_str("action")
    if not action:
        return None  # Can't do anything without an action

    return {
        "action":        action,
        "target_id":     get_str("target_id"),
        "phrase":        get_str("phrase"),
        "dx":            get_num("dx") or 0.0,
        "dy":            get_num("dy") or 0.0,
        "mood_delta":    get_num("mood_delta") or 0.0,
        "resource_name": get_str("resource_name"),
        "give_items":    {},
        "receive_items": {},
        "craft_a":       get_str("craft_a"),
        "craft_b":       get_str("craft_b"),
        "project_id":    get_str("project_id"),
        "memory_note":   get_str("memory_note"),
        "rel_updates":   {},
    }