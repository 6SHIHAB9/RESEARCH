import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
CEREBRAS_API_KEYS = [
    os.getenv("CEREBRAS_API_KEY_1"),
    os.getenv("CEREBRAS_API_KEY_2"),
    os.getenv("CEREBRAS_API_KEY_3"),
]
CEREBRAS_API_KEYS = [k for k in CEREBRAS_API_KEYS if k]

if not CEREBRAS_API_KEYS:
    raise RuntimeError("No API keys found. Set CEREBRAS_API_KEY_1 (and optionally _2, _3) in .env")

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# llama3.1-8b  — fast, reliable, free tier works fine, ~2300 tok/s
# llama-3.3-70b — smarter but slower, still reliable
# gpt-oss-120b  — DO NOT USE: free-tier rate limits severely reduced
MODEL = "llama3.1-8b"

# ── Simulation ────────────────────────────────────────────────────────────────
TICK_INTERVAL_SECONDS = int(os.getenv("TICK_INTERVAL_SECONDS", "20"))
AGENT_COUNT = 12
AGENTS_PER_KEY = 4

# ── World ─────────────────────────────────────────────────────────────────────
WORLD_WIDTH = 60
WORLD_HEIGHT = 60
PERCEPTION_RADIUS = 15

# ── Needs decay per tick ──────────────────────────────────────────────────────
HUNGER_DECAY     = 0.04
THIRST_DECAY     = 0.05
ENERGY_DECAY     = 0.03
LONELINESS_DECAY = 0.02

# ── Resource replenish (ticks between +1) ─────────────────────────────────────
RESOURCE_REPLENISH = {
    "berries": 6,
    "fish":    8,
    "water":   4,
    "wood":    12,
    "stone":   20,
    "herbs":   15,
}

# ── Crafting recipes ──────────────────────────────────────────────────────────
CRAFTING_RECIPES = {
    ("wood", "stone"): "tool",
    ("herbs", "water"): "medicine",
    ("fish", "herbs"):  "preserved_food",
}

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_EVENTS       = 500
MAX_AGENT_MEMORY = 20