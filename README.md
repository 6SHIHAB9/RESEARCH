# Persistent AI Civilization — Backend

An emergent social simulation. 25 autonomous agents exist in a shared world.
They perceive nearby entities, remember fragments of interactions, react emotionally, and form habits.
No goals. No objectives. Just existence.

## Architecture

```
backend/
├── main.py              # FastAPI app + WebSocket server
├── core/
│   ├── world.py         # World state, agent registry, perception
│   └── simulation.py    # Async tick loop, sleep phase, memory compression
├── agents/
│   ├── agent.py         # Agent data model (mood, memory, bonds, movement)
│   └── names.py         # Name pool + personality archetypes
├── llm/
│   └── batch_prompt.py  # ONE batched Cerebras call per tick
└── api/
    └── routes.py        # REST endpoints for world inspection
```

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your CEREBRAS_API_KEY
bash start.sh
```

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /world` | Full world snapshot |
| `GET /agents` | All agents |
| `GET /agents/{id}` | Agent detail + full memory |
| `GET /events` | Recent event log |
| `WS /ws` | Live world updates |

## How It Works

**Tick cycle (every 45s):**
1. Select 8 random agents to process
2. Each agent perceives nearby agents within radius 18
3. Build compact context: personality, mood, nearby faces, recent memories
4. Send ONE batched prompt to Cerebras (`llama3.1-8b`)
5. Parse JSON response → apply movement, mood shifts, bond changes, memory notes
6. Broadcast updated world snapshot to all WebSocket observers

**Sleep phase (4h/day):**
- Agents stop acting
- Memories compress: keep oldest 2 + newest 4 entries
- World state persists

**Emergence happens through:**
- Repeated proximity → bond formation
- Emotional contagion via nearby mood states
- Memory influencing who agents approach/avoid
- Controlled randomness (temperature 1.15, random wandering)

## Token Budget

~600 tokens per prompt for 8 agents. At one tick per 45s:
- ~80 ticks/hour → ~48k tokens/hour
- Well within Cerebras free tier limits

## Next: Frontend

React canvas observer UI showing:
- Moving agent dots (colored by mood)
- Social bond lines forming/fading
- Live interaction feed
- Cluster formation over time
