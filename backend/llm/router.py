import asyncio
import logging
import os
from openai import AsyncOpenAI
from agents.agent import Agent
from core.world import World
from llm.prompt import build_agent_prompt
from llm.parser import parse_agent_response
from config import CEREBRAS_API_KEYS, CEREBRAS_BASE_URL, MODEL, AGENTS_PER_KEY

logger = logging.getLogger(__name__)


def _get_client(key_index: int) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=CEREBRAS_API_KEYS[key_index],
        base_url=CEREBRAS_BASE_URL,
    )


async def _call_single_agent(
    agent: Agent,
    world: World,
    client: AsyncOpenAI,
    key_index: int,
) -> tuple[str, dict | None]:
    """Fire one LLM call for one agent. Returns (agent_id, parsed_result)."""
    prompt = build_agent_prompt(agent, world)
    nearby = world.nearby_agents(agent)
    valid_target_ids = [a.id for a in nearby]

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.1,
            max_tokens=400,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = parse_agent_response(raw, agent.id, valid_target_ids)
        return agent.id, result
    except Exception as e:
        logger.error(f"  LLM error for {agent.name} (key {key_index}): {e}")
        return agent.id, None


async def _call_group(
    agents: list[Agent],
    world: World,
    key_index: int,
) -> list[tuple[str, dict | None]]:
    """Call all agents in a group sequentially using one API key."""
    client = _get_client(key_index)
    results = []
    for agent in agents:
        result = await _call_single_agent(agent, world, client, key_index)
        results.append(result)
    return results


async def run_tick(world: World) -> list[tuple[str, dict | None]]:
    """
    Split 12 agents across 3 API keys, fire all groups in parallel.
    Returns list of (agent_id, parsed_result) for all agents.
    """
    all_agents = [a for a in world.agents.values() if a.alive]

    # Split into groups of AGENTS_PER_KEY (4)
    groups = []
    for i in range(0, len(all_agents), AGENTS_PER_KEY):
        groups.append(all_agents[i:i + AGENTS_PER_KEY])

    # Assign each group to an API key (cycle if more groups than keys)
    num_keys = len(CEREBRAS_API_KEYS)
    tasks = [
        _call_group(group, world, i % num_keys)
        for i, group in enumerate(groups)
    ]

    logger.info(f"  → Firing {len(groups)} parallel groups ({len(all_agents)} agents)...")

    # All groups fire simultaneously
    group_results = await asyncio.gather(*tasks)

    # Flatten
    all_results = []
    for group in group_results:
        all_results.extend(group)

    return all_results
