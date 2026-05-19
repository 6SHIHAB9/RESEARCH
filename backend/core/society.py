import random
from dataclasses import dataclass, field


WEATHER_STATES = {
    "clear": {
        "label": "clear",
        "thirst": 1.0,
        "energy": 1.0,
        "resource": 1.0,
        "mood": 0.0,
    },
    "hot": {
        "label": "heat wave",
        "thirst": 1.45,
        "energy": 1.2,
        "resource": 0.85,
        "mood": -0.02,
    },
    "rain": {
        "label": "rain",
        "thirst": 0.75,
        "energy": 1.05,
        "resource": 1.25,
        "mood": 0.01,
    },
    "cold": {
        "label": "cold snap",
        "thirst": 0.9,
        "energy": 1.35,
        "resource": 0.9,
        "mood": -0.02,
    },
    "storm": {
        "label": "storm",
        "thirst": 0.7,
        "energy": 1.55,
        "resource": 0.7,
        "mood": -0.04,
    },
}


@dataclass
class WeatherSystem:
    state: str = "clear"
    remaining_ticks: int = 8

    def tick(self, world) -> list[dict]:
        self.remaining_ticks -= 1
        if self.remaining_ticks > 0:
            return []

        old_state = self.state
        self.state = random.choices(
            ["clear", "hot", "rain", "cold", "storm"],
            weights=[42, 18, 18, 14, 8],
            k=1,
        )[0]
        self.remaining_ticks = random.randint(5, 12)

        if self.state == old_state:
            return []

        return [{
            "type": "weather_shift",
            "weather": self.label,
            "previous": WEATHER_STATES[old_state]["label"],
        }]

    @property
    def profile(self) -> dict:
        return WEATHER_STATES[self.state]

    @property
    def label(self) -> str:
        return self.profile["label"]

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "label": self.label,
            "remaining_ticks": self.remaining_ticks,
            "modifiers": {
                "thirst": self.profile["thirst"],
                "energy": self.profile["energy"],
                "resource": self.profile["resource"],
                "mood": self.profile["mood"],
            },
        }


@dataclass
class SocialGroup:
    id: str
    name: str
    founder_id: str
    anchor_name: str
    x: float
    y: float
    members: set[str] = field(default_factory=set)
    stash: dict = field(default_factory=dict)
    created_tick: int = 0
    cohesion: float = 0.5
    ethos: str = "survival"

    def to_dict(self, world) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "anchor_name": self.anchor_name,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "members": [
                {"id": agent_id, "name": world.agents[agent_id].name}
                for agent_id in sorted(self.members)
                if agent_id in world.agents
            ],
            "member_count": len(self.members),
            "stash": dict(sorted(self.stash.items())),
            "created_tick": self.created_tick,
            "cohesion": round(self.cohesion, 2),
            "ethos": self.ethos,
        }


@dataclass
class GroupProject:
    id: str
    group_id: str
    kind: str
    name: str
    required: dict
    contributed: dict = field(default_factory=dict)
    complete: bool = False
    started_tick: int = 0
    completed_tick: int | None = None

    def add(self, item: str, amount: int) -> int:
        if self.complete or amount <= 0:
            return 0
        needed = self.required.get(item, 0) - self.contributed.get(item, 0)
        taken = min(max(0, needed), amount)
        if taken:
            self.contributed[item] = self.contributed.get(item, 0) + taken
        return taken

    def is_ready(self) -> bool:
        return all(self.contributed.get(item, 0) >= amount for item, amount in self.required.items())

    def to_dict(self, world) -> dict:
        group = world.groups.get(self.group_id)
        return {
            "id": self.id,
            "group_id": self.group_id,
            "group_name": group.name if group else self.group_id,
            "kind": self.kind,
            "name": self.name,
            "required": self.required,
            "contributed": self.contributed,
            "complete": self.complete,
            "started_tick": self.started_tick,
            "completed_tick": self.completed_tick,
        }


class SocietySystem:
    def __init__(self):
        self.rumors: list[dict] = []
        self._next_group_id = 1
        self._next_project_id = 1

    def tick(self, world) -> list[dict]:
        events = []
        events.extend(self._relationship_drift(world))
        events.extend(self._form_groups(world))
        events.extend(self._update_groups(world))
        events.extend(self._spread_rumors(world))
        events.extend(self._advance_projects(world))
        return events

    def remember_event(self, world, event: dict):
        event_type = event.get("type")
        if event_type not in {
            "speech",
            "trade",
            "give",
            "confrontation",
            "territory_claimed",
            "territory_seized",
            "resource_conflict",
            "discovery",
            "camp_founded",
            "project_completed",
            "weather_shift",
        }:
            return
        rumor = {
            "tick": world.tick_number,
            "type": event_type,
            "summary": self._summarize_event(event),
            "heard_by": set(),
            "heat": 5,
        }
        for key in ("agent_id", "target_id", "from_id", "to_id"):
            agent_id = event.get(key)
            if agent_id:
                rumor["heard_by"].add(agent_id)
        self.rumors.append(rumor)
        self.rumors = self.rumors[-80:]

    def start_project(self, world, group_id: str, kind: str = "shelter") -> GroupProject | None:
        group = world.groups.get(group_id)
        if not group:
            return None
        active = [
            project for project in world.projects
            if project.group_id == group_id and project.kind == kind and not project.complete
        ]
        if active:
            return active[0]

        templates = {
            "shelter": ("Shared Shelter", {"wood": 5, "stone": 2}),
            "watchpost": ("Watchpost", {"wood": 3, "stone": 3}),
            "storehouse": ("Storehouse", {"wood": 4, "stone": 1}),
        }
        name, required = templates.get(kind, templates["shelter"])
        project = GroupProject(
            id=f"project_{self._next_project_id:03d}",
            group_id=group_id,
            kind=kind,
            name=name,
            required=required,
            started_tick=world.tick_number,
        )
        self._next_project_id += 1
        world.projects.append(project)
        world.log("project_started", {
            "group_id": group.id,
            "group": group.name,
            "project_id": project.id,
            "project": project.name,
        })
        return project

    def contribute_to_project(self, world, agent, project_id: str | None = None) -> bool:
        group_id = agent.home_group
        if not group_id:
            return False

        project = None
        if project_id:
            project = next((p for p in world.projects if p.id == project_id and not p.complete), None)
        if not project:
            project = next(
                (p for p in world.projects if p.group_id == group_id and not p.complete),
                None,
            )
        if not project:
            project = self.start_project(world, group_id)
        if not project:
            return False

        any_contribution = False
        for item in list(project.required):
            available = int(agent.inventory.get(item, 0))
            taken = project.add(item, available)
            if taken:
                agent.inventory[item] -= taken
                agent.remember(world.tick_number, f"I gave {taken} {item} to {project.name}.")
                any_contribution = True

        if any_contribution:
            world.log("project_contribution", {
                "agent": agent.name,
                "agent_id": agent.id,
                "project_id": project.id,
                "project": project.name,
            })
        return any_contribution

    def _relationship_drift(self, world) -> list[dict]:
        events = []
        agents = [a for a in world.agents.values() if a.alive]
        for i, agent in enumerate(agents):
            for other in agents[i + 1:]:
                dist = world.distance(agent, other)
                if dist > 5:
                    continue
                if agent.needs.anger > 0.65 or other.needs.anger > 0.65:
                    agent.update_rel(other.id, rivalry=0.01)
                    other.update_rel(agent.id, rivalry=0.01)
                else:
                    gain = 0.006 + ((agent.traits.get("empathy", 0.5) + other.traits.get("empathy", 0.5)) * 0.004)
                    agent.update_rel(other.id, trust=gain)
                    other.update_rel(agent.id, trust=gain)

                if random.random() < 0.025:
                    events.append({
                        "type": "bond_shift",
                        "agent": agent.name,
                        "agent_id": agent.id,
                        "target": other.name,
                        "target_id": other.id,
                        "bond": round(agent.get_rel(other.id).bond_score(), 2),
                    })
        return events

    def _form_groups(self, world) -> list[dict]:
        events = []
        ungrouped = [a for a in world.agents.values() if a.alive and not a.home_group]
        for agent in ungrouped:
            if agent.social_status < 0.35 and agent.traits.get("empathy", 0.5) < 0.45:
                continue
            nearby = [
                other for other in world.nearby_agents(agent, radius=8)
                if not other.home_group and agent.get_rel(other.id).bond_score() > 0.12
            ]
            if len(nearby) < 2:
                continue
            anchor = world.nearest_landmark(agent.x, agent.y)
            group = SocialGroup(
                id=f"group_{self._next_group_id:03d}",
                name=self._group_name(anchor["name"], agent.traits),
                founder_id=agent.id,
                anchor_name=anchor["name"],
                x=anchor["x"],
                y=anchor["y"],
                members={agent.id, nearby[0].id, nearby[1].id},
                created_tick=world.tick_number,
                ethos=self._ethos(agent.traits),
            )
            self._next_group_id += 1
            world.groups[group.id] = group
            for member_id in group.members:
                world.agents[member_id].home_group = group.id
                world.agents[member_id].remember(world.tick_number, f"We formed {group.name}.")
            events.append({
                "type": "camp_founded",
                "group": group.name,
                "group_id": group.id,
                "agent": agent.name,
                "agent_id": agent.id,
                "anchor": group.anchor_name,
                "members": [world.agents[mid].name for mid in group.members],
            })
        return events

    def _update_groups(self, world) -> list[dict]:
        events = []
        for group in world.groups.values():
            members = [world.agents[mid] for mid in group.members if mid in world.agents and world.agents[mid].alive]
            if not members:
                continue

            # Invite trusted nearby agents.
            for member in members:
                for other in world.nearby_agents(member, radius=7):
                    if other.home_group or member.get_rel(other.id).bond_score() < 0.25:
                        continue
                    other.home_group = group.id
                    group.members.add(other.id)
                    other.remember(world.tick_number, f"I joined {group.name}.")
                    events.append({
                        "type": "camp_joined",
                        "group": group.name,
                        "group_id": group.id,
                        "agent": other.name,
                        "agent_id": other.id,
                    })

            # Surplus becomes a shared stash, which makes groups feel materially real.
            for member in members:
                for item, amount in list(member.inventory.items()):
                    if amount > 2 and item in {"berries", "fish", "water", "wood", "stone", "herbs"}:
                        deposit = 1
                        member.inventory[item] -= deposit
                        group.stash[item] = group.stash.get(item, 0) + deposit

            # Hungry/thirsty members can draw from stash.
            for member in members:
                if member.needs.hunger > 0.68:
                    self._withdraw(group, member, ["berries", "fish"], "hunger", 0.22)
                if member.needs.thirst > 0.68:
                    self._withdraw(group, member, ["water"], "thirst", 0.28)

            cohesion = self._cohesion(world, group)
            if cohesion < 0.16 and len(group.members) > 2:
                weakest = min(members, key=lambda a: self._member_fit(world, group, a))
                group.members.remove(weakest.id)
                weakest.home_group = None
                weakest.remember(world.tick_number, f"I left {group.name}.")
                events.append({
                    "type": "camp_left",
                    "group": group.name,
                    "group_id": group.id,
                    "agent": weakest.name,
                    "agent_id": weakest.id,
                })
            group.cohesion = cohesion

            if len(group.members) >= 4 and not any(p.group_id == group.id and not p.complete for p in world.projects):
                if random.random() < 0.08:
                    self.start_project(world, group.id, random.choice(["shelter", "watchpost", "storehouse"]))
        return events

    def _spread_rumors(self, world) -> list[dict]:
        events = []
        for rumor in self.rumors:
            if rumor["heat"] <= 0:
                continue
            hearers = [world.agents[aid] for aid in rumor["heard_by"] if aid in world.agents]
            for hearer in hearers:
                for other in world.nearby_agents(hearer, radius=6):
                    if other.id in rumor["heard_by"]:
                        continue
                    chance = 0.05 + hearer.traits.get("empathy", 0.5) * 0.05
                    if random.random() > chance:
                        continue
                    rumor["heard_by"].add(other.id)
                    other.remember(world.tick_number, f"Rumor: {rumor['summary']}")
                    events.append({
                        "type": "rumor_spread",
                        "agent": hearer.name,
                        "agent_id": hearer.id,
                        "target": other.name,
                        "target_id": other.id,
                        "rumor": rumor["summary"],
                    })
            rumor["heat"] -= 1
        self.rumors = [r for r in self.rumors if r["heat"] > 0 or world.tick_number - r["tick"] < 20]
        return events

    def _advance_projects(self, world) -> list[dict]:
        events = []
        for project in world.projects:
            if project.complete:
                continue
            group = world.groups.get(project.group_id)
            if not group:
                continue
            for item in list(project.required):
                available = group.stash.get(item, 0)
                taken = project.add(item, available)
                if taken:
                    group.stash[item] -= taken
            if project.is_ready():
                project.complete = True
                project.completed_tick = world.tick_number
                self._apply_project_bonus(world, group, project)
                events.append({
                    "type": "project_completed",
                    "group": group.name,
                    "group_id": group.id,
                    "project": project.name,
                    "project_id": project.id,
                })
        return events

    def _apply_project_bonus(self, world, group: SocialGroup, project: GroupProject):
        for member_id in group.members:
            agent = world.agents.get(member_id)
            if not agent:
                continue
            if project.kind == "shelter":
                agent.needs.energy = min(1.0, agent.needs.energy + 0.18)
                agent.nudge_mood(0.08)
            elif project.kind == "watchpost":
                agent.needs.fear = max(0.0, agent.needs.fear - 0.16)
                agent.nudge_mood(0.04)
            elif project.kind == "storehouse":
                agent.social_status = min(1.0, agent.social_status + 0.04)
            agent.remember(world.tick_number, f"{group.name} finished {project.name}.")

    def _withdraw(self, group: SocialGroup, member, items: list[str], need_name: str, relief: float):
        for item in items:
            if group.stash.get(item, 0) <= 0:
                continue
            group.stash[item] -= 1
            member.inventory[item] = member.inventory.get(item, 0) + 1
            setattr(member.needs, need_name, max(0.0, getattr(member.needs, need_name) - relief))
            member.nudge_mood(0.04)
            return True
        return False

    def _cohesion(self, world, group: SocialGroup) -> float:
        members = [world.agents[mid] for mid in group.members if mid in world.agents and world.agents[mid].alive]
        if len(members) < 2:
            return 0.0
        scores = []
        for agent in members:
            for other in members:
                if agent.id != other.id:
                    scores.append(agent.get_rel(other.id).bond_score())
        avg = sum(scores) / len(scores) if scores else 0.0
        shared_food = min(0.2, sum(group.stash.get(k, 0) for k in ("berries", "fish", "water")) / 30)
        return max(0.0, min(1.0, 0.45 + avg + shared_food))

    def _member_fit(self, world, group: SocialGroup, agent) -> float:
        others = [world.agents[mid] for mid in group.members if mid != agent.id and mid in world.agents]
        if not others:
            return 0.0
        return sum(agent.get_rel(other.id).bond_score() for other in others) / len(others)

    def _group_name(self, anchor: str, traits: dict) -> str:
        if traits.get("aggression", 0.5) > 0.7:
            prefix = "Guardians of"
        elif traits.get("empathy", 0.5) > 0.7:
            prefix = "Circle of"
        elif traits.get("curiosity", 0.5) > 0.7:
            prefix = "Seekers of"
        else:
            prefix = "Camp at"
        return f"{prefix} {anchor}"

    def _ethos(self, traits: dict) -> str:
        if traits.get("aggression", 0.5) > 0.7:
            return "security"
        if traits.get("empathy", 0.5) > 0.7:
            return "mutual aid"
        if traits.get("curiosity", 0.5) > 0.7:
            return "discovery"
        if traits.get("greed", 0.5) > 0.7:
            return "accumulation"
        return "survival"

    def _summarize_event(self, event: dict) -> str:
        event_type = event.get("type")
        if event_type == "speech":
            return f"{event.get('agent')} told {event.get('target')}: {event.get('phrase')}"
        if event_type == "trade":
            return f"{event.get('from')} traded with {event.get('to')}"
        if event_type == "give":
            return f"{event.get('from')} gave supplies to {event.get('to')}"
        if event_type == "confrontation":
            return f"{event.get('agent')} confronted {event.get('target')}"
        if event_type == "territory_claimed":
            return f"{event.get('agent')} claimed {event.get('resource')}"
        if event_type == "territory_seized":
            return f"{event.get('agent')} seized {event.get('resource')}"
        if event_type == "discovery":
            return f"{event.get('agent')} discovered {event.get('recipe')}"
        if event_type == "camp_founded":
            return f"{event.get('group')} formed near {event.get('anchor')}"
        if event_type == "project_completed":
            return f"{event.get('group')} finished {event.get('project')}"
        if event_type == "weather_shift":
            return f"weather changed to {event.get('weather')}"
        return event_type or "something happened"
