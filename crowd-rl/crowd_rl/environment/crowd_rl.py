import os
from os import path as os_path
from copy import copy, deepcopy
import random
import functools

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle
from collections import deque


from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .config import Config, Queue, Agent, Coords


def get_image(path):
    # pylint: disable=no-member
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def fill(surface, color):
    w, h = surface.get_size()
    r, g, b, _ = color
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            surface.set_at((x, y), pygame.Color(r, g, b, a))


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "crowd_rl_v0",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        config: Config,
        render_mode: str | None = None,
        screen_scaling: int = 12,
        render_fps=10,
    ):
        EzPickle.__init__(self, config, render_mode, screen_scaling)
        super().__init__()
        self.config = config

        self.screen = None
        self.render_mode = render_mode
        self.tile_map = self._load_images() if render_mode else {}

        self.agents_obj = None
        self.map = np.array(self.config.worldmap, dtype=np.int8)
        self.height = self.config.height
        self.width = self.config.width

        entrance_agents = sum(list(len(x.agents) for x in self.config.entrances))
        self.agents = [
            f"agent_{i}" for i in range(len(self.config.agents) + entrance_agents)
        ]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {i: spaces.Discrete(5) for i in self.agents}

        self.max_progress = max(list(x.order for x in self.config.queues)) + 1
        self.max_targets = max(
            list(len(x.attendants) for x in self.config.queues)
            + [len(self.config.queues)]
        )
        self.unflattened_observation_spaces = {
            i: spaces.Dict(
                {
                    "own_position": gymnasium.spaces.Box(
                        low=-1024, high=1024, shape=(2,), dtype=np.float32
                    ),
                    "target_position": gymnasium.spaces.Box(
                        low=-1024,
                        high=1024,
                        shape=(self.max_targets, 2),
                        dtype=np.float32,
                    ),
                    "distance": spaces.Box(
                        low=0,
                        high=255,
                        shape=(4,),
                        dtype=np.float32,
                    ),
                }
            )
            for i in self.agents
        }

        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.flatten_space(
                        self.unflattened_observation_spaces[i]
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                }
            )
            for i in self.unflattened_observation_spaces
        }

        self.metadata["render_fps"] = render_fps * len(self.agents)

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        # pylint: disable=attribute-defined-outside-init
        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.queues = [[] for _ in range(self.max_progress)]
        for queue in deepcopy(self.config.queues):
            self.queues[queue.order].append(queue)

        self.attendants = []
        for queue_order in self.queues:
            for queue in queue_order:
                for attendant in queue.attendants:
                    attendant.queue = queue
                    attendant.order = queue.order
                    self.attendants.append(attendant)

        self.entrances = deepcopy(self.config.entrances)
        self.exits = deepcopy(self.config.exits)

        queue_agents_obj = []
        for entrance in self.entrances:
            entrance.cooldown = entrance.rate
            for agent in entrance.agents:
                agent.undeployed = True
                agent.pos.x, agent.pos.y = (-1, -1)
                queue_agents_obj.append(agent)
                entrance.agent_queue.append(agent)

        self.agents_obj = dict(
            zip(self.agents, deepcopy(self.config.agents) + queue_agents_obj)
        )

        self.agent_dist_map = {i: None for i in self.agents}

        self.last_rewards = {i: 0 for i in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}

        self.done = {i: False for i in self.agents}

        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}

        self.infos = {i: {} for i in self.agents}

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        self._agent_act(agent, action)

        if self._agent_selector.is_last():
            self._update_attendants()
            self._update_queues()
            self._allocate_rewards()
            self._update_exit()
            self._update_door()

            if all(self.done.values()):
                self.truncations = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        self._deads_step_first()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        agent_obj = self.agents_obj[agent]
        if self.done[agent] or agent_obj.undeployed:
            own_position_obs = np.full((2,), -1, "int16")
            target_postion = np.full((self.max_targets, 2), -1, "int16")
            distance_obs = np.zeros((4,), "int16")
            action_mask = np.array([1, 0, 0, 0, 0], "int8")
        else:
            own_position_obs = self._get_agent_pos(agent)
            distance_obs = self._get_agent_dist_obs(agent)
            target_postion = self._get_target_pos(agent)
            action_mask = self._get_action_mask(agent)

        return {
            "observation": spaces.flatten(
                self.unflattened_observation_spaces[agent],
                {
                    "own_position": own_position_obs,
                    "target_position": target_postion,
                    "distance": distance_obs,
                },
            ).astype("float32"),
            "action_mask": action_mask,
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        tile_size = 40

        screen_width = tile_size * self.config.width
        screen_height = tile_size * self.config.height

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("Crowd-RL Prototype")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.render_mode == "human":
            pygame.event.get()

        self.screen.fill([255, 255, 255])
        full_feature_map = copy(self.map)

        for queue_order in self.queues:
            for queue in queue_order:
                for spot in queue.wait_spots:
                    full_feature_map[spot.y][spot.x] = 4

                if not queue.full:
                    prio_wait_spot = queue.wait_spots[queue.busy]
                    full_feature_map[prio_wait_spot.y][prio_wait_spot.x] = 2

        for attendant in self.attendants:
            full_feature_map[attendant.pos.y][attendant.pos.x] = 3

        for entrance in self.entrances:
            full_feature_map[entrance.pos.y][entrance.pos.x] = 6

        for _exit in self.exits:
            full_feature_map[_exit.pos.y][_exit.pos.x] = 5

        for _, agent_char in self.agents_obj.items():
            if agent_char.pos.list != [-1, -1]:
                full_feature_map[agent_char.pos.y][agent_char.pos.x] = (
                    9 + agent_char.type
                )

        for index, tile in enumerate(full_feature_map.flat):
            x = index % self.width
            y = int(index / self.width)

            if tile != 0:
                self.screen.blit(
                    self.tile_map[tile],
                    ((x * tile_size, y * tile_size)),
                )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _get_agent_dist_map(self, agent_id):
        self.agent_dist_map[agent_id] = self._distance_map_merge_targets(
            self._get_target_pos(agent_id)
        )

        return self.agent_dist_map[agent_id]

    def _get_agent_dist_obs(self, agent_id):
        directions = {
            0: [-1, 0],
            1: [1, 0],
            2: [0, 1],
            3: [0, -1],
        }

        r = np.full((4,), 255, "int16")
        dist_map = self._get_agent_dist_map(agent_id)
        current_pos = self.agents_obj[agent_id].pos

        for key, di in directions.items():
            pos = [current_pos.x + di[0], current_pos.y + di[1]]
            if 0 <= pos[1] < self.height and 0 <= pos[0] < self.width:
                r[key] = dist_map[pos[1], pos[0]]

        return r

    def _get_agent_pos(self, agent_id: str):
        return self.agents_obj[agent_id].pos.list

    def _get_target_pos(self, agent_id: str):
        target_array = np.full((self.max_targets, 2), -1, "int16")
        agent = self.agents_obj[agent_id]

        if agent.progress >= self.max_progress:
            targets = [x.pos.list for x in self.exits if agent.type in x.accepts]
            target_array[: len(targets)] = np.array(targets, "int16")

        elif agent.next_attendant:
            targets = [x.pos.list for x in agent.next_attendant if not x.busy]
            target_array[: len(targets)] = np.array(targets, "int16")

        else:
            free_queues = np.array(
                list(
                    queue.wait_spots[queue.busy].list
                    for queue in self.queues[agent.progress]
                    if agent.type in queue.accepts and not queue.full
                ),
                "int8",
            )
            if free_queues.size > 0:
                target_array[: len(free_queues)] = np.array(free_queues, "int16")

        return target_array

    def _update_queues(self):
        for queue_order in self.queues:
            for queue in queue_order:
                changed = True
                while not queue.full and changed:
                    changed = False

                    prio_wait_spot = queue.wait_spots[queue.busy].list
                    for _, agent in self.agents_obj.items():
                        if (
                            agent.type in queue.accepts
                            and not agent.being_served
                            and agent.pos.list == prio_wait_spot
                        ):
                            queue.members.append(agent)
                            agent.waiting = True

                            changed = True
                            break

        for queue_order in self.queues:
            for queue in queue_order:
                if queue.free_spots > 0:
                    for q_pos, agent in enumerate(queue.members):
                        agent.pos.x, agent.pos.y = (
                            queue.wait_spots[q_pos].x,
                            queue.wait_spots[q_pos].y,
                        )
                    queue.free_spots -= 1

                if queue.busy:
                    if queue.busy_attendants < len(queue.attendants):
                        agent = queue.members.popleft()
                        agent.next_attendant = queue.attendants
                        agent.waiting = False
                        agent.being_served = True
                        queue.busy_attendants += 1
                        queue.free_spots += 1

    def _update_attendants(self):
        for attendant in self.attendants:
            if attendant.busy:
                attendant.cooldown -= 1
                if attendant.cooldown <= 0:
                    attendant.queue.busy_attendants -= 1
                    attendant.busy = False

                    key, agent = attendant.client
                    attendant.client = None

                    agent.waiting = False
                    agent.progress += 1

                    if agent.progress < self.max_progress:
                        while not [
                            i
                            for i in self.queues[agent.progress]
                            if agent.type in i.accepts
                        ]:
                            agent.progress += 1
                            if agent.progress >= self.max_progress:
                                break

            else:
                for key, agent in self.agents_obj.items():
                    if (
                        agent.being_served
                        and agent.pos == attendant.pos
                        and attendant.order == agent.progress
                    ):
                        attendant.busy = True
                        attendant.cooldown = attendant.rate
                        attendant.client = (key, agent)

                        agent.next_attendant = None
                        agent.being_served = False
                        agent.waiting = True
                        break

    def _update_door(self):
        for entrance in self.entrances:
            if len(entrance.agent_queue):
                if entrance.cooldown <= 0:
                    entrance.cooldown = entrance.rate
                    agent = entrance.agent_queue.popleft()
                    agent.undeployed = False
                    agent.pos.x, agent.pos.y = (entrance.pos.x, entrance.pos.y)
                else:
                    entrance.cooldown -= 1

    def _update_exit(self):
        for _exit in self.exits:
            for key, agent in self.agents_obj.items():
                if agent.pos.list == _exit.pos.list:
                    if agent.progress >= self.max_progress:
                        self.done[key] = True
                        agent.pos.x = -1
                        agent.pos.y = -1

    def _get_action_mask(self, agent_id: str):
        # [no_action, move_left, move_right, move_down, move_up]
        directions = {
            "1": [-1, 0],
            "2": [1, 0],
            "3": [0, 1],
            "4": [0, -1],
        }

        action_mask = np.ones(5, "int8")
        current_pos = self.agents_obj[agent_id].pos

        if self.agents_obj[agent_id].waiting:
            return np.array([1, 0, 0, 0, 0], "int8")

        for key, di in directions.items():
            mask_index = int(key)
            pos = [current_pos.x + di[0], current_pos.y + di[1]]

            if 0 <= pos[1] < self.height and 0 <= pos[0] < self.width:
                if self.map[pos[1], pos[0]] != 0:
                    action_mask[mask_index] = 0
            else:
                action_mask[mask_index] = 0

        return action_mask

    def _agent_act(self, agent_id: str, action: int):
        if action == 0:
            # no_move
            ...

        elif action == 1:
            # move_left
            current_pos = self.agents_obj[agent_id].pos
            if [current_pos.x - 1, current_pos.y] not in [
                i.pos.list for i in self.agents_obj.values()
            ]:
                current_pos.x -= 1

        elif action == 2:
            # move_left
            current_pos = self.agents_obj[agent_id].pos
            if [current_pos.x + 1, current_pos.y] not in [
                i.pos.list for i in self.agents_obj.values()
            ]:
                current_pos.x += 1

        elif action == 3:
            # move_down
            current_pos = self.agents_obj[agent_id].pos
            if [current_pos.x, current_pos.y + 1] not in [
                i.pos.list for i in self.agents_obj.values()
            ]:
                current_pos.y += 1

        elif action == 4:
            # move_up
            current_pos = self.agents_obj[agent_id].pos
            if [current_pos.x, current_pos.y - 1] not in [
                i.pos.list for i in self.agents_obj.values()
            ]:
                current_pos.y -= 1

    def _distance_map_merge_targets(self, targets):
        dist_map = np.full((self.config.height, self.config.width), 255, "int16")
        for x in targets:
            dist_map = np.minimum(dist_map, self._distance_map_target(tuple(x)))

        return dist_map

    def _distance_map_target(self, target):
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        q = deque()
        q.append(target)

        dist_map = np.full((self.config.height, self.config.width), 255, "int16")
        dist_map[target[1]][target[0]] = 0

        full_feature_map = copy(self.map)
        for agent in self.agents_obj.values():
            if agent.waiting:
                full_feature_map[agent.pos.y][agent.pos.x] = 1

        while q:
            node = q.popleft()
            dist = dist_map[node[1]][node[0]]

            for di in directions:
                next_node = [node[0] + di[0], node[1] + di[1]]

                if 0 <= next_node[1] < self.height and 0 <= next_node[0] < self.width:
                    if (
                        dist_map[next_node[1]][next_node[0]] == 255
                        and full_feature_map[next_node[1]][next_node[0]] == 0
                    ):
                        dist_map[next_node[1]][next_node[0]] = dist + 1
                        q.append(next_node)

        return dist_map

    def _allocate_rewards(self):
        self.rewards = {i: -1 for i in self.agents}
        for key, agent in self.agents_obj.items():
            if not self.done[key] and not agent.waiting and not agent.undeployed:
                dist_map = self.agent_dist_map[key]

                current_pos = agent.pos
                current_dist = dist_map[current_pos.y][current_pos.x]

                if agent.proximity is None:
                    agent.proximity = current_dist
                elif agent.proximity > current_dist:
                    self.rewards[key] = 1
                    agent.proximity = current_dist

    def _load_images(self):
        tile_size = 40
        random.seed(255)

        agent = get_image(os.path.join("img", "agent.png"))
        agent = pygame.transform.scale(agent, (tile_size, tile_size))

        agents = []
        for _ in range(10):
            single_agent = copy(agent)
            fill(
                single_agent,
                (
                    random.randrange(255),
                    random.randrange(255),
                    random.randrange(255),
                    0,
                ),
            )
            agents.append(single_agent)

        wall = get_image(os.path.join("img", "wall.png"))
        wall = pygame.transform.scale(wall, (tile_size, tile_size))

        target = get_image(os.path.join("img", "target.png"))
        target = pygame.transform.scale(target, (tile_size, tile_size))

        target_inactive = get_image(os.path.join("img", "target_inactive.png"))
        target_inactive = pygame.transform.scale(
            target_inactive, (tile_size, tile_size)
        )

        target_primary = get_image(os.path.join("img", "target_primary.png"))
        target_primary = pygame.transform.scale(target_primary, (tile_size, tile_size))

        entrance_sprite = get_image(os.path.join("img", "entrance.png"))
        entrance_sprite = pygame.transform.scale(
            entrance_sprite, (tile_size, tile_size)
        )

        exit_sprite = get_image(os.path.join("img", "exit.png"))
        exit_sprite = pygame.transform.scale(exit_sprite, (tile_size, tile_size))

        tiles = {
            1: wall,
            2: target,
            3: target_primary,
            4: target_inactive,
            5: exit_sprite,
            6: entrance_sprite,
        }

        for index, agent in enumerate(agents):
            tiles[9 + index] = agent

        return tiles
