import os
from os import path as os_path
from copy import copy

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

from .config import Config


def get_image(path):
    # pylint: disable=no-member
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


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
        "render_fps": 2,
    }

    def __init__(
        self, config: Config, render_mode: str | None = None, screen_scaling: int = 12
    ):
        EzPickle.__init__(self, config, render_mode, screen_scaling)
        super().__init__()
        self.config = config

        self.screen = None
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling
        self.config = config

        self.agents_pos = None
        self.agents_progress = None
        self.map = None
        self.height = None
        self.width = None

        self.agents = [f"agent_{i}" for i in range(len(self.config.agents))]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        # [no_action, move_left, move_right, move_down, move_up]`
        self.action_spaces = {i: spaces.Discrete(5) for i in self.agents}

        self.unflattened_observation_spaces = {
            i: spaces.Dict(
                {
                    "own_position": gymnasium.spaces.Box(
                        low=-1024, high=1024, shape=(2,), dtype=np.float32
                    ),
                    "target_position": gymnasium.spaces.Box(
                        low=-1024, high=1024, shape=(2,), dtype=np.float32
                    ),
                    "obstacles": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.config.width, self.config.height),
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

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        # pylint: disable=attribute-defined-outside-init
        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.map = np.array(self.config.worldmap)
        self.height = self.config.height
        self.width = self.config.width

        self.targets = {}
        for target in self.config.targets:
            if hasattr(self.targets, str(target.order)):
                self.targets[str(target.order)].append(target)
            else:
                self.targets[str(target.order)] = [target]

        self.agents_pos = {
            i: [u.starting.x, u.starting.y]
            for i, u in zip(self.agents, self.config.agents)
        }
        self.agents_progress = {i: 0 for i in self.agents}

        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.done = {i: False for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.infos["agent_mask"] = {i: True for i in self.agents}
        self.infos["env_defined_actions"] = {i: None for i in self.agents}

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        self._agent_act(agent, action)

        if self._was_on_target(agent) and not self.done[agent]:
            self.rewards[agent] = 1
            self.infos["agent_mask"][agent] = False
            self.done[agent] = True

        if self._agent_selector.is_last():
            if all(self.done.values()):
                self.truncations = dict(zip(self.agents, [True for _ in self.agents]))

        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()
        self._deads_step_first()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        if self.done[agent]:
            own_postion_obs = np.zeros((2,), "int8")
            target_postion = np.zeros((2,), "int8")
            obstacles_obs = np.zeros((self.config.width, self.config.height), "int8")
            action_mask = np.zeros((5,), "int8")
        else:
            own_postion_obs = self._get_agent_pos(agent)
            obstacles_obs = self._get_obstacles(agent)
            target_postion = self._get_target_pos(agent)
            action_mask = self._get_action_mask(agent)

        return {
            "observation": spaces.flatten(
                self.unflattened_observation_spaces[agent],
                {
                    "own_position": own_postion_obs,
                    "target_position": target_postion,
                    "obstacles": obstacles_obs,
                },
            ).astype("float32"),
            "action_mask": action_mask,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

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

        # Loading and scaling images
        agent = get_image(os.path.join("img", "agent.png"))
        agent = pygame.transform.scale(agent, (tile_size, tile_size))

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

        self.screen.fill([255, 255, 255])

        full_feature_map = copy(self.map)
        for steps in self.targets.values():
            for target_obj in steps:
                full_feature_map[target_obj.pos.y][target_obj.pos.x] = 2
        for _, agent_pos in self.agents_pos.items():
            full_feature_map[agent_pos[1]][agent_pos[0]] = 9

        for index, tile in enumerate(full_feature_map.flat):
            x = index % self.width
            y = int(index / self.width)

            # pygame.draw.rect(
            #     self.screen,
            #     [0, 0, 0],
            #     pygame.Rect((x * tile_size, y * tile_size), (tile_size, tile_size)),
            #     1,
            # )

            if tile == 1:
                self.screen.blit(
                    wall,
                    ((x * tile_size, y * tile_size)),
                )
            elif tile == 2:
                self.screen.blit(
                    target,
                    ((x * tile_size, y * tile_size)),
                )
            elif tile == 3:
                self.screen.blit(
                    target_primary,
                    ((x * tile_size, y * tile_size)),
                )
            elif tile == 4:
                self.screen.blit(
                    target_inactive,
                    ((x * tile_size, y * tile_size)),
                )
            elif tile == 9:
                self.screen.blit(
                    agent,
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

    def _get_agent_pos(self, agent_id: str):
        return self.agents_pos[agent_id]

    def _was_on_target(self, agent_id: str):
        return self._get_agent_pos(agent_id) == self._get_target_pos(agent_id)

    def _get_target_pos(self, agent_id: str):
        # TODO: TEMPORARY
        return [self.targets["0"][0].pos.x, self.targets["0"][0].pos.y]

    # def _get_other_agents(self, agent_id: str):
    #
    #     other_agents_pos = np.zeros((self.width, self.height), "int8")
    #     for agent_index, agent_pos in self.agents_pos.items():
    #         if agent_index is not agent_id:
    #             other_agents_pos[agent_pos[0], agent_pos[1]] = 1
    #
    #     return other_agents_pos

    def _get_obstacles(self, agent_id: str):
        return self.map

    def _get_action_mask(self, agent_id: str):
        # [no_action, move_left, move_right, move_down, move_up]`

        action_mask = np.ones(5, "int8")
        current_pos_x, current_pos_y = (
            self.agents_pos[agent_id][0],
            self.agents_pos[agent_id][1],
        )

        colision_map = copy(self.map)
        if current_pos_x - 1 < 0 or colision_map[current_pos_y][current_pos_x - 1] != 0:
            action_mask[1] = 0

        if (
            current_pos_x + 1 >= self.width
            or colision_map[current_pos_y][current_pos_x + 1] != 0
        ):
            action_mask[2] = 0

        if (
            current_pos_y + 1 >= self.height
            or colision_map[current_pos_y + 1][current_pos_x] != 0
        ):
            action_mask[3] = 0

        if current_pos_y - 1 < 0 or colision_map[current_pos_y - 1][current_pos_x] != 0:
            action_mask[4] = 0

        return action_mask

    def _agent_act(self, agent_id: str, action: int):
        if action == 0:
            ...
            # if not self.done[agent_id]:
                # self.rewards[agent_id] = -1

        elif action == 1:
            # move_left
            old_pos = self.agents_pos[agent_id]
            assert self.map[old_pos[1]][old_pos[0] - 1] == 0

            new_pos = [old_pos[0] - 1, old_pos[1]]
            if new_pos not in self.agents_pos.values():
                self.agents_pos[agent_id] = new_pos

        elif action == 2:
            # move_right
            old_pos = self.agents_pos[agent_id]
            assert self.map[old_pos[1]][old_pos[0] + 1] == 0

            new_pos = [old_pos[0] + 1, old_pos[1]]
            if new_pos not in self.agents_pos.values():
                self.agents_pos[agent_id] = new_pos

        elif action == 3:
            # move_down
            old_pos = self.agents_pos[agent_id]
            assert self.map[old_pos[1] + 1][old_pos[0]] == 0

            new_pos = [old_pos[0], old_pos[1] + 1]
            if new_pos not in self.agents_pos.values():
                self.agents_pos[agent_id] = new_pos

        elif action == 4:
            # move_up
            old_pos = self.agents_pos[agent_id]
            assert self.map[old_pos[1] - 1][old_pos[0]] == 0

            new_pos = [old_pos[0], old_pos[1] - 1]
            if new_pos not in self.agents_pos.values():
                self.agents_pos[agent_id] = new_pos

    # def _flood_rewards(self, agent_id):
    #     moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    #     # TODO: temporary!
    #     target_pos = (self.targets[0][0].pos.x, self.targets[0][0].pos.y)

    #     vis_map = np.zeros(self.width, self.height)
    #     dist_map = np.zeros(self.width, self.height)

    #     q = deque()
    #     q.append(target_pos)
    #     vis_map[target_pos[0]][target_pos[1]] = True
    #     dist_map[target_pos[0]][target_pos[1]] = True
    #     while len(q):
    #         x, y = q.popleft()
    #         for dx, dy in moves:
    #             new_x = x + dx
    #             new_y = y + dy
    #             if (
    #                 self.width > new_x >= 0
    #                 and self.height > new_y >= 0
    #                 and not vis_map[new_x][new_y]
    #                 and self.map[new_x][new_y] == 0
    #             ):
    #                 q.append((new_x, new_y))
    #                 vis_map[new_x][new_y] = True
    #                 dist_map[new_x][new_y] = dist_map[x][y] + 1
    #         if not vis_map[self.width - 1][self.height - 1]:
    #             print(-1)
    #         else:
    #             print(dist_map[self.width - 1][self.height - 1])
