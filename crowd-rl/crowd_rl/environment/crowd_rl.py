import os
from os import path as os_path
from copy import copy

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gymnasium.wrappers import FlattenObservation

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

        self.agents = [f"player_{i}" for i in range(self.config.num_agents)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        # Handles agent termination
        self.kill_list = []
        self.dead_agents = []

        # [no_action, move_left, move_right, move_down, move_up]`
        self.action_spaces = {i: spaces.Discrete(5) for i in self.agents}

        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "obstacles": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.config.width, self.config.height),
                        dtype=np.int8,
                    ),
                    "agents": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.config.width, self.config.height),
                        dtype=np.int8,
                    ),
                    "own_position": gymnasium.spaces.Box(
                        low=-1024, high=1024, shape=(2,), dtype=int
                    ),
                    "target_position": gymnasium.spaces.Box(
                        low=-1024, high=1024, shape=(2,), dtype=int
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
                }
            )
            for i in self.agents
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

        self.agents_pos = dict(zip(self.agents, self.config.agents_starting_xy))
        self.targets_xy = dict(zip(self.agents, self.config.targets_xy))
        self.agents_progress = {i: 0 for i in self.agents}

        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
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

        if self._was_on_target(agent):
            self.rewards[agent] += 1
            self.kill_list.append(agent)

        # manage the kill list
        if self._agent_selector.is_last():
            # start iterating on only the living agents
            _live_agents = self.agents[:]
            for k in self.kill_list:
                # kill the agent
                _live_agents.remove(k)
                # set the termination for this agent for one round
                self.terminations[k] = True
                # add that we know this guy is dead
                self.dead_agents.append(k)

            # reset the kill list
            self.kill_list = []

            # reinit the agent selector with existing agents
            self._agent_selector.reinit(_live_agents)

        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()
        self._deads_step_first()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        obstacles_obs = self._get_obstacles(agent)
        agents_obs = self._get_other_agents(agent)
        own_postion_obs = self._get_other_agents(agent)
        target_postion = self._get_target_pos(agent)
        action_mask = self._get_action_mask(agent)

        # print(f"Obstacles for {agent}: {obstacles_obs}")
        # print(f"Agents_obs for {agent}: {agents_obs}")
        # print(f"own_position_obs for {agent}: {own_postion_obs}")
        # print(f"target_position for {agent}: {target_postion}")
        # print(f"Action mask for {agent}: {action_mask}")
        return {
            "obstacles": obstacles_obs,
            "agents": agents_obs,
            "own_position": own_postion_obs,
            "target_postion": target_postion,
            "action_mask": action_mask,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        # TODO: Implement screen scaling

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
        target_inactive = pygame.transform.scale(target_inactive, (tile_size, tile_size))

        target_primary = get_image(os.path.join("img", "target_primary.png"))
        target_primary = pygame.transform.scale(target_primary, (tile_size, tile_size))

        self.screen.fill([255, 255, 255])

        full_feature_map = copy(self.map)
        for _, target_pos in self.targets_xy.items():
            full_feature_map[target_pos[0][1]][target_pos[0][0]] = 2
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

    def _get_other_agents(self, agent_id: str):
        # TODO: Manter a lista permanentemente ou gerar na hora?

        other_agents_pos = np.zeros((self.width, self.height), "int8")
        for agent_index, agent_pos in self.agents_pos.items():
            if agent_index is not agent_id:
                other_agents_pos[agent_pos[0], agent_pos[1]] = 1

        return other_agents_pos

    def _get_obstacles(self, agent_id: str):
        # TODO: Different maps to different kinds of agents
        return self.map

    def _get_target_pos(self, agent_id: str):
        return self.targets_xy[agent_id][self.agents_progress[agent_id]]

    def _get_action_mask(self, agent_id: str):
        # [no_action, move_left, move_right, move_down, move_up]`

        action_mask = np.ones(5, "int8")
        current_pos_x, current_pos_y = self.agents_pos[agent_id]

        colision_map = copy(self.map)
        for _, agent_pos in self.agents_pos.items():
            colision_map[agent_pos[1]][agent_pos[0]] = 9

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
            # no_action
            ...
        elif action == 1:
            # move_left
            current_pos = self.agents_pos[agent_id]
            assert self.map[current_pos[1]][current_pos[0] - 1] == 0
            self.agents_pos[agent_id] = [current_pos[0] - 1, current_pos[1]]

        elif action == 2:
            # move_right
            current_pos = self.agents_pos[agent_id]

            assert self.map[current_pos[1]][current_pos[0] + 1] == 0
            self.agents_pos[agent_id] = [current_pos[0] + 1, current_pos[1]]

        elif action == 3:
            # move_down
            current_pos = self.agents_pos[agent_id]
            assert self.map[current_pos[1] + 1][current_pos[0]] == 0
            self.agents_pos[agent_id] = [current_pos[0], current_pos[1] + 1]

        elif action == 4:
            # move_up
            current_pos = self.agents_pos[agent_id]
            assert self.map[current_pos[1] - 1][current_pos[0]] == 0
            self.agents_pos[agent_id] = [current_pos[0], current_pos[1] - 1]
