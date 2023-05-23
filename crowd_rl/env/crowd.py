import functools

import numpy as np
import gymnasium as gym

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "crowd_rl"}
    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """
        self.place_height = 200
        self.place_width = 200

        self.possible_agents = ["player_" + str(r) for r in range(2)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {agent: gym.spaces.Discrete(4) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(self.place_height, self.place_width)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(self.place_height, self.place_width)),
            xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
        ) for agent in self.possible_agents
        }

        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(self.place_height, self.place_width)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(self.place_height, self.place_width)),
            xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.timestep = 0

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_pos = {agent: (0, 0) for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, actions):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        


    def render(self):
        pass
