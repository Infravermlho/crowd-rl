# TODO: 
# self.world.width | self.world.height
# Initialize world on function: load_Location()

import os

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium import spaces
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from crowd import Crowd

from pettingzoo import AECEnv


def env(render_mode=None, args=None):
    env = CrowdEnvironment(render_mod=render_mode, args=None)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class CrowdEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "text", "text_full"],
        "name": "crowd_v0",
        "is_parallelizable": True,
        "render_fps": 60,
        "has_manual_policy": True,
    }

    def __init__(self, render_mode=None, args=None):
        # INICIA A SOBRECLASSE da SUBCALSSE (AECEnv)
        super().__init__()

        self.num_agents = args.num_agents if hasattr(args, "num_agents") else 1
        self.location = args.location if hasattr(
            args, "location") else "default"
        self.runtime = args.runtime if hasattr(args, "runtime") else 360

        self.possible_agents = ["player_" +
                                str(r) for r in range(self.num_agents)]

        self.filename = f"{self.level}_agents{self.num_agents}"
        self.termination_info = ""

        self.world = CrowdEnvironment()
        self.timestep = 0

        self.world.load_Location(
            location=self.location, num_agents=self.num_agents)
        
        
        numeric_obs_space = {'symbolic_observation': gym.spaces.Box(low=0, high=10,
                                                            shape=(self.world.width, self.world.height,
                                                                   self.graph_representation_length), dtype=np.int32),

                             'agent_location': gym.spaces.Box(low=0, high=max(self.world.width, self.world.height),
                                                              shape=(2,)),

                             'goal_vector': gym.spaces.MultiBinary(NUM_GOALS)}
        
        self.observation_spaces = {agent: gym.spaces.Dict(numeric_obs_space) for agent in self.possible_agents}
        self.action_spaces = {agent: gym.spaces.Discrete(4) for agent in self.possible_agents}

        # Didn't UNDERSTAND YET!!!!
        # NO IDEA WHAT MAPPING IS
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))
        # Didn't UNDERSTAND YET!!!!

        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []

    def reset(self):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]