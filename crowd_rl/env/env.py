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
    env = CrowdEnvironment(render_mode=render_mode, args=args)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    return env


parallel_env = parallel_wrapper_fn(env)


class CrowdEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "crowd_v0",
        "render_fps": 60,
        "has_manual_policy": True,
    }

    def __init__(self, render_mode=None, args=None):
        """
        Aceita os argumentos
        - num_agents
        - location
        - Runtime
        """
        # INICIA A SOBRECLASSE da SUBCALSSE (AECEnv)
        super().__init__()

        self.num_agents = args.num_agents if hasattr(args, "num_agents") else 1
        self.location = args.location if hasattr(
            args, "location") else "default"

        # self.runtime = args.runtime if hasattr(args, "runtime") else 360
        self.possible_agents = ["player_" +
                                str(r) for r in range(self.num_agents)]
        self.agents = self.possible_agents[:]

        self.filename = f"{self.level}_agents{self.num_agents}"

        self.world = CrowdEnvironment()
        self.world.load_Location(
            location=self.location, num_agents=self.num_agents)

        self.timestep = 0

        obs_space = {
            'agent_location': gym.spaces.Box(
                low=0, high=max(self.world.width, self.world.height), shape=(2,)),
            'goal_vector': gym.spaces.MultiBinary(self.num_goals)
        }

        self._observation_spaces = dict(
            zip(self.possible_agents, [obs_space for _ in enumerate(self.possible_agents)]))

        self._action_spaces = dict(zip(self.possible_agents, [
                                   Discrete(4) for _ in enumerate(self.possible_agents)]))

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
