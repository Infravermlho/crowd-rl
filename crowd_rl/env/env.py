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

        self.env = Crowd()
        # INICIALMENTE VAI SER RANDOM, MAS DEFINIR POR ARG DEPOIS
        self.agents = 5

        # ESTRUTURA DE AÇÕES LEGITMAS
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}

        # ESTRUTURA DE VALORES LEGITMO PARA LEITURA
        # +---------------------------------------+
        # The observation for the mountain car environment 
        # is a vector of two numbers representing velocity 
        # and position. The middle point between the two 
        # mountains is taken to be the origin, with right 
        # being the positive direction and left being the 
        # negative direction.
        self.observation_space = spaces.Box(2,)
