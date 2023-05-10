# TODO:
# self.world.width | self.world.height
# Initialize world on function: load_Location()

import functools

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def env(render_mode=None, args=None):
    env = raw_env(render_mode=render_mode, args=args)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    return env


# parallel_env = parallel_wrapper_fn(env)
"""
TODO: MUDAR PARA PARALLEL NO FUTURO A O INVEZ DE ->AEC<-
"""


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "crowd_v0",
        "render_fps": 60,
        "has_manual_policy": True,
    }

    def __init__(self, render_mode=None, args=None):
        """
        +--------------------------------------------------------------+
        |Aceita os argumentos                                          |
        |- num_agents                                                  |
        |- location                                                    |
        |                                                              |
        +--------------------------------------------------------------+
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents    [x]
        - action_spaces      [x]
        - observation_spaces [x]
        These attributes should not be changed after initialization.
        """
        # --------------------> ARGS <----------------------------
        num_agents = args.num_agents if hasattr(args, "num_agents") else 2
        
        self.location = args.location if hasattr(
            args, "location") else "default"
        # --------------------> ARGS <----------------------------
        # self.world = CrowdLocation()
        # self.filename = f"{self.level}_agents{self.num_agents}"
        # self.world.load_Location(
        # location=self.location, num_agents=self.num_agents)

        self.possible_agents = ["player_" +
                                str(r) for r in range(num_agents)]

        obs_space = {
            'agent_location': gym.spaces.Box(
                low=0, high=max(self.world.width, self.world.height), shape=(2,)),
            'goal_vector': gym.spaces.MultiBinary(self.num_goals)
        }

        self._observation_spaces = dict(
            zip(self.possible_agents, [obs_space for _ in enumerate(self.possible_agents)]))

        self._action_spaces = dict(zip(self.possible_agents, [
                                   Discrete(4) for _ in enumerate(self.possible_agents)]))

        self.render_mode = "human"

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        """
        +-----------------------------------------------------------------------------------------+
        |TODO: N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA |
        |TODO: N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA |
        |TODO: N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA N IMPLEMENTADO AINDA |
        +-----------------------------------------------------------------------------------------+
        """
        if self.render_mode is None:
            gym.logger.warn(
                "Render method called without specifying any render mode"
            )
            return

    def observe(self, agent):
        """
        +-------------------------------+
        | TODO: criar self.observations |
        +-------------------------------+
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        +---------------------------+
        | TODO: matar o pygame      |
        +---------------------------+
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        +------------------------------------+
        | TODO STATE? OBSERVATIONS? ???????? |
        +------------------------------------+
        Reset needs to initialize the following attributes
        - agents              [X]
        - rewards             [X]
        - _cumulative_rewards [X]
        - terminations        [X]
        - truncations         [X]
        - infos               [X]
        - agent_selection     [X]
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # self.state = {agent: None for agent in self.agents}
        # self.observations = {agent: None for agent in self.agents}
        self.timestep = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        +-------------------------------------------+
        |IMPLEMENTAR REWARD                         |
        +-------------------------------------------+
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards               [ ]
        - _cumulative_rewards   [ ]
        - terminations          [ ]
        - truncations           [ ]
        - infos                 [ ]
        - agent_selection       [ ]
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
