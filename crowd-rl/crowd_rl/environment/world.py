import numpy as np
from .config import Config

class World:
    def __init__(self, config: Config):
        self.config = config
        self.rnd = np.random.default_rng(config.seed)

        self.height = self.config.height
        self.width = self.config.width

        self.map = np.array(self.config.map)
        assert self.map.shape == (self.width, self.height)

        self.agents_pos = self.config.agents_starting_xy

        
        self.agents = self.possible_agents[:]

        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

    def get_agent_pos(self, agent_id: int):
        ...

    def get_other_agents(self, agent_id: int):
        ...
    
    def get_obstacles(self, agent_id: int):
        ...

    def get_target_pos(self, agent_id: int):
        ...
    
    def agent_act(self, action: int):
        ...
    