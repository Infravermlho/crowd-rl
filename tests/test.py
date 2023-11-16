"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

from crowd_rl import crowd_rl_v0 as crowd
from test_config_small import env_config
import numpy as np
import time

if __name__ == "__main__":
    env = crowd.env(config=env_config, render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            obs = observation["observation"]
            print(obs)
            action = env.action_space(agent).sample(mask)

        env.step(action)

env.close()
