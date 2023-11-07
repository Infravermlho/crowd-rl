"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

from crowd_rl import crowd_rl_v0 as crowd
from test_config import env_config
import time

if __name__ == "__main__":
    env = crowd.env(config=env_config, render_mode="human")
    env.reset(seed=42)

    print(env.agents)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            action = env.action_space(agent).sample(mask)

        env.step(action)

env.close()
