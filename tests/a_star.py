"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

from crowd_rl import crowd_rl_v0 as crowd
from test_config_mall import env_config
import random
import numpy as np
import numpy.ma as ma

if __name__ == "__main__":
    env = crowd.env(config=env_config, render_mode="human", render_fps=8)
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            obs = observation["observation"]

            inv_mask = 1 - mask

            moves = np.insert(obs[:4], 0, 255, axis=0)
            masked_moves = ma.array(moves, mask=inv_mask)
            action = np.where(masked_moves == masked_moves.min())[0]
            action = action[random.randint(0, len(action) - 1)]

        env.step(action)

env.close()
