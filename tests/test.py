"""
File for testing the env
Run pip install . on root folder before running
"""

from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.crowd_rl_v0 import Config

if __name__ == "__main__":
    env_config = Config(
        worldmap=[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        num_agents=4,
        agents_starting_xy=[[0, 0], [0, 1], [4, 3], [3, 2]],
        targets_xy=[[2, 9], [0, 9], [4, 9], [3, 9]],
    )

    env = crowd.env(config=env_config, render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            action = env.action_space(agent).sample(mask)

        env.step(action)

env.close()
