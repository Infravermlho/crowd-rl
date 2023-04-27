import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10
import dumbtest

env = dumbtest.env()
env.reset()

clock = pygame.time.Clock()
manual_policy = knights_archers_zombies_v10.ManualPolicy(env)

for agent in env.agent_iter():
    clock.tick(env.metadata["render_fps"])

    observation, reward, termination, truncation, info = env.last()
    if agent == manual_policy.agent:
        action = manual_policy(observation, agent)
    else:
        action = env.action_space(agent).sample()

    env.step(action)