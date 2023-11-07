from crowd_rl.crowd_rl_v0 import Config, Agent, Coords, Target

targets = [
    Target(order=0, pos=Coords(x=3, y=7), accepts=[0]),
]

agents = [
    Agent(type=0, starting=Coords(x=3, y=1)),
]

worldmap = [
    [1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 1],
]

env_config = Config(
    worldmap=worldmap,
    targets=targets,
    agents=agents,
)
