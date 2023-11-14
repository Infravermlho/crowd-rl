from crowd_rl.crowd_rl_v0 import Config, Agent, Coords, Queue, Attendant

queues = [
    Queue(
        order=0,
        accepts=[0],
        attendants=[
            Attendant(pos=Coords(x=2, y=1)),
            # Attendant(pos=Coords(x=4, y=1)),
            # Attendant(pos=Coords(x=6, y=1)),
        ],
        wait_spots=[Coords(x=6, y=4), Coords(x=6, y=5), Coords(x=6, y=6), Coords(x=6, y=7)],
    ),
]

agents = [
    Agent(type=0, pos=Coords(x=3, y=8)),
    # Agent(type=0, pos=Coords(x=1, y=8)),
    # Agent(type=0, pos=Coords(x=2, y=8)),
    # Agent(type=0, pos=Coords(x=4, y=9)),
    # Agent(type=0, pos=Coords(x=3, y=9)),
    # Agent(type=0, pos=Coords(x=5, y=1)),
    # Agent(type=0, pos=Coords(x=2, y=1)),
    # Agent(type=0, pos=Coords(x=1, y=3)),
]

worldmap = [
    [1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
]

env_config = Config(
    worldmap=worldmap,
    queues=queues,
    agents=agents,
)


# from crowd_rl.crowd_rl_v0 import Config, Agent, Coords, Target

# targets = [
#     Target(order=0, pos=Coords(x=3, y=7), accepts=[0]),
#     Target(order=0, pos=Coords(x=4, y=7), accepts=[0]),
#     Target(order=0, pos=Coords(x=6, y=9), accepts=[0]),
# ]

# agents = [
#     Agent(type=0, starting=Coords(x=3, y=1)),
#     Agent(type=0, starting=Coords(x=7, y=1)),
#     Agent(type=0, starting=Coords(x=7, y=3)),
# ]

# worldmap = [
#     [1, 0, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 0, 0, 1],
# ]

# env_config = Config(
#     worldmap=worldmap,
#     targets=targets,
#     agents=agents,
# )
