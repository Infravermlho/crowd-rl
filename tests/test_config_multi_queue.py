from crowd_rl.crowd_rl_v0 import Config, Agent, Coords, Queue, Attendant, Entrance, Exit

queues = [
    Queue(
        order=0,
        accepts=[0],
        attendants=[
            Attendant(pos=Coords(x=1, y=1), rate=3),
        ],
        wait_spots=[
            Coords(x=1, y=4),
            Coords(x=1, y=5),
            Coords(x=1, y=6),
            Coords(x=1, y=7),
            Coords(x=1, y=8),
        ],
    ),
    Queue(
        order=1,
        accepts=[0],
        attendants=[
            Attendant(pos=Coords(x=4, y=11), rate=15),
            Attendant(pos=Coords(x=6, y=11), rate=15),
            Attendant(pos=Coords(x=8, y=11), rate=15),
        ],
        wait_spots=[
            Coords(x=7, y=8),
            Coords(x=7, y=7),
            Coords(x=7, y=6),
            Coords(x=7, y=5),
            Coords(x=6, y=5),
            Coords(x=5, y=5),
            Coords(x=4, y=5),
            Coords(x=3, y=5),
            Coords(x=3, y=4),
        ],
    ),
]

entrances = [
    Entrance(
        pos=Coords(x=0, y=11),
        rate=1,
        agents=[
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
            Agent(type=0),
        ],
    )
]

exits = [
    Exit(pos=Coords(x=14, y=0), accepts=[1]),
    Exit(pos=Coords(x=14, y=12), accepts=[0, 1]),
]

worldmap = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
]

env_config = Config(worldmap=worldmap, queues=queues, entrances=entrances, exits=exits)
