from crowd_rl.crowd_rl_v0 import Config, Agent, Coords, Queue, Attendant, Entrance, Exit

queues = [
    Queue(
        order=0,
        accepts=[0],
        attendants=[
            Attendant(pos=Coords(x=2, y=1), rate=15),
        ],
        wait_spots=[
            Coords(x=4, y=3),
            Coords(x=4, y=4),
            Coords(x=4, y=5),
        ],
    ),
]

entrances = [
    Entrance(
        pos=Coords(x=1, y=6),
        rate=1,
        agents=[
            Agent(type=0),
        ],
    )
]

exits = [
    Exit(pos=Coords(x=5, y=0), accepts=[0]),
]

worldmap = [
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1],
]

env_config = Config(worldmap=worldmap, queues=queues, entrances=entrances, exits=exits)
