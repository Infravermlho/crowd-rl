"""
Config file that will be passed to world.py
"""

from typing import Optional, Literal, List, Deque
from pydantic import BaseModel, computed_field
from collections import deque


class Coords(BaseModel):
    x: int
    y: int

    @computed_field
    @property
    def list(self) -> list[int]:
        return [self.x, self.y]


class Exit(BaseModel):
    pos: Coords
    accepts: List[int]


class Attendant(BaseModel):
    pos: Coords
    rate: int = 0

    client: "Optional[Agent]" = None
    busy: bool = False
    cooldown: int = 0
    order: int | None = None
    queue: "Optional[Queue]" = None


class Agent(BaseModel):
    type: int
    pos: Coords = Coords(x=-1, y=-1)

    proximity: int | None = None
    next_attendant: Optional[List[Attendant]] = None
    progress: int = 0
    undeployed: bool = False
    waiting: bool = False
    being_served: bool = False


class Entrance(BaseModel):
    pos: Coords
    rate: int = 0
    agents: list[Agent]

    cooldown: int = 0
    agent_queue: Deque[Agent] = deque()


class Queue(BaseModel):
    order: int
    accepts: List[int]
    attendants: List[Attendant]
    wait_spots: List[Coords]

    free_spots: int = 0
    busy_attendants: int = 0
    members: Deque[Agent] = deque()

    @computed_field
    @property
    def full(self) -> bool:
        return self.busy >= len(self.wait_spots)

    @computed_field
    @property
    def busy(self) -> int:
        return len(self.members) + self.free_spots


class Config(BaseModel):
    worldmap: list
    queues: List[Queue]
    exits: list[Exit]
    entrances: list[Entrance] = []
    agents: List[Agent] = []

    seed: Optional[int] = None

    @computed_field
    @property
    def height(self) -> int:
        return len(self.worldmap)

    @computed_field
    @property
    def width(self) -> int:
        return len(self.worldmap[0])


# """
# Config file that will be passed to world.py
# """

# from typing import Optional, Literal
# from pydantic import BaseModel, computed_field


# class Config(BaseModel):
#     """
#     Pydantic class that will recieve args to be passed to the
#     World class
#     """

#     worldmap: Optional[list]
#     on_truncation: Literal["finish", "restart"] = "finish"
#     num_agents: int = 1
#     seed: Optional[int] = None
#     agents_starting_xy: list
#     targets_xy: Optional[list] = None

#     @computed_field
#     @property
#     def height(self) -> int:
#         return len(self.worldmap)

#     @computed_field
#     @property
#     def width(self) -> int:
#         return len(self.worldmap[0])

# -----------------------------------------------
