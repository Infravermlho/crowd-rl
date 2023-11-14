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


class Agent(BaseModel):
    type: int
    pos: Coords

    proximity: int | None = None
    next_target: Optional[Coords] = None
    progress: int = 0
    stall: int = 0
    waiting: bool = False
    being_served: bool = False


class Attendant(BaseModel):
    pos: Coords
    busy: bool = False

    order: int | None = None
    queue: "Optional[Queue]" = None


class Queue(BaseModel):
    order: int
    accepts: List[int]
    attendants: List[Attendant]
    wait_spots: List[Coords]

    busy_attendants: int = 0
    members: Deque[Agent] = deque()

    @computed_field
    @property
    def full(self) -> bool:
        return self.busy >= len(self.wait_spots)

    @computed_field
    @property
    def busy(self) -> int:
        return len(self.members)


class Config(BaseModel):
    worldmap: list
    queues: List[Queue]
    agents: List[Agent]

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
