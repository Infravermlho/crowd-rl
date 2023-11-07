"""
Config file that will be passed to world.py
"""

from typing import Optional, Literal, List
from pydantic import BaseModel, computed_field


class Coords(BaseModel):
    x: int
    y: int


class Target(BaseModel):
    order: int
    pos: Coords
    accepts: List[int]


class Agent(BaseModel):
    type: int
    starting: Coords


class Config(BaseModel):
    on_truncation: Literal["finish", "restart"] = "finish"
    seed: Optional[int] = None
    worldmap: list
    targets: List[Target]
    agents: List[Agent]

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
