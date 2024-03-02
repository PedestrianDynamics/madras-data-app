"""Manages config data for Voronoi profile calculations"""

from dataclasses import dataclass
from typing import List

import pedpy


@dataclass
class Config:
    files: List[str]
    profile_data_file: str
    result_file: str
    speed_frame_rate: int
    fps: int
    walkable_area: pedpy.WalkableArea
    grid_size: float
    rmax: int
    vmax: int
    jmax: int
