"""Collection of functions used throughout the app."""

import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import pedpy
import requests  # type: ignore
import streamlit as st
from shapely import Polygon

from ..classes.datafactory import Direction, DirectionInfo


def is_running_locally() -> bool:
    """Check if the Streamlit app is running locally."""
    streamlit_server = "/mount/src/madras-data-app"
    current_working_directory = os.getcwd()
    return current_working_directory != streamlit_server


def download(url: str, destination: Union[str, Path]) -> None:
    """Download a file from a specified URL and saves it to a given destination."""
    # Send a GET request
    response = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kbyte
    progress_bar = st.progress(0)
    progress_status = st.empty()
    written = 0

    with open(destination, "wb") as f:
        for data in response.iter_content(block_size):
            written += len(data)
            f.write(data)
            # Update progress bar
            progress = int(100 * written / total_size)
            progress_bar.progress(progress)
            progress_status.text(f"> {progress}%")

    progress_status.text("Download complete.")
    progress_bar.empty()  # clear  the progress bar after completion


def get_color_by_name(direction_name: str) -> str:
    """Get color from directions_infos."""
    for direction_info in st.session_state.direction_infos:
        if direction_info.name == direction_name:
            return str(direction_info.color)
    return "Color not found!"


def get_id_by_name(direction_name: str) -> int:
    """Get id from directions_infos."""
    for direction_info in st.session_state.direction_infos:
        if direction_info.name == direction_name:
            return int(direction_info.id)
    return -1


def get_measurement_lines(
    trajectory_data: pd.DataFrame, distance_to_bounding: float
) -> List[Direction]:
    """Create 4 measurement lines inside the walkable_area."""
    min_x = trajectory_data.data["x"].min() + distance_to_bounding
    max_x = trajectory_data.data["x"].max() - distance_to_bounding
    min_y = trajectory_data.data["y"].min() + distance_to_bounding
    max_y = trajectory_data.data["y"].max() - distance_to_bounding

    return [
        Direction(
            info=DirectionInfo(
                id=get_id_by_name("Right"),
                name="Right",
                color=get_color_by_name("Right"),
            ),
            line=pedpy.MeasurementLine([[min_x, min_y], [min_x, max_y]]),
        ),
        Direction(
            info=DirectionInfo(
                id=get_id_by_name("Left"), name="Left", color=get_color_by_name("Left")
            ),
            line=pedpy.MeasurementLine([[max_x, min_y], [max_x, max_y]]),
        ),
        Direction(
            info=DirectionInfo(
                id=get_id_by_name("Top"),
                name="Top",
                color=get_color_by_name("Top"),
            ),
            line=pedpy.MeasurementLine([[min_x, max_y], [max_x, max_y]]),
        ),
        Direction(
            info=DirectionInfo(
                id=get_id_by_name("Bottom"),
                name="Bottom",
                color=get_color_by_name("Bottom"),
            ),
            line=pedpy.MeasurementLine([[max_x, min_y], [min_x, min_y]]),
        ),
    ]


def setup_walkable_area(trajectory_data: pd.DataFrame) -> pedpy.WalkableArea:
    """Create walkable_area from trajectories."""
    min_x = trajectory_data.data["x"].min()
    max_x = trajectory_data.data["x"].max()
    min_y = trajectory_data.data["y"].min()
    max_y = trajectory_data.data["y"].max()
    rectangle_coords = [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
    rectangle_polygon = Polygon(rectangle_coords)
    return pedpy.WalkableArea(rectangle_polygon)


def setup_measurement_area(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> pedpy.MeasurementArea:
    """Create measurement_area from trajectories."""
    rectangle_coords = [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
    rectangle_polygon = Polygon(rectangle_coords)
    return pedpy.MeasurementArea(rectangle_polygon)
