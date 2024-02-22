"""Collection of functions used throughout the app."""

import os
from pathlib import Path
from typing import Union

import pandas as pd
import pedpy
import requests
import streamlit as st
from shapely import Polygon


def is_running_locally() -> bool:
    """Check if the Streamlit app is running locally."""

    streamlit_server = "/mount/src/madras-data-app"
    current_working_directory = os.getcwd()
    return current_working_directory != streamlit_server


def download(url: str, destination: Union[str, Path]) -> None:
    """
    Downloads a file from a specified URL and saves it to a given destination.
    """
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


def get_measurement_lines(trajectory_data: pd.DataFrame):
    """Create 4 measurement lines inside the walkable_area"""
    eps = 1.0
    min_x = trajectory_data.data["x"].min() + eps
    max_x = trajectory_data.data["x"].max() - eps
    min_y = trajectory_data.data["y"].min() + eps
    max_y = trajectory_data.data["y"].max() - eps
    measurement_lines = [
        pedpy.MeasurementLine([[min_x, min_y], [min_x, max_y]]),  # left
        pedpy.MeasurementLine([[min_x, max_y], [max_x, max_y]]),  # top
        pedpy.MeasurementLine([[max_x, max_y], [max_x, min_y]]),  # right
        pedpy.MeasurementLine([[max_x, min_y], [min_x, min_y]]),  # buttom
    ]

    return measurement_lines


def setup_walkable_area(trajectory_data):
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
    walkable_area = pedpy.WalkableArea(rectangle_polygon)

    return walkable_area
