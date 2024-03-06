"""Unsorted datastructure for the app."""

import glob
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import pedpy
import requests  # type: ignore
import streamlit as st


@dataclass
class DirectionInfo:
    """Measurement line metadata."""

    id: int
    name: str
    color: str


@dataclass
class Direction:
    """Data for measurement line."""

    info: DirectionInfo
    line: pedpy.MeasurementLine


@dataclass
class DataConfig:
    """Datastructure for the app."""

    directory: Path
    files: List[str] = field(default_factory=list)
    # data: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    url: str = "https://go.fzj.de/madras-data"

    def __post_init__(self) -> None:
        """Initialize the DataConfig instance by retrieving files for each country."""
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.retrieve_files()

    def retrieve_files(self) -> None:
        """Retrieve the files for each country specified in the countries list."""
        if not self.directory.exists():
            st.warning(f"{self.directory} does not exist yet!")
            with st.status("Downloading ...", expanded=True):
                download_and_unzip_files(self.url, "data.zip", self.directory)

        self.files = glob.glob(f"{self.directory}/*.txt")


def increment_frame_start(page_size: int) -> None:
    """Increment animation starting frame."""
    st.session_state.start_frame += page_size


def decrement_frame_start(page_size: int) -> None:
    """Decrease animation starting frame."""
    st.session_state.start_frame -= page_size


def reset_frame_start(start: int) -> None:
    """Reset animation starting frame to min(frames)."""
    st.session_state.start_frame = start


def init_state_bg_image() -> None:
    """Init state of background image."""
    if "bg_img" not in st.session_state:
        st.session_state.bg_img = None

    if "scale" not in st.session_state:
        st.session_state.scale = 0.5

    if "dpi" not in st.session_state:
        st.session_state.dpi = 100

    if "img_height" not in st.session_state:
        st.session_state.img_height = 100

    if "img_width" not in st.session_state:
        st.session_state.img_width = 100


def init_session_state() -> None:
    """Init session_state throughout the app."""
    path = Path(__file__)
    trajectories_directory = path.parent.parent.parent.absolute() / "data" / "trajectories"
    logging.info(f"{trajectories_directory = }")
    init_state_bg_image()
    # Initialize a list of DirectionInfo objects using the provided dictionaries
    if "direction_infos" not in st.session_state:
        st.session_state.direction_infos = [
            DirectionInfo(id=1, name="North", color="blue"),
            DirectionInfo(id=2, name="South", color="red"),
            DirectionInfo(id=3, name="East", color="green"),
            DirectionInfo(id=4, name="West", color="gray"),
        ]

    if "start_frame" not in st.session_state:
        st.session_state.start_frame = 0

    if not hasattr(st.session_state, "files"):
        st.session_state.files = []

    if not hasattr(st.session_state, "selected_file"):
        st.session_state.selected_file = ""

    if not hasattr(st.session_state, "file_changes"):
        st.session_state.file_changed = ""

    if not hasattr(st.session_state, "trajectory_data"):
        st.session_state.trajectoryData = pedpy.TrajectoryData

    dataconfig = DataConfig(trajectories_directory)
    st.session_state.files = dataconfig.files


def unzip_files(zip_path: Union[str, Path], destination: Union[str, Path]) -> None:
    """
    Unzip a ZIP file directly into the specified destination directory.

    Ignoring the original directory structure in the ZIP file.

    Parameters:
    - zip_path (str): The path to the ZIP file.
    - destination (str): The directory where files should be extracted.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            # Extract only if file (ignores directories)
            if not member.is_dir():
                # Build target filename path
                target_path = os.path.join(destination, os.path.basename(member.filename))
                # Ensure target directory exists (e.g., if not extracting directories)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # Extract file
                with zip_ref.open(member, "r") as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)


def download_and_unzip_files(url: str, destination: Union[str, Path], unzip_destination: Union[str, Path]) -> None:
    """
    Download a ZIP file from a specified URL.

    Saves it to a given destination, and unzips it into a specified directory.
    Displays the download and unzipping progress in a Streamlit app.
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

    progress_status.text("Download complete. Unzipping...")
    unzip_files(destination, unzip_destination)

    progress_status.text("Unzipping complete.")
    progress_bar.empty()  # Clear the progress bar after completion


def load_file(filename: str) -> pedpy.TrajectoryData:
    """Load and processes a file to create a TrajectoryData object.

    This function reads a space-separated values file into a pedpy-trajectoryData
    fps = 30 and unit=meters

    Parameters:
    - filename (str): The path to the file to be loaded.

    Returns:
    - An instance of TrajectoryData containing the processed data and frame rate.
    """
    return pedpy.load_trajectory(
        trajectory_file=Path(filename),
        default_frame_rate=30,
        default_unit=pedpy.TrajectoryUnit.METER,
    )
