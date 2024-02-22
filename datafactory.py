import glob
import os
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
import pedpy

import requests
import streamlit as st

import datafactory


@dataclass
class DataConfig:
    """Datastructure for the app."""

    directory: Path
    files: List[str] = field(default_factory=list)
    data: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    url: str = "https://go.fzj.de/madras-data"

    def __post_init__(self):
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


def increment_frame_start(page_size):
    st.session_state.start_frame += page_size


def decrement_frame_start(page_size):
    st.session_state.start_frame -= page_size


def reset_frame_start(start):
    st.session_state.start_frame = start


def init_session_state():
    """Init session_state. throughout the app."""

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

    dataconfig = datafactory.DataConfig(Path("AppData"))
    st.session_state.files = dataconfig.files


def unzip_files(zip_path: str, destination: str) -> None:
    """
    Unzips a ZIP file directly into the specified destination directory,
    ignoring the original directory structure in the ZIP file.

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
    Downloads a ZIP file from a specified URL, saves it to a given destination, and unzips it into a specified directory.
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

    # Unzip the file
    # with zipfile.ZipFile(destination, "r") as zip_ref:
    #     zip_ref.extractall(unzip_destination)

    unzip_files(destination, unzip_destination)

    progress_status.text("Unzipping complete.")
    progress_bar.empty()  # Clear the progress bar after completion


def load_file(filename: str) -> pedpy.TrajectoryData:
    """Load and processes a file to create a TrajectoryData object.

    This function reads a space-separated values file into a pedpy-trajectoryData
    fps = 16 and unit=meters

    Parameters:
    - filename (str): The path to the file to be loaded.

    Returns:
    - An instance of TrajectoryData containing the processed data and frame rate.
    """
    return pedpy.load_trajectory(
        trajectory_file=Path(filename),
        default_frame_rate=16,
        default_unit=pedpy.TrajectoryUnit.METER,
    )
