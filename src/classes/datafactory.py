"""Unsorted datastructure for the app."""

import glob
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import List, Union
import os
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

    zenodo_directory: Path
    flow_directory: Path
    # results
    processed_directory: Path
    files: List[str] = field(default_factory=list)
    url: str = "https://zenodo.org/records/13830435/files/Data_Madras.zip?download=1"  # "https://go.fzj.de/madras-data"

    def __post_init__(self) -> None:
        """Initialize the DataConfig instance by retrieving files."""
        # self.data.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Create {self.processed_directory}")
        self.processed_directory.mkdir(parents=True, exist_ok=True)
        self.retrieve_files()

    def retrieve_files(self) -> None:
        """Retrieve the files for each country specified in the list."""
        logging.info("Retrieve data ...")
        if not self.zenodo_directory.exists():
            st.warning(f"{self.zenodo_directory} does not exist yet!")
            with st.status("Downloading ...", expanded=False):
                zip_path = "Data_Madras.zip"
                unzip_root = self.zenodo_directory.parent
                download_and_unzip_files(self.url, zip_path, unzip_root)
        else:
            logging.info("Found zenodo directory. Nothing to retrieve!")

        self.files = sorted(
            f
            for f in glob.glob(f"{self.zenodo_directory}/**/*.txt", recursive=True)
            if "density" not in os.path.basename(f).lower() and "tracers" not in os.path.basename(f).lower()
        )


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
    zenodo_directory = path.parent.parent.parent.absolute() / "data" / "zenodo"
    flow_directory = path.parent.parent.parent.absolute() / "data" / "flow"
    processed_directory = path.parent.parent.parent.absolute() / "data" / "processed"

    logging.info(f"{zenodo_directory = }")
    init_state_bg_image()
    # Initialize a list of DirectionInfo objects using the provided dictionaries
    if "direction_infos" not in st.session_state:
        st.session_state.direction_infos = [
            DirectionInfo(id=1, name="Top", color="blue"),
            DirectionInfo(id=2, name="Bottom", color="red"),
            DirectionInfo(id=3, name="Right", color="green"),
            DirectionInfo(id=4, name="Left", color="gray"),
        ]

    if "start_frame" not in st.session_state:
        st.session_state.start_frame = 0

    if "config" not in st.session_state:
        st.session_state.config = DataConfig

    if not hasattr(st.session_state, "files"):
        st.session_state.files = []

    if not hasattr(st.session_state, "selected_file"):
        st.session_state.selected_file = ""

    if not hasattr(st.session_state, "file_changes"):
        st.session_state.file_changed = ""

    if not hasattr(st.session_state, "trajectory_data"):
        st.session_state.trajectoryData = pedpy.TrajectoryData

    dataconfig = DataConfig(
        zenodo_directory=zenodo_directory,
        processed_directory=processed_directory,
        flow_directory=flow_directory,
    )
    st.session_state.files = dataconfig.files
    st.session_state.config = dataconfig


def unzip_files(zip_path: Union[str, Path], destination: Union[str, Path]) -> None:
    """
    Unzip a ZIP file into the specified destination directory,
    preserving the internal structure but renaming the top-level
    folder to 'zenodo'.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = PurePosixPath(member.filename)

            if not member_path.parts:
                continue

            # Drop the original top-level folder (e.g. Data_Madras/)
            relative_parts = member_path.parts[1:]

            # Rebuild under destination/zenodo/
            target_path = destination / "zenodo" / Path(*relative_parts)

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(member, "r") as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)


def download_and_unzip_files(url: str, destination: str | Path, unzip_destination: str | Path) -> None:
    """
    Download a ZIP file from a specified URL.

    Saves it to a given destination, and unzips it into a specified directory.
    Displays the download and unzipping progress in a Streamlit app.
    """
    destination = Path(destination)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kbyte
    progress_bar = st.progress(0)
    progress_status = st.empty()
    written = 0

    with open(destination, "wb") as f:
        for data in response.iter_content(block_size):
            written += len(data)
            f.write(data)
            if total_size > 0:
                progress = int(100 * written / total_size)
                progress_bar.progress(progress)
                progress_status.text(f"> {progress}%")

    progress_status.text("Download complete. Unzipping...")
    unzip_files(destination, unzip_destination)

    progress_status.text("Unzipping complete.")
    progress_bar.empty()


def load_file(filename: str) -> pedpy.TrajectoryData:
    """
    Load and processes a file to create a TrajectoryData object.

    This function reads a space-separated values file into a pedpy-trajectoryData
    fps = 30 and unit=meters

    Args:
        filename (str): The path to the file to be loaded.

    Returns:
        pedpy.TrajectoryData: An instance of TrajectoryData containing the processed data and frame rate.
    """
    return pedpy.load_trajectory(
        trajectory_file=Path(filename),
        default_frame_rate=30,
        default_unit=pedpy.TrajectoryUnit.METER,
    )
