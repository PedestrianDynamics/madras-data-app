"""Master module for profile calculations.

This module calls other modules to calculate density, speed and flow profiles as pdf files.
"""

import glob
from pathlib import Path

import calculate_profiles
import create_profile_data
import ploting_profiles
from pedpy import WalkableArea
from profile_config_data import Config


def run_all() -> None:
    """Initialize the configuration."""
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    trajectories_directory = path.parent.parent.parent.absolute() / "data" / "trajectories"
    area = [[-6, 0], [5, 0], [5, 7], [-6, 7]]
    config = Config(
        files=sorted(glob.glob(f"{str(trajectories_directory)}/*.txt")),
        grid_size=0.4,
        speed_frame_rate=10,
        fps=30,
        walkable_area=WalkableArea(area),
        profile_data_file=f"{str(data_directory)}/profile_data.pkl",
        result_file=f"{str(data_directory)}/density_speed_profiles.pkl",
        rmax=3.0,
        vmax=1.2,
        jmax=2.0,
    )

    # Run the modules in sequence
    create_profile_data.main(config)
    calculate_profiles.main(config)
    ploting_profiles.main(config)


if __name__ == "__main__":
    run_all()
