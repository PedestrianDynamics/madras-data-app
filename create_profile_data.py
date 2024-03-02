"""Create Voronoi profile density."""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import pedpy as pp
from tqdm import tqdm
from joblib import Parallel, delayed
from pedpy.column_identifier import FRAME_COL, ID_COL

from log_config import setup_logging
from profile_config_data import Config


def process_trajectory(
    file_: str, walkable_area: pp.WalkableArea, frame_step: int, fps: int = 30
) -> Tuple[str, pd.DataFrame]:
    """Calculate Voronoi polys, Individual speed, Voronoi density and merge all with traj data.

    frame_step: steps forward and backward. See pedpy docu for compute_individual_speed.
    fps: frames per second of the trajectory data
    """
    traj = pp.load_trajectory(
        trajectory_file=Path(file_),
        default_frame_rate=fps,
        default_unit=pp.TrajectoryUnit.METER,
    )

    voronoi_polygons = pp.compute_individual_voronoi_polygons(
        traj_data=traj, walkable_area=walkable_area
    )
    density_voronoi, intersecting = pp.compute_voronoi_density(
        individual_voronoi_data=voronoi_polygons,
        measurement_area=walkable_area,
    )
    individual_speed = pp.compute_individual_speed(
        traj_data=traj,
        frame_step=frame_step,
        compute_velocity=True,
        speed_calculation=pp.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    merged_data = individual_speed.merge(voronoi_polygons, on=[ID_COL, FRAME_COL])
    merged_data = merged_data.merge(traj.data, on=[ID_COL, FRAME_COL])

    return (file_, merged_data)


def main(config: Config) -> None:
    """Process_trajectory in parallel and dump result in file."""
    setup_logging()
    files = config.files
    profile_data_file = config.profile_data_file
    frame_step = config.speed_frame_rate
    fps = config.fps
    walkable_area = config.walkable_area
    # Parallel execution
    logging.info("Process trajectories and create profile data ...")
    results = Parallel(n_jobs=-1)(
        delayed(process_trajectory)(file_, walkable_area, frame_step, fps)
        for file_ in tqdm(files, desc="Processing files")
    )
    # Aggregate results into 'profile_data'
    # profile_data = {file: data for file, data in results}
    profile_data = dict(results)
    with open(profile_data_file, "wb") as f:
        pickle.dump(profile_data, f)

    logging.info(f"Profile data computed and saved to {profile_data_file}")
