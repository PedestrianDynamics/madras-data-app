"""Input profile data and calculate the profiles."""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pedpy as pp
from joblib import Parallel, delayed
from log_config import setup_logging
from pedpy import (
    DensityMethod,
    SpeedMethod,
    compute_grid_cell_polygon_intersection_area,
    compute_speed_profile,
    get_grid_cells,
)
from profile_config_data import Config
from tqdm import tqdm


def process_file(
    file_: str,
    grid_cells: List[Any],
    profile_data: Dict[str, pd.DataFrame],
    walkable_area: pp.WalkableArea,
    grid_size: float,
) -> Tuple[str, List[npt.NDArray[np.float64]], List[np.typing.NDArray[np.float64]]]:
    """Calculate density profile and speed profile."""
    logging.info(file_)
    profile_data_file = profile_data[file_]
    grid_cell_intersection_area, resorted_profile_data = (
        compute_grid_cell_polygon_intersection_area(
            data=profile_data_file, grid_cells=grid_cells
        )
    )
    logging.info("Compute density profile")
    density_profile = pp.compute_density_profile(
        data=resorted_profile_data,
        walkable_area=walkable_area,
        grid_intersections_area=grid_cell_intersection_area,
        grid_size=grid_size,
        density_method=DensityMethod.VORONOI,
    )
    logging.info("Compute speed profile")
    speed_profile = compute_speed_profile(
        data=resorted_profile_data,
        walkable_area=walkable_area,
        grid_intersections_area=grid_cell_intersection_area,
        grid_size=grid_size,
        speed_method=SpeedMethod.VORONOI,
    )
    return (file_, density_profile, speed_profile)


def calculate(
    result_file: str,
    grid_size: float,
    files: List[str],
    walkable_area: pp.WalkableArea,
    profile_data: Dict[str, pd.DataFrame],
) -> None:
    """Parallelize calculation of density and speed profile data and save a pickle file.

    This function utilizes the process_file function to perform calculations in parallel across the given files.
    The results are then serialized and saved to the specified result_file.

    """
    logging.info("Compute_grid ...")
    grid_cells, _, _ = get_grid_cells(walkable_area=walkable_area, grid_size=grid_size)
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(file_, grid_cells, profile_data, walkable_area, grid_size)
        for file_ in tqdm(files, desc="Processing files")
    )
    logging.info("Aggregate results")
    density_profiles = {}
    speed_profiles = {}
    for file_, density_profile, speed_profile in results:
        density_profiles[file_] = density_profile
        speed_profiles[file_] = speed_profile

    with open(result_file, "wb") as f:
        pickle.dump((density_profiles, speed_profiles), f)

    logging.info(f"Results in {result_file}")


def main(config: Config) -> None:
    """Contains main logic for calculation of profiles."""
    setup_logging()
    grid_size = config.grid_size
    result_file = config.result_file
    walkable_area = config.walkable_area
    logging.info("Read trajectories")
    files = config.files
    profile_data_file = config.profile_data_file
    logging.info(f"Read file {profile_data_file}")
    if Path(profile_data_file).exists():
        with open(profile_data_file, "rb") as f:
            profile_data = pickle.load(f)
            profile_data = cast(Dict[str, pd.DataFrame], profile_data)
    else:
        logging.error(f"file: {profile_data_file} does not exist!")
        sys.exit()
    logging.info("Calculate profiles ...")
    calculate(
        result_file=result_file,
        grid_size=grid_size,
        files=files,
        walkable_area=walkable_area,
        profile_data=profile_data,
    )
