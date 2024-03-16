"""Calculate speed profiles using Gaussian kernel"""

from typing import List, Optional, Tuple
import shapely
import numpy as np
import pandas as pd
import numpy.typing as npt
import pandas
from pedpy.column_identifier import FRAME_COL
from pedpy import SpeedMethod, WalkableArea


def _compute_gaussian_weights(x: npt.NDArray[np.float64], fwhm: float) -> npt.NDArray[np.float64]:
    """Computes the Gaussian density for given values and FWHM.

    The Gaussian density is defined as:
        G(x) = 1 / (sigma * sqrt(2 * pi)) * exp(-x^2 / (2 * sigma^2))
    where sigma is derived from FWHM as:
        sigma = FWHM / (2 * sqrt(2 * ln(2)))

    Args:
        x: Value(s) for which the Gaussian should be computed.
        fwhm: Full width at half maximum, a measure of spread.

    Returns:
        Gaussian density corresponding to the given values and FWHM.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x**2) / (2 * sigma**2))


def compute_gaussian_weighted_speed_profile(
    *, frame_data: pd.DataFrame, center_x: npt.NDArray[np.float64], center_y: npt.NDArray[np.float64], fwhm: float, fill_value: float = np.nan, grid_size
) -> np.ndarray:
    positions_x = frame_data.x.values
    positions_y = frame_data.y.values
    speeds = frame_data.speed.values

    # distance from each grid center x/y coordinates to the pedestrian positions
    distance_x = np.subtract.outer(center_x, positions_x)
    distance_y = np.subtract.outer(center_y, positions_y)

    distance_x_expanded = np.expand_dims(distance_x, axis=1)
    distance_y_expanded = np.expand_dims(distance_y, axis=0)

    distance = np.sqrt(distance_x_expanded**2 + distance_y_expanded**2)

    weights = _compute_gaussian_weights(distance, fwhm)
    normalized_weights = weights  # / np.sum(weights, axis=(0, 1), keepdims=True)
    weighted_speeds = np.tensordot(normalized_weights, speeds, axes=([2], [0]))
    return np.array(weighted_speeds.T)


# todo: copy/pasted from pedpy
def get_grid_cells(*, walkable_area: WalkableArea, grid_size: float) -> Tuple[npt.NDArray[shapely.Polygon], int, int]:
    """Creates a list of square grid cells covering the geometry.

    .. image:: /images/profile_grid.svg
        :width: 60 %
        :align: center

    Args:
        walkable_area (shapely.Polygon): geometry for which the profiles are
            computed.
        grid_size (float): resolution of the grid used for computing the
            profiles.

    Returns:
        (List of grid cells, number of grid rows, number of grid columns)
    """
    bounds = walkable_area.bounds
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(max_y, min_y - grid_size, -grid_size)

    grid_cells = []
    for j in range(len(y_coords) - 1):
        for i in range(len(x_coords) - 1):
            grid_cell = shapely.box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
            grid_cells.append(grid_cell)

    return np.array(grid_cells), len(y_coords) - 1, len(x_coords) - 1


def compute_speed_profile(
    *,
    data: pd.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
    speed_method: SpeedMethod,
    grid_intersections_area: Optional[npt.NDArray[np.float64]] = None,
    fill_value: float = np.nan,
    width: float,
) -> List[npt.NDArray[np.float64]]:
    speed_profiles = []
    data_grouped_by_frame = data.groupby(FRAME_COL)
    grid_cells, rows, cols = get_grid_cells(walkable_area=walkable_area, grid_size=grid_size)
    grid_center = np.vectorize(shapely.centroid)(grid_cells)
    center_x = shapely.get_x(grid_center[:cols])
    center_y = shapely.get_y(grid_center[::cols])
    for frame, frame_data in data_grouped_by_frame:
        if True or speed_method == SpeedMethod.GAUSSIAN:  # todo
            speed_profile = compute_gaussian_weighted_speed_profile(
                frame_data=frame_data,
                center_x=center_x,
                center_y=center_y,
                grid_size=grid_size,
                fill_value=fill_value,
                width=width,
            )
        else:
            raise ValueError(f"Speed method {speed_method} not accepted")

        speed_profiles.append(speed_profile.reshape(rows, cols))

    return speed_profiles
