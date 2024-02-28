from pedpy.column_identifier import FRAME_COL, SPEED_COL, ID_COL
from pedpy import get_grid_cells
import numpy as np


def compute_classic_speed_profile(frame_data, walkable_area, grid_size):
    min_x, min_y, max_x, max_y = walkable_area.bounds

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)

    hist, _, _ = np.histogram2d(x=frame_data.x, y=frame_data.y, bins=[x_coords, y_coords])
    hist_speed, _, _ = np.histogram2d(
        x=frame_data.x,
        y=frame_data.y,
        bins=[x_coords, y_coords],
        weights=frame_data.speed,
    )
    speed_zero = np.divide(
        hist_speed,
        hist,
        # out=np.full(shape=hist.shape, fill_value=np.nan),
        out=np.zeros_like(hist),
        where=hist != 0,
    )
    speed_nan = np.divide(
        hist_speed,
        hist,
        out=np.full(shape=hist.shape, fill_value=np.nan),
        where=hist != 0,
    )

    # rotate the result, such that is displayed with imshow correctly and
    # has the same orientation as the other results
    speed_zero = np.rot90(speed_zero)
    speed_nan = np.rot90(speed_nan)

    return speed_zero, speed_nan


def compute_speed_profile(data, walkable_area, grid_size):
    grid_cells, rows, cols = get_grid_cells(walkable_area=walkable_area, grid_size=grid_size)

    data_grouped_by_frame = data.groupby(FRAME_COL)

    speed_zero_profiles = []
    speed_nan_profiles = []

    for _, frame_data in data_grouped_by_frame:
        speed_zero, speed_nan = compute_classic_speed_profile(
            frame_data=frame_data,
            walkable_area=walkable_area,
            grid_size=grid_size,
        )

        speed_zero_profiles.append(speed_zero.reshape(rows, cols))
        speed_nan_profiles.append(speed_nan.reshape(rows, cols))

    return speed_zero_profiles, speed_nan_profiles


# speed_profiles = compute_speed_profile(combined, walkable_area, 0.5)
