import os
import pickle
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

########## PARAMETERS ##########

FOLDER_TRAJ = Path("../../trajectories/")  # Folder with all the trajectory files
FOLDER_SAVE = Path("analysis")  # Folder where the fields should be saved

START_TIME = 0.0  # Start time of the video sequence, in seconds, since 8pm on Friday
DURATION = 10.0  # Duration of the animation, in seconds

X_MIN = 500  # Minimum x-coordinate of the field
X_MAX = -1  # Maximum x-coordinate of the field
Y_MIN = 500  # Minimum y-coordinate of the field
Y_MAX = -1  # Maximum y-coordinate of the field

DT = 1.0  # Time interval between two frames, in seconds
XI = 0.75  # Decay parameter for the Gaussian kernel
R_C = 4.0 * XI  # Cutoff distance for the Gaussian kernel
R_CG = 0.25  # Grid cell radius
DELTA = int(ceil(R_C / R_CG)) + 1  # Number of cells to consider around the cell containing the point

CUM_TIME = 0.0  # Cumulative time during which the trajectories are observed

# Butterworth Filter requirements
CUTOFF = 0.25  # Desired cutoff frequency of the filter, Hz
DELTA_T = 0.1  # Sampling interval, in seconds


########## FUNCTIONS ##########


def calculate_distance(pos1: tuple, pos2: tuple) -> float:
    """
    Calculate the distance between two points.

    Parameters:
    - pos1 (tuple): The first point (x1, y1).
    - pos2 (tuple): The second point (x2, y2).

    Returns:
    - float: The distance between the two points.
    """
    return np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)


def gaussian_kernel(r: float) -> float:
    """
    Calculate the phi value based on distance.

    Parameters:
    - r (float): The distance.

    Returns:
    - float: The phi value.
    """
    if r > R_C:
        return 0.0
    return np.exp(-0.5 * (r / XI) ** 2)  # Prefactor is omitted because it cancels out


def get_r(i: int, j: int) -> tuple:
    """
    Calculate the position based on grid indices. The position is the center of the cell.

    Parameters:
    - i (int): The grid index i.
    - j (int): The grid index j.

    Returns:
    - tuple: The position (x, y).
    """
    return (float(i) * R_CG + 0.5 * R_CG + X_MIN, float(j) * R_CG + 0.5 * R_CG + Y_MIN)


def butter_lowpass_filter(data: np.ndarray, delta_t: float, order: int) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter to the data.

    Parameters:
    - data (np.ndarray): The data to filter.
    - delta_t (float): The time interval between data points.
    - order (int): The order of the filter.

    Returns:
    - np.ndarray: The filtered data.
    """
    nyquist_freq = 0.5 / delta_t  # Nyquist Frequency
    normal_cutoff = CUTOFF / nyquist_freq  # Normalized cutoff frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)  # Generate the filter coefficients
    y = filtfilt(b, a, data, padlen=int(1 / delta_t) + 1)  # Apply the filter.
    # This function performs forward and backward filtering to eliminate phase distortion,
    # ensuring that the output signal has no phase shift relative to the input.
    # The padlen parameter is set to int(1 / delta_t) + 1 to determine the number of samples for edge padding,
    # which helps in reducing boundary effects in the filtering process.
    return y


########## MAIN CODE ##########

# Locate files starting with 'LargeView_zoom'
large_view_files = [f for f in os.listdir(FOLDER_TRAJ) if f.startswith("LargeView_zoom") and f.endswith(".txt")]

# Initialize a list to store DataFrames for each file
dataframes = []

# Process each file
for file_name in large_view_files:
    file_path = os.path.join(FOLDER_TRAJ, file_name)

    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep=" ", comment="#", header=None)

    # Assign column names based on the data structure provided
    df.columns = ["id", "frame", "x_m", "y_m", "z_m", "t_s", "x_RGF", "y_RGF"]

    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
all_data = pd.concat(dataframes, ignore_index=True)
all_data["t_s"] = all_data["t_s"] - all_data["t_s"].min()

# Initialize speeds
all_data["vx"] = np.nan
all_data["vy"] = np.nan

# Create folder and prepare saving
START_TIME = all_data["t_s"].min()
FOLDER_SAVE = FOLDER_SAVE / f"start={START_TIME:.1f}_dur={DURATION:.1f}_dt={DELTA_T:.1f}_xi={DT:.1f}_rcg={R_CG:.1f}"
FOLDER_SAVE.mkdir(parents=True, exist_ok=True)

# Save parameters to a text file
params_content = (
    f"DT={DT:.2f}\n"
    f"XI={XI:.2f}\n"
    f"R_C={R_C:.2f}\n"
    f"R_CG={R_CG:.2f}\n"
    f"START_TIME={START_TIME:.2f}\n"
    f"DURATION={DURATION:.2f}\n"
    f"CUTOFF_BUTTERWORTH={CUTOFF:.2f}\n"
    f"DELTA_T_BUTTERWORTH={DELTA_T:.2f}\n"
    f"DELTA={DELTA}\n"
)

# Initialize variables
all_trajs = {}
list_skipped = []

# Loop over id values
print(f"Processing {len(all_data['id'].unique())} trajectories...")
for traj_no in all_data["id"].unique():
    # Get the trajectory data
    traj_data = all_data[all_data["id"] == traj_no].copy()

    # Skip trajectories with no data
    if traj_data.empty or traj_data["t_s"].isnull().all():
        list_skipped.append(traj_no)
        continue

    # Smooth trajectories using Butterworth filter
    try:
        X_bw = butter_lowpass_filter(traj_data["x_m"].dropna(), DELTA_T, 2)
        Y_bw = butter_lowpass_filter(traj_data["y_m"].dropna(), DELTA_T, 2)
    except Exception as e:
        print(f"Skipping trajectory {traj_no} due to an error: {e}")
        list_skipped.append(traj_no)
        X_bw = traj_data["x_m"].dropna()
        Y_bw = traj_data["y_m"].dropna()
        continue

    # Interpolate between actual and smoothed trajectories
    start_time = traj_data["t_s"].min(skipna=True)
    end_time = traj_data["t_s"].max(skipna=True)
    alpha = traj_data["t_s"].apply(
        lambda t: max(
            np.exp(np.clip(-4.0 * CUTOFF * (t - start_time), -700, 700)),
            np.exp(np.clip(-4.0 * CUTOFF * (end_time - t), -700, 700)),
        )
    )
    traj_data.loc[:, "x_m"] = (1.0 - alpha) * X_bw + alpha * traj_data["x_m"]
    traj_data.loc[:, "y_m"] = (1.0 - alpha) * Y_bw + alpha * traj_data["y_m"]

    # Restrict focus to the specified time window
    traj_data = traj_data.loc[(traj_data["t_s"] >= START_TIME) & (traj_data["t_s"] < START_TIME + DURATION + 2.0 * DT)]

    # Skip if the trajectory is empty after filtering
    if traj_data.empty:
        list_skipped.append(traj_no)
        continue

    # Compute the cumulative time
    CUM_TIME += traj_data["t_s"].max(skipna=True) - traj_data["t_s"].min(skipna=True)

    # Update min and max coordinates
    X_MIN = min(X_MIN, traj_data["x_m"].min(skipna=True))
    X_MAX = max(X_MAX, traj_data["x_m"].max(skipna=True))
    Y_MIN = min(Y_MIN, traj_data["y_m"].min(skipna=True))
    Y_MAX = max(Y_MAX, traj_data["y_m"].max(skipna=True))

    # Calculate velocities
    time_max = traj_data["t_s"].max(skipna=True)
    for row in traj_data.loc[traj_data["t_s"] < time_max - DT].itertuples():
        current_time = row.t_s
        next_rows = traj_data[traj_data["t_s"] >= current_time + DT]
        if next_rows.empty:
            continue
        next_row = next_rows.iloc[0]
        next_time = next_row.t_s

        if next_time - current_time > 2.0 * DT:
            list_skipped.append(traj_no)
            continue

        traj_data.at[row.Index, "vx"] = (next_row.x_m - row.x_m) / (next_time - current_time)
        traj_data.at[row.Index, "vy"] = (next_row.y_m - row.y_m) / (next_time - current_time)

    # Remove invalid velocity entries with NaN values
    traj_data = traj_data.dropna(subset=["vx", "vy"])

    # Add the trajectory data to the dictionary
    all_trajs[traj_no] = traj_data


# Print skipped trajectories
print(f"{len(list_skipped)} trajs have been skipped: {list_skipped}")

# Save the all_trajs
with open(FOLDER_SAVE / "traj_data.pickle", "wb") as mydumpfile:
    pickle.dump(all_trajs, mydumpfile)

# Calculate grid dimensions
nb_cg_x = int((X_MAX - X_MIN) / R_CG) + DELTA + 2
nb_cg_y = int((Y_MAX - Y_MIN) / R_CG) + DELTA + 2

# Add grid dimensions to the parameters
params_content += f"X_MIN={X_MIN}\nX_MAX={X_MAX}\nY_MIN={Y_MIN}\nY_MAX={Y_MAX}\n"
params_content += f"nb_cg_x={nb_cg_x}\nnb_cg_y={nb_cg_y}\nCUM_TIME={CUM_TIME:.2f}\n"

# Save the parameters to a text file
with open(FOLDER_SAVE / "params.txt", "w", encoding="utf-8") as f:
    f.write(params_content)

# Initialize arrays
X = np.zeros((nb_cg_x, nb_cg_y), dtype="d")
Y = np.zeros((nb_cg_x, nb_cg_y), dtype="d")
rho_array = np.zeros((nb_cg_x, nb_cg_y), dtype="d")

# Fill X and Y arrays with grid coordinates
for i in range(nb_cg_x):
    for j in range(nb_cg_y):
        X[i, j], Y[i, j] = get_r(i, j)

# Initialize arrays for velocity fields
vxs = np.zeros((nb_cg_x, nb_cg_y), dtype="d")
vys = np.zeros((nb_cg_x, nb_cg_y), dtype="d")
vxs2 = np.zeros((nb_cg_x, nb_cg_y), dtype="d")
vys2 = np.zeros((nb_cg_x, nb_cg_y), dtype="d")
std_vs = np.zeros((nb_cg_x, nb_cg_y), dtype="d")

# Compute mean velocity field
for i in range(nb_cg_x):
    for j in range(nb_cg_y):
        for traj in all_trajs.values():
            if traj.empty:
                continue

            # Calculate distance to the centre of the cell selected
            traj["dist_to_centre_cell"] = calculate_distance((X[i, j], Y[i, j]), (traj["x_m"], traj["y_m"]))
            ind_min = traj["dist_to_centre_cell"].idxmin()
            row = traj.loc[ind_min]
            phi_r = gaussian_kernel(row.dist_to_centre_cell)

            # Update sums
            rho_array[i, j] += phi_r
            vxs[i, j] += phi_r * row.vx
            vys[i, j] += phi_r * row.vy
            vxs2[i, j] += phi_r * row.vx**2
            vys2[i, j] += phi_r * row.vy**2

        # Normalize and calculate standard deviation
        if rho_array[i, j] > 1e-10:
            vxs[i, j] /= rho_array[i, j]
            vys[i, j] /= rho_array[i, j]
            vxs2[i, j] /= rho_array[i, j]
            vys2[i, j] /= rho_array[i, j]
            vxs2[i, j] -= vxs[i, j] ** 2
            vys2[i, j] -= vys[i, j] ** 2
            # Ensure non-negative values for sqrt
            variance_sum = np.nan_to_num(vxs2[i, j] + vys2[i, j], nan=0.0, posinf=0.0, neginf=0.0)
            std_vs[i, j] = np.sqrt(np.maximum(variance_sum, 0.0))

# Save the data
FOLDER_SAVE = Path(FOLDER_SAVE)
save_data = {
    "X.pickle": X,
    "Y.pickle": Y,
    "Somme_phi.pickle": rho_array,
    "vx_mean.pickle": vxs,
    "vy_mean.pickle": vys,
    "vx_var.pickle": vxs2,
    "vy_var.pickle": vys2,
    "v_std.pickle": std_vs,
}

for filename, data in save_data.items():
    with open(FOLDER_SAVE / filename, "wb") as mydumpfile:
        pickle.dump(data, mydumpfile)
