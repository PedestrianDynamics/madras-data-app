import pickle
from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

########## PARAMETERS ##########

start_time = 0.0  #  start time of the video sequence, in seconds, since 8pm on Friday
duration = 10.0  # duration of the animation, in seconds
dt = 0.1
xi = 1.00
r_c = 4.0 * xi
r_cg = 0.2

FOLDER_TRAJ = Path("../../trajectories/")
FOLDER_SAVE = Path("analysis") / f"start={start_time}_dur={duration}_dt={dt}_xi={xi}_rcg={r_cg}"

########## LOAD DATA ##########

# Load the parameters
params_path = FOLDER_SAVE / "params.txt"
with open(params_path, "r", encoding="utf-8") as file:
    params = file.readlines()
params_dict = {line.split("=")[0]: float(line.split("=")[1]) for line in params}

(
    DT,
    XI,
    R_C,
    R_CG,
    START_TIME,
    DURATION,
    CUTOFF,
    DELTA_T,
    DELTA,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    nb_cg_x,
    nb_cg_y,
    CUM_TIME,
) = params_dict.values()
nb_cg_x = int(nb_cg_x)
nb_cg_y = int(nb_cg_y)
DELTA = int(DELTA)

# Read the data
with open(FOLDER_SAVE / "X.pickle", "rb") as mydumpfile:
    X = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "Y.pickle", "rb") as mydumpfile:
    Y = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "Somme_phi.pickle", "rb") as mydumpfile:
    rho_array = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "vx_mean.pickle", "rb") as mydumpfile:
    vxs = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "vy_mean.pickle", "rb") as mydumpfile:
    vys = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "vx_var.pickle", "rb") as mydumpfile:
    vxs2 = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "vy_var.pickle", "rb") as mydumpfile:
    vys2 = pickle.load(mydumpfile)
with open(FOLDER_SAVE / "v_std.pickle", "rb") as mydumpfile:
    std_vs = pickle.load(mydumpfile)

# Load the trajectories
with open(FOLDER_SAVE / "traj_data.pickle", "rb") as mydumpfile:
    all_trajs = pickle.load(mydumpfile)


########## FUNCTIONS ##########


def euclidean_norm(v: np.ndarray) -> float:
    """
    Calculate the Euclidean norm of a 2D vector.

    Parameters:
    - v (np.ndarray): A 2D vector.

    Returns:
    - float: The Euclidean norm of the vector.
    """
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


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


def get_cell(r: tuple) -> tuple:
    """
    Calculate the cell indices for a given position.

    Parameters:
    - r (tuple): The position (x, y).

    Returns:
    - tuple: The cell indices (i, j).
    """
    i = int(floor((r[0] - X_MIN) / R_CG))
    j = int(floor((r[1] - Y_MIN) / R_CG))
    return (i, j)


########## PLOT DENSITY ##########

# Create the figure
fig, ax = plt.subplots(figsize=(5.5, 7))
nb_ped = len(all_trajs.keys())
density = np.zeros((nb_cg_x, nb_cg_y), dtype="d")

# Calculate the density
for traj in all_trajs.values():
    traj = traj.loc[(traj["t_s"] >= START_TIME) & (traj["t_s"] < START_TIME + DURATION + 2.0 * DT)]

    if traj.empty:
        continue

    for row in traj.itertuples():
        R = (row.x_m, row.y_m)
        i_rel, j_rel = get_cell(R)
        for i in range(i_rel - DELTA, i_rel + DELTA + 1):
            for j in range(j_rel - DELTA, j_rel + DELTA + 1):
                if 0 <= i < nb_cg_x and 0 <= j < nb_cg_y:
                    phi_r = gaussian_kernel(calculate_distance(get_r(i, j), R))
                    density[i, j] += phi_r

N_nonrenormalised = R_CG**2 * np.nansum(density)
density *= float(CUM_TIME / DURATION) / float(N_nonrenormalised)
print("Total number of ped: ", nb_ped, " vs ", CUM_TIME / DURATION)

# Plot the density
Xp1 = np.zeros((nb_cg_x + 1, nb_cg_y + 1), dtype="d")
Yp1 = np.zeros((nb_cg_x + 1, nb_cg_y + 1), dtype="d")

for i in range(nb_cg_x + 1):
    for j in range(nb_cg_y + 1):
        Xp1[i, j] = float(i) * R_CG + X_MIN
        Yp1[i, j] = float(j) * R_CG + Y_MIN

cmesh = ax.pcolormesh(Xp1, Yp1, density, cmap="YlOrRd", vmax=4, shading="auto")
cb = plt.colorbar(cmesh, ax=ax)
cb.ax.set_title("Density (ped/m²)")

ax.set_xlabel("x [m]", fontsize=14)
ax.set_ylabel("y [m]", fontsize=14)

step = 4
for i in range(nb_cg_x):
    for j in range(nb_cg_y):
        if X[i, j] > 11.5:
            vxs[i, j] = 0
            vys[i, j] = 0

ax.quiver(
    X[::step, ::step], Y[::step, ::step], vxs[::step, ::step], vys[::step, ::step], linewidth=2, headwidth=7, scale=5.0
)
ax.quiver(9, 2.5, 1, 0, linewidth=3, headwidth=7, scale=5.0, color="green")
ax.text(9.5, 3, "1 m/s", color="green", fontsize=12)

ax.axis("equal")
plt.tight_layout()

ax.set_xlim(0.149, 11.0)
ax.set_ylim(2.183, 23)

plt.show()
plt.close()
