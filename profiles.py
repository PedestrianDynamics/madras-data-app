import pedpy as pp
from pedpy.column_identifier import SPEED_COL, FRAME_COL, ID_COL
from pedpy import (
    get_grid_cells,
    compute_grid_cell_polygon_intersection_area,
    plot_profiles,
    compute_speed_profile,
    compute_density_profile,
    DensityMethod,
    SpeedMethod
)
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging
import glob

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

area = [[-6, 0], [5, 0], [5, 7], [-6, 7]]
grid_size = 0.4
walkable_area = pp.WalkableArea(area)

logging.info("read trajectories")
files = sorted(glob.glob("AppData/*.txt"))

profile_file = "AppData/profile_data.pkl"
logging.info(f"read file {profile_file}")
if Path(profile_file).exists():
    with open(profile_file, "rb") as f:
        profile_data = pickle.load(f)

        
logging.info("compute_grid ...")
grid_cells, _, _ = get_grid_cells(
    walkable_area=walkable_area, grid_size=grid_size
)

def process_file(file_):
    logging.info(file_)
    profile_data_file = profile_data[file_]
    grid_cell_intersection_area, resorted_profile_data = compute_grid_cell_polygon_intersection_area(data=profile_data_file, grid_cells=grid_cells)

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


results = Parallel(n_jobs=-1)(delayed(process_file)(file_) for file_ in files)

# Aggregate results
density_profiles = {}
speed_profiles = {}  # Fixed variable name for consistency
for file_, density_profile, speed_profile in results:
    density_profiles[file_] = density_profile
    speed_profiles[file_] = speed_profile  # Fixed variable name for consistency


result_file = "AppData/density_speed_profiles.pkl"
with open(result_file, "wb") as f:
    pickle.dump((density_profiles, speed_profiles), f)

logging.info(f"Results in {result_file}")


    


# logging.info("Plotting ... ")

# rmax=4 
# vmax=2
# jmax=5
# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, layout="constrained")
# fig.set_size_inches(12, 5)
# fig.suptitle("Density profile")
# cm = plot_profiles(
#     walkable_area=walkable_area,
#     profiles=density_profile,
#     axes=ax0,
#     label=r"$\\rho$ / 1/$m^2$",
#     vmin=0,
#     vmax=rmax,
#     #title="Voronoi",
# )
# print(len(fig.axes))

# #colorbar_ax = fig.axes[]
# #colorbar_ax.set_ylabel("$\\rho$ / 1/$m^2$", size=18)
# #colorbar_ax.tick_params(labelsize=18)
# cm = plot_profiles(
#     walkable_area=walkable_area,
#     profiles=speed_profile,
#     axes=ax1,
#     label=r"$v / m/s$",
#     vmin=0,
#     vmax=vmax,
#     #title="Speed",
# )
# fig.tight_layout(pad=2)
# cm = plot_profiles(
#     walkable_area=walkable_area,
#     profiles=density_profile*speed_profile,
#     axes=ax2,
#     label=r"$J$ / 1/$m.s$",
#     vmin=0,
#     vmax=8,
#     #title="Classic",
# )

# for ax in [ax0, ax1, ax2]:
#     ax.tick_params(axis="x", length=0)
#     ax.tick_params(axis="y", length=0)
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])


# plt.savefig("Profiles.pdf")
# logging.info("Save results in Profiles.pdf")
