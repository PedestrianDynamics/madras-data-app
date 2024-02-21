"""Analysis data, speed, density, flow, etc."""

import glob
import os
import pickle
import time
from pathlib import Path

import pedpy
import streamlit as st
from shapely import Polygon

import datafactory
import docs
import plots


def setup_walkable_area(trajectory_data):
    min_x = trajectory_data.data["x"].min()
    max_x = trajectory_data.data["x"].max()
    min_y = trajectory_data.data["y"].min()
    max_y = trajectory_data.data["y"].max()
    rectangle_coords = [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
    rectangle_polygon = Polygon(rectangle_coords)
    walkable_area = pedpy.WalkableArea(rectangle_polygon)

    return walkable_area


def calculate_or_load_classical_density(
    precalculated_density,
    filename,
):
    """Calculate classical density or load existing calculation."""
    if not Path(precalculated_density).exists():
        trajectory_data = datafactory.load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        classic_density = pedpy.compute_classic_density(
            traj_data=trajectory_data, measurement_area=walkable_area
        )
        with open(precalculated_density, "wb") as f:
            pickle.dump(classic_density, f)
    else:
        print(f"load precalculated density: {precalculated_density}")
        with open(precalculated_density, "rb") as f:
            classic_density = pickle.load(f)

    return classic_density


def calculate_or_load_voronoi_diagrams(
    precalculated_voronoi_polygons,
    filename,
):
    """Calculate Voronoi diagrams or load existing calculation."""
    if not Path(precalculated_voronoi_polygons).exists():
        trajectory_data = datafactory.load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        voronoi_polygons = pedpy.compute_individual_voronoi_polygons(
            traj_data=trajectory_data, walkable_area=walkable_area
        )

        with open(precalculated_voronoi_polygons, "wb") as f:
            pickle.dump(voronoi_polygons, f, pickle.HIGHEST_PROTOCOL)
    else:
        print(f"load precalculated voronoi polygons: {precalculated_voronoi_polygons}")
        with open(precalculated_voronoi_polygons, "rb") as f:
            voronoi_polygons = pickle.load(f)

    return voronoi_polygons


def calculate_or_load_voronoi_speed(
    precalculated_voronoi_speed,
    intersecting,
    individual_speed,
    filename,
):
    """Calculate Voronoi speed or load existing calculation."""
    if not Path(precalculated_voronoi_speed).exists():
        trajectory_data = datafactory.load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        voronoi_speed = pedpy.compute_voronoi_speed(
            traj_data=trajectory_data,
            individual_voronoi_intersection=intersecting,
            measurement_area=walkable_area,
            individual_speed=individual_speed,
        )
        with open(precalculated_voronoi_speed, "wb") as f:
            pickle.dump(voronoi_speed, f, pickle.HIGHEST_PROTOCOL)
    else:
        print(f"load precalculated voronoi speed: {precalculated_voronoi_speed}")
        with open(precalculated_voronoi_speed, "rb") as f:
            voronoi_speed = pickle.load(f)

    return voronoi_speed


def calculate_or_load_voronoi_density(
    precalculated_voronoi_density,
    voronoi_polygons,
    filename,
):
    """Calculate Voronoi density or load existing calculation."""
    if not Path(precalculated_voronoi_density).exists():
        trajectory_data = datafactory.load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        voronoi_density, intersecting = pedpy.compute_voronoi_density(
            individual_voronoi_data=voronoi_polygons,
            measurement_area=walkable_area,
        )

        with open(precalculated_voronoi_density, "wb") as f:
            pickle.dump((voronoi_density, intersecting), f, pickle.HIGHEST_PROTOCOL)
    else:
        print(f"load precalculated voronoi density: {precalculated_voronoi_density}")
        with open(precalculated_voronoi_density, "rb") as f:
            voronoi_density, intersecting = pickle.load(f)

    return voronoi_density, intersecting


def calculate_or_load_individual_speed(
    precalculated_speed: str, filename: str, dv: int
):
    """Calculate speed or load precalculated values if exist"""
    if not Path(precalculated_speed).exists():
        trajectory_data = datafactory.load_file(filename)
        individual_speed = pedpy.compute_individual_speed(
            traj_data=trajectory_data,
            frame_step=dv,
            speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
        )
        with open(precalculated_speed, "wb") as f:
            pickle.dump(individual_speed, f)
    else:
        print(f"load precalculated speed: {precalculated_speed}")
        with open(precalculated_speed, "rb") as f:
            individual_speed = pickle.load(f)

    return individual_speed


def run_tab3(selected_file):
    docs_expander = st.expander("Documentation (click to expand)", expanded=False)
    with docs_expander:
        docs.density_speed()

    c0, c1, c2 = st.columns((1, 1, 1))
    if c1.button(
        "Delete Precalculated Files",
        help="To enhance performance, some densities/speeds are loaded instead of calculated. With this button you can delete these files and hence start fresh claculations",
    ):
        precalculated_files_pattern = "AppData/*.pkl"
        files_to_delete = glob.glob(precalculated_files_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                st.toast(f"Deleted {file_path}", icon="✅")
            except Exception as e:
                st.error(f"Error deleting {file_path}: {e}")

    c2.write("**Speed calculation parameters**")
    calculations = c0.radio(
        "Choose calculation",
        [
            "time_series",
            "FD_classical",
            "FD_voronoi",
        ],
        horizontal=False,
    )
    dv = c2.slider(
        r"$\Delta t$",
        1,
        100,
        10,
        5,
        help="To calculate the displacement over a specified number of frames. See Eq. (1)",
    )

    if calculations == "time_series":
        if selected_file != st.session_state.file_changed:
            with st.status(f"Loading {selected_file}"):
                trajectory_data = datafactory.load_file(selected_file)
                st.session_state.trajectory_data = trajectory_data

            st.session_state.file_changed = selected_file

        trajectory_data = st.session_state.trajectory_data
        walkable_area = setup_walkable_area(trajectory_data)

        individual_speed = pedpy.compute_individual_speed(
            traj_data=trajectory_data,
            frame_step=dv,
            speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
        )
        mean_speed = pedpy.compute_mean_speed_per_frame(
            traj_data=trajectory_data,
            measurement_area=walkable_area,
            individual_speed=individual_speed,
        )

        classic_density = pedpy.compute_classic_density(
            traj_data=trajectory_data, measurement_area=walkable_area
        )

        fig = plots.plot_time_series(classic_density, mean_speed, fps=16)
        figname = selected_file.split("/")[-1].split(".txt")[0] + ".pdf"
        plots.show_fig(fig, figname=figname, html=True, write=True)

    if calculations == "FD_classical":
        densities = {}
        speeds = {}
        with st.status("Calculating...", expanded=True):
            progress_bar = st.progress(0)
            progress_status = st.empty()
            for i, filename in enumerate(st.session_state.files):
                basename = filename.split("/")[-1].split(".txt")[0]
                precalculated_density = f"AppData/density_{basename}.pkl"
                precalculated_speed = f"AppData/speed_{basename}_{dv}.pkl"

                speeds[basename] = calculate_or_load_individual_speed(
                    precalculated_speed, filename, dv
                )
                densities[basename] = calculate_or_load_classical_density(
                    precalculated_density, filename
                )
                progress = int(100 * (i + 1) / len(st.session_state.files))
                progress_bar.progress(progress)
                progress_status.text(f"> {progress}%")

        figname = "fundamental_diagram_classical.pdf"
        fig = plots.plot_fundamental_diagram_all(densities, speeds)
        plots.show_fig(fig, figname=figname, html=True, write=True)

    if calculations == "FD_voronoi":
        voronoi_polygons = {}
        voronoi_density = {}
        voronoi_speed = {}
        individual_speed = {}
        intersecting = {}
        start = time.time()
        with st.status("Calculating Voronoi FD ...", expanded=True):
            progress_bar = st.progress(0)
            progress_status = st.empty()
            for i, filename in enumerate(st.session_state.files):

                basename = filename.split("/")[-1].split(".txt")[0]
                # saved files ============
                precalculated_voronoi_polygons = (
                    f"AppData/voronoi_polygons_{basename}.pkl"
                )
                precalculated_speed = f"AppData/speed_{basename}_{dv}.pkl"
                precalculated_voronoi_speed = f"AppData/voronoi_speed_{basename}.pkl"
                precalculated_voronoi_density = (
                    f"AppData/voronoi_density_{basename}.pkl"
                )
                # saved files ============
                voronoi_polygons[basename] = calculate_or_load_voronoi_diagrams(
                    precalculated_voronoi_polygons, filename
                )

                individual_speed[basename] = calculate_or_load_individual_speed(
                    precalculated_speed, filename, dv
                )
                # todo save to files
                trajectory_data = datafactory.load_file(filename)
                walkable_area = setup_walkable_area(trajectory_data)

                # voronoi_density[basename], intersecting[basename] = (
                #     pedpy.compute_voronoi_density(
                #         individual_voronoi_data=voronoi_polygons[basename],
                #         measurement_area=walkable_area,
                #     )
                # )
                voronoi_density[basename], intersecting[basename] = (
                    calculate_or_load_voronoi_density(
                        precalculated_voronoi_density,
                        voronoi_polygons[basename],
                        filename,
                    )
                )

                voronoi_speed[basename] = calculate_or_load_voronoi_speed(
                    precalculated_voronoi_speed,
                    intersecting[basename],
                    individual_speed[basename],
                    filename,
                )

                progress = int(100 * (i + 1) / len(st.session_state.files))
                progress_bar.progress(progress)
                progress_status.text(f"> {progress}%")

        end = time.time()
        st.info(f"Computation time: {end-start:.2f} seconds.")
        figname = "fundamental_diagram_voronoi.pdf"
        fig = plots.plot_fundamental_diagram_all(voronoi_density, voronoi_speed)
        plots.show_fig(fig, figname=figname, html=True, write=True)
