"""Analysis data, speed, density, flow, etc."""

import pedpy
import streamlit as st
from shapely import Polygon
from pathlib import Path
import pickle

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


def calculate_or_load_speed(precalculated_speed: str, filename: str, dv: int):
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

    c2.write("**Speed calculation parameters**")
    calculations = c0.radio(
        "Choose calculation",
        [
            "time_series",
            "FD",
        ],
        horizontal=True,
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

    if calculations == "FD":
        st.warning("Not yet implemented!")
        densities = {}
        speeds = {}
        for filename in st.session_state.files:
            basename = filename.split("/")[-1].split(".txt")[0]
            precalculated_density = f"AppData/density_{basename}.pkl"
            precalculated_speed = f"AppData/speed_{basename}_{dv}.pkl"

            speeds[basename] = calculate_or_load_speed(
                precalculated_speed, filename, dv
            )
            densities[basename] = calculate_or_load_classical_density(
                precalculated_density, filename
            )

        figname = "fundamental_diagram.pdf"
        fig = plots.plot_fundamental_diagram_all(densities, speeds)
        plots.show_fig(fig, figname=figname, html=True, write=True)
