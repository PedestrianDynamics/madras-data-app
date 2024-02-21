"""Analysis data, speed, density, flow, etc."""

import glob
import os
import pickle
import time
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from plotly.graph_objs import Figure
import socket
import pedpy

# from pedpy import compute_density_profile, DensityMethod

import streamlit as st
from shapely import Polygon

import datafactory
import docs
import plots
import requests


voronoi_results = "voronoi_density_speed.pkl"
url = "https://go.fzj.de/voronoi-data"


def is_running_locally() -> bool:
    """Check if the Streamlit app is running locally."""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return (
        ip_address == "127.0.0.1"
        or ip_address.startswith("192.168.")
        or hostname == "localhost"
    )


def download(url: str, destination: Union[str, Path]) -> None:
    """
    Downloads a file from a specified URL and saves it to a given destination,
    """
    # Send a GET request
    response = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kbyte
    progress_bar = st.progress(0)
    progress_status = st.empty()
    written = 0

    with open(destination, "wb") as f:
        for data in response.iter_content(block_size):
            written += len(data)
            f.write(data)
            # Update progress bar
            progress = int(100 * written / total_size)
            progress_bar.progress(progress)
            progress_status.text(f"> {progress}%")

    progress_status.text("Download complete.")
    progress_bar.empty()  # clear  the progress bar after completion


def get_measurement_lines(trajectory_data):
    eps = 1.0
    min_x = trajectory_data.data["x"].min() + eps
    max_x = trajectory_data.data["x"].max() - eps
    min_y = trajectory_data.data["y"].min() + eps
    max_y = trajectory_data.data["y"].max() - eps
    measurement_lines = [
        pedpy.MeasurementLine([[min_x, min_y], [min_x, max_y]]),  # left
        pedpy.MeasurementLine([[min_x, max_y], [max_x, max_y]]),  # top
        pedpy.MeasurementLine([[max_x, max_y], [max_x, min_y]]),  # right
        pedpy.MeasurementLine([[max_x, min_y], [min_x, min_y]]),  # buttom
    ]

    return measurement_lines


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

    c0, c1, c2 = st.columns((1, 1, 1))
    if c1.button(
        "Delete files",
        help="To improve efficiency, certain density and speed values are pre-loaded rather than dynamically computed. By using this button, you have the option to remove these pre-loaded files, allowing for fresh calculations to be initiated from the beginning.",
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
            "Time series",
            "FD_classical",
            "FD_voronoi",
            "FD_voronoi (local)",
            "N-T",
            "Profiles",
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
    if selected_file != st.session_state.file_changed:
        with st.status(f"Loading {selected_file}"):
            trajectory_data = datafactory.load_file(selected_file)
            st.session_state.trajectory_data = trajectory_data
            st.session_state.file_changed = selected_file

    trajectory_data = st.session_state.trajectory_data
    walkable_area = setup_walkable_area(trajectory_data)

    if calculations == "Time series":
        docs_expander = st.expander("Documentation (click to expand)", expanded=False)
        with docs_expander:
            docs.density_speed()

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

    # we can run this function only locally. Streamlit server has some memory-limits
    if calculations == "FD_voronoi (local)":
        voronoi_polygons = {}
        voronoi_density = {}
        voronoi_speed = {}
        individual_speed = {}
        intersecting = {}
        figname = "fundamental_diagram_voronoi.pdf"
        msg = c1.empty()
        calculate = c1.button(
            "Calculate", type="primary", help="Calculate fundamental diagram Voronoi"
        )
        if not is_running_locally():
            st.warning(
                """
                This calculation is disabled when running in a deployed environment.\n
                You should run the app locally:
                """
            )
            st.code("streamlit run app.py")

        if is_running_locally() and calculate:
            with st.status("Calculating Voronoi FD ...", expanded=True):
                progress_bar = st.progress(0)
                progress_status = st.empty()
                start = time.time()
                for i, filename in enumerate(st.session_state.files):

                    basename = filename.split("/")[-1].split(".txt")[0]
                    # saved files ============
                    precalculated_voronoi_polygons = (
                        f"AppData/voronoi_polygons_{basename}.pkl"
                    )
                    precalculated_speed = f"AppData/speed_{basename}_{dv}.pkl"
                    precalculated_voronoi_speed = (
                        f"AppData/voronoi_speed_{basename}.pkl"
                    )
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

            with open(voronoi_results, "wb") as f:
                pickle.dump(
                    (voronoi_density, voronoi_speed), f, pickle.HIGHEST_PROTOCOL
                )

            fig = plots.plot_fundamental_diagram_all(voronoi_density, voronoi_speed)

            plots.show_fig(fig, figname=figname, html=True, write=True)
            end = time.time()
            st.info(f"Computation time: {end-start:.2f} seconds.")

        if Path(figname).exists():
            plots.download_file(figname, msg)
        else:
            st.warning(
                f"File {figname} does not exist yet! You should calculate it first"
            )

    if calculations == "FD_voronoi":
        voronoi_density = {}
        voronoi_speed = {}
        figname = "fundamental_diagram_voronoi.pdf"
        msg = st.empty()
        if not Path(voronoi_results).exists():
            msg.warning(f"{voronoi_results} does not exist yet!")
            with st.status("Downloading ...", expanded=True):
                download(url, voronoi_results)

        if Path(voronoi_results).exists():
            with open(voronoi_results, "rb") as f:
                voronoi_density, voronoi_speed = pickle.load(f)

            fig = plots.plot_fundamental_diagram_all(voronoi_density, voronoi_speed)
            plots.show_fig(fig, figname=figname, html=True, write=True)

        if Path(figname).exists():
            plots.download_file(figname, msg)
        else:
            st.warning(
                f"File {figname} does not exist yet! You should calculate it first"
            )

    if calculations == "N-T":
        measurement_lines = get_measurement_lines(trajectory_data)
        docs_expander = st.expander("Documentation (click to expand)", expanded=False)
        with docs_expander:
            docs.flow(measurement_lines)

        names = ["left", "top", "right", "buttom"]
        colors = ["red", "blue", "magenta", "green"]
        selected_measurement_lines = st.multiselect(
            "Measurement line", options=names, default=names
        )
        fig = Figure()
        for i, (name, color) in enumerate(zip(selected_measurement_lines, colors)):
            measurement_line = measurement_lines[i]
            nt, _ = pedpy.compute_n_t(
                traj_data=trajectory_data,
                measurement_line=measurement_line,
            )

            trace, _ = plots.plot_x_y(
                nt["cumulative_pedestrians"],
                nt["time"],
                xlabel="time",
                ylabel="#pedestrians",
                color=color,
                title=f"{name}",
            )
            fig.add_trace(trace)

        plots.show_fig(fig, figname="flow.pdf")

    if calculations == "Profiles":
        c1, c2, c3 = st.columns((1, 1, 1))
        chose_method = c3.radio(
            "Method",
            ["Gaussian", "Classic"],
            help="See [PedPy-documentation](https://pedpy.readthedocs.io/en/latest/user_guide.html#density-profiles).",
        )
        method = {
            "Classic": pedpy.DensityMethod.CLASSIC,
            "Gaussian": pedpy.DensityMethod.GAUSSIAN,
        }
        grid_size = c1.number_input(
            "Grid size",
            value=0.4,
            min_value=0.05,
            max_value=1.0,
            step=0.05,
            placeholder="Type the grid size",
            format="%.2f",
        )
        width = c2.number_input(
            "Gaussian width",
            value=0.5,
            min_value=0.2,
            max_value=1.0,
            step=0.1,
            placeholder="full width at half maximum for Gaussian.",
            format="%.2f",
        )

        gaussian_density_profile = pedpy.compute_density_profile(
            data=trajectory_data.data,
            walkable_area=walkable_area,
            grid_size=grid_size,
            density_method=method[chose_method],
            gaussian_width=width,
        )
        fig, ax0 = plt.subplots(nrows=1, ncols=1)
        pedpy.plot_profiles(
            walkable_area=walkable_area,
            profiles=gaussian_density_profile,
            axes=ax0,
            label="$\\rho$ / 1/$m^2$",
            title="Density",
        )
        fig.tight_layout(pad=2)
        st.pyplot(fig)
