"""Analysis data, speed, density, flow, etc."""

import glob
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pedpy
import streamlit as st
from plotly.graph_objs import Figure
import pandas as pd
import datafactory
import docs
import plots
import utilities

from typing import TypeAlias

st_column: TypeAlias = st.delta_generator.DeltaGenerator

voronoi_results = "voronoi_density_speed.pkl"
url = "https://go.fzj.de/voronoi-data"


def calculate_or_load_classical_density(
    precalculated_density,
    filename,
):
    """Calculate classical density or load existing calculation."""
    if not Path(precalculated_density).exists():
        trajectory_data = datafactory.load_file(filename)
        walkable_area = utilities.setup_walkable_area(trajectory_data)
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
        walkable_area = utilities.setup_walkable_area(trajectory_data)
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
        walkable_area = utilities.setup_walkable_area(trajectory_data)
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
        walkable_area = utilities.setup_walkable_area(trajectory_data)
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
    """Calculate speed or load precalculated values if exist."""
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


def calculate_time_series(
    trajectory_data: pd.DataFrame,
    dv: int,
    walkable_area: pedpy.WalkableArea,
    selected_file: str,
) -> None:
    """Calculate speed and density."""
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
    plots.download_file(figname)


def calculate_fd_classical(dv) -> None:
    """Calculate FD classical and write result in pdf file."""
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


def calculate_fd_voronoi_local(c1: st_column, dv: int) -> None:
    """Calculate FD voronoi (locally)."""
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
    if not utilities.is_running_locally():
        st.warning(
            """
            This calculation is disabled when running in a deployed environment.\n
            You should run the app locally:
            """
        )
        st.code("streamlit run app.py")
        st.warning(
            "See [README](https://github.com/PedestrianDynamics/madras-data-app?tab=readme-ov-file#local-execution-guide) for more information."
        )

    if utilities.is_running_locally() and calculate:
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
                # trajectory_data = datafactory.load_file(filename)
                # walkable_area = utilities.setup_walkable_area(trajectory_data)

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
            pickle.dump((voronoi_density, voronoi_speed), f, pickle.HIGHEST_PROTOCOL)

        fig = plots.plot_fundamental_diagram_all(voronoi_density, voronoi_speed)

        plots.show_fig(fig, figname=figname, html=True, write=True)
        end = time.time()
        st.info(f"Computation time: {end-start:.2f} seconds.")

    if Path(figname).exists():
        plots.download_file(figname, msg)
    else:
        st.warning(f"File {figname} does not exist yet! You should calculate it first")


def download_fd_voronoi():
    """Download preexisting voronoi calculation."""
    voronoi_density = {}
    voronoi_speed = {}
    figname = "fundamental_diagram_voronoi.pdf"
    msg = st.empty()
    if not Path(voronoi_results).exists():
        msg.warning(f"{voronoi_results} does not exist yet!")
        with st.status("Downloading ...", expanded=True):
            utilities.download(url, voronoi_results)

    if Path(voronoi_results).exists():
        with open(voronoi_results, "rb") as f:
            voronoi_density, voronoi_speed = pickle.load(f)

        fig = plots.plot_fundamental_diagram_all(voronoi_density, voronoi_speed)
        plots.show_fig(fig, figname=figname, html=True, write=True)

    if Path(figname).exists():
        plots.download_file(figname)
    else:
        st.warning(f"File {figname} does not exist yet! You should calculate it first")


def calculate_nt(
    trajectory_data,
    selected_file,
):
    """Calculate N-T Diagram."""
    measurement_lines = utilities.get_measurement_lines(trajectory_data)
    docs_expander = st.expander("Documentation (click to expand)", expanded=False)
    with docs_expander:
        docs.flow(measurement_lines)

    names = ["left", "top", "right", "buttom"]
    colors = ["red", "blue", "magenta", "green"]
    selected_measurement_lines = st.multiselect(
        "Measurement line", options=names, default=names
    )
    fig = Figure()
    figname = "NT=" + selected_file.split("/")[-1].split(".txt")[0]
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
        figname += f"_{name}"

    figname += ".pdf"
    plots.show_fig(fig, figname=figname)
    plots.download_file(figname)
    return figname


def calculate_profiles(
    trajectory_data: pedpy.TrajectoryData,
    walkable_area: pedpy.WalkableArea,
    selected_file: str,
) -> None:
    """Calculate density profiles based on different methods."""
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
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    pedpy.plot_profiles(
        walkable_area=walkable_area,
        profiles=gaussian_density_profile,
        axes=ax0,
        label="$\\rho$ / 1/$m^2$",
        title="Density",
    )
    figname = (
        "density_profile_"
        + selected_file.split("/")[-1].split(".txt")[0]
        + str(chose_method)
        + ".pdf"
    )
    st.pyplot(fig)
    fig.savefig(figname)
    plots.download_file(figname)
    return figname


def ui_tab3_analysis():
    """Prepare ui elements."""
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

    return calculations, dv, c1


def prepare_data(selected_file):
    """Load file, setup state_session and get walkable_area."""
    if selected_file != st.session_state.file_changed:
        trajectory_data = datafactory.load_file(selected_file)
        st.session_state.trajectory_data = trajectory_data
        st.session_state.file_changed = selected_file

    trajectory_data = st.session_state.trajectory_data
    walkable_area = utilities.setup_walkable_area(trajectory_data)

    return trajectory_data, walkable_area


def run_tab3(selected_file):
    """Run the main logic in tab analysis."""
    calculations, dv, c1 = ui_tab3_analysis()
    trajectory_data, walkable_area = prepare_data(selected_file)
    if calculations == "Time series":
        calculate_time_series(trajectory_data, dv, walkable_area, selected_file)
    if calculations == "FD_classical":
        calculate_fd_classical(dv)
    if calculations == "FD_voronoi (local)":
        calculate_fd_voronoi_local(c1, dv)
    if calculations == "FD_voronoi":
        download_fd_voronoi()
    if calculations == "N-T":
        calculate_nt(
            trajectory_data,
            selected_file,
        )
    if calculations == "Profiles":
        calculate_profiles(
            trajectory_data,
            walkable_area,
            selected_file,
        )
