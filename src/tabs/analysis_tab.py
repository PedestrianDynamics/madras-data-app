"""Analysis data, speed, density, flow, etc."""

import glob
import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedpy
import streamlit as st
from matplotlib.dates import DateFormatter
from pedpy import SpeedMethod
from scipy.ndimage import gaussian_filter

from ..classes.datafactory import load_file
from ..docs.docs import density_speed, flow
from ..helpers.speed_profile import compute_speed_profile
from ..helpers.utilities import (
    download,
    get_measurement_lines,
    is_running_locally,
    setup_walkable_area,
)
from ..plotting.drawing import drawing_canvas, get_measurement_area
from ..plotting.plots import (
    download_file,
    plot_fundamental_diagram_all,
    plot_fundamental_diagram_all_mpl,
    plot_time_series,
    plt_plot_time_series,
    show_fig,
)
from .LargeView_analysis import run_cctv_analysis

st_column: TypeAlias = st.delta_generator.DeltaGenerator


def calculate_or_load_classical_density(
    precalculated_density: Path,
    filename: str,
) -> pd.DataFrame:
    """Calculate classical density or load existing calculation."""
    if not Path(precalculated_density).exists():
        trajectory_data = load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        classic_density = pedpy.compute_classic_density(traj_data=trajectory_data, measurement_area=walkable_area)
        with open(precalculated_density, "wb") as f:
            pickle.dump(classic_density, f)
    else:
        logging.info("load precalculated density: %s", precalculated_density)
        with open(precalculated_density, "rb") as f:
            classic_density = pickle.load(f)

    return classic_density


def calculate_or_load_voronoi_diagrams(
    precalculated_voronoi_polygons: Path,
    filename: str,
) -> pd.DataFrame:
    """Calculate Voronoi diagrams or load existing calculation."""
    if not Path(precalculated_voronoi_polygons).exists():
        trajectory_data = load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        voronoi_polygons = pedpy.compute_individual_voronoi_polygons(traj_data=trajectory_data, walkable_area=walkable_area)

        with open(precalculated_voronoi_polygons, "wb") as f:
            pickle.dump(voronoi_polygons, f, pickle.HIGHEST_PROTOCOL)
    else:
        logging.info("load precalculated voronoi polygons: %s", precalculated_voronoi_polygons)
        with open(precalculated_voronoi_polygons, "rb") as f:
            voronoi_polygons = pickle.load(f)

    return voronoi_polygons


def calculate_or_load_voronoi_speed(
    precalculated_voronoi_speed: Path,
    intersecting: pd.DataFrame,
    individual_speed: pd.DataFrame,
    filename: str,
) -> pd.Series:
    """Calculate Voronoi speed or load existing calculation."""
    if not Path(precalculated_voronoi_speed).exists():
        trajectory_data = load_file(filename)
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
        logging.info("load precalculated voronoi speed: %s", precalculated_voronoi_speed)
        with open(precalculated_voronoi_speed, "rb") as f:
            voronoi_speed = pickle.load(f)

    return voronoi_speed


def calculate_or_load_voronoi_density(
    precalculated_voronoi_density: Path,
    voronoi_polygons: pd.DataFrame,
    filename: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Voronoi density or load existing calculation."""
    if not Path(precalculated_voronoi_density).exists():
        trajectory_data = load_file(filename)
        walkable_area = setup_walkable_area(trajectory_data)
        voronoi_density, intersecting = pedpy.compute_voronoi_density(
            individual_voronoi_data=voronoi_polygons,
            measurement_area=walkable_area,
        )

        with open(precalculated_voronoi_density, "wb") as f:
            pickle.dump((voronoi_density, intersecting), f, pickle.HIGHEST_PROTOCOL)
    else:
        logging.info("load precalculated voronoi density: %s", precalculated_voronoi_density)
        with open(precalculated_voronoi_density, "rb") as f:
            voronoi_density, intersecting = pickle.load(f)

    return voronoi_density, intersecting


def calculate_or_load_individual_speed(precalculated_speed: Path, filename: str, dv: Optional[int]) -> pd.DataFrame:
    """Calculate speed or load precalculated values if exist."""
    if not Path(precalculated_speed).exists():
        trajectory_data = load_file(filename)
        individual_speed = pedpy.compute_individual_speed(
            traj_data=trajectory_data,
            frame_step=dv,
            speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
        )
        with open(precalculated_speed, "wb") as f:
            pickle.dump(individual_speed, f)
    else:
        logging.info("load precalculated speed: %s", precalculated_speed)
        with open(precalculated_speed, "rb") as f:
            individual_speed = pickle.load(f)

    return individual_speed


def calculate_or_load_mean_speed(precalculated_speed: Path, filename: str, dv: Optional[int]) -> pd.DataFrame:
    """Calculate mean speed per frame if not already done."""
    speed = calculate_or_load_individual_speed(precalculated_speed, filename, dv)
    trajectory_data = load_file(filename)
    walkable_area = setup_walkable_area(trajectory_data)
    return pedpy.compute_mean_speed_per_frame(
        traj_data=trajectory_data,
        measurement_area=walkable_area,
        individual_speed=speed,
    )


def calculate_time_series(
    trajectory_data: pd.DataFrame,
    dv: Optional[int],
    walkable_area: pedpy.WalkableArea,
    selected_file: str,
) -> None:
    """Calculate speed and density."""
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    docs_expander = st.expander(":orange_book: Documentation (click to expand)", expanded=False)
    with docs_expander:
        density_speed()
    canvas, dpi, scale, img_height = drawing_canvas(trajectory_data, walkable_area)
    measurement_areas = get_measurement_area(trajectory_data, canvas, dpi, scale, img_height)
    individual_speed = pedpy.compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=dv,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    for mai, ma in enumerate(measurement_areas):
        mean_speed = pedpy.compute_mean_speed_per_frame(
            traj_data=trajectory_data,
            measurement_area=ma,
            individual_speed=individual_speed,
        )
        classic_density = pedpy.compute_classic_density(traj_data=trajectory_data, measurement_area=ma)
        fig = plot_time_series(classic_density, mean_speed, fps=30)
        show_fig(fig, html=True, write=False)
        # for plots
        pfig1, pfig2 = plt_plot_time_series(classic_density, mean_speed, fps=30)
        c1, c2 = st.columns(2)
        bounds = ma.polygon.bounds
        formatted_bounds = f"({bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f})"
        st.info(f"Measurement area coordinates: {formatted_bounds}, Area: {ma.area:.2} $m^2$.")
        # download figures
        base_name = Path(selected_file).stem
        speed_file_name = f"speed_{base_name}_ma_{mai}.pdf"
        density_file_name = f"density_{base_name}_ma_{mai}.pdf"
        figname1 = data_directory / density_file_name
        pfig1.savefig(figname1, bbox_inches="tight", pad_inches=0.1)
        download_file(figname1, c1, "density")
        figname2 = data_directory / speed_file_name
        pfig2.savefig(figname2, bbox_inches="tight", pad_inches=0.1)
        download_file(figname2, c2, "speed")


def calculate_fd_classical(dv: Optional[int]) -> None:
    """Calculate FD classical and write result in pdf file."""
    densities = {}
    speeds = {}
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    with st.status("Calculating...", expanded=True):
        progress_bar = st.progress(0)
        progress_status = st.empty()
        for i, filename in enumerate(st.session_state.files):
            basename = filename.split("/")[-1].split(".txt")[0]
            precalculated_density = Path(data_directory / f"density_{basename}.pkl")
            precalculated_speed = Path(data_directory / f"speed_{basename}_{dv}.pkl")
            speeds[basename] = calculate_or_load_mean_speed(precalculated_speed, filename, dv)
            densities[basename] = calculate_or_load_classical_density(precalculated_density, filename)
            progress = int(100 * (i + 1) / len(st.session_state.files))
            progress_bar.progress(progress)
            progress_status.text(f"> {progress}%")

    figname = data_directory / "fundamental_diagram_classical.pdf"
    fig = plot_fundamental_diagram_all_mpl(densities, speeds)
    fig.savefig(figname, bbox_inches="tight", pad_inches=0.1)
    st.pyplot(fig)
    st.info(figname)
    download_file(figname)
    # plots.show_fig(fig, figname=figname, html=True, write=True)


def calculate_fd_voronoi_local(dv: Optional[int]) -> None:
    """Calculate FD voronoi (locally)."""
    # todo this should be in datafactory or session_state
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    voronoi_polygons = {}
    voronoi_density = {}
    voronoi_speed = {}
    individual_speed = {}
    intersecting = {}
    voronoi_results = "voronoi_density_speed.pkl"  # todo should go to datafactory
    figname = data_directory / "fundamental_diagram_voronoi.pdf"
    st.sidebar.divider()
    msg = st.sidebar.empty()
    calculate = msg.button("Calculate", type="primary", help="Calculate fundamental diagram Voronoi")
    if not is_running_locally():
        st.warning(
            """
            This calculation is disabled when running in a deployed environment.\n
            You should run the app locally:
            """
        )
        st.code("streamlit run app.py")
        st.warning("See [README](https://github.com/PedestrianDynamics/madras-data-app?tab=readme-ov-file#local-execution-guide) for more information.")

    if is_running_locally() and calculate:
        with st.status("Calculating Voronoi FD ...", expanded=True):
            progress_bar = st.progress(0)
            progress_status = st.empty()
            start = time.time()
            for i, filename in enumerate(st.session_state.files):
                basename = filename.split("/")[-1].split(".txt")[0]
                # saved files ============
                precalculated_voronoi_polygons = data_directory / f"voronoi_polygons_{basename}.pkl"
                precalculated_speed = data_directory / f"speed_{basename}_{dv}.pkl"
                precalculated_voronoi_speed = data_directory / f"voronoi_speed_{basename}.pkl"
                precalculated_voronoi_density = data_directory / f"voronoi_density_{basename}.pkl"
                # saved files ============
                voronoi_polygons[basename] = calculate_or_load_voronoi_diagrams(precalculated_voronoi_polygons, filename)

                individual_speed[basename] = calculate_or_load_individual_speed(precalculated_speed, filename, dv)
                # todo save to files
                # trajectory_data = datafactory.load_file(filename)
                # walkable_area = setup_walkable_area(trajectory_data)

                voronoi_density[basename], intersecting[basename] = calculate_or_load_voronoi_density(
                    precalculated_voronoi_density,
                    voronoi_polygons[basename],
                    filename,
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

        fig = plot_fundamental_diagram_all(voronoi_density, voronoi_speed)

        show_fig(fig, figname=str(figname), html=True, write=True)
        end = time.time()
        st.info(f"Computation time: {end-start:.2f} seconds.")

    if calculate and Path(figname).exists():
        download_file(figname, msg)
    if not Path(figname).exists():
        st.warning(f"File {figname} does not exist yet! You should calculate it first")


def download_fd_voronoi() -> None:
    """Download preexisting voronoi calculation."""
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    voronoi_results = data_directory / "voronoi_density_speed.pkl"
    url = "https://go.fzj.de/voronoi-data"
    voronoi_density = {}
    voronoi_speed = {}
    figname = data_directory / "fundamental_diagram_voronoi.pdf"
    msg = st.empty()
    if not Path(voronoi_results).exists():
        msg.warning(f"{voronoi_results} does not exist yet!")
        with st.status("Downloading ...", expanded=True):
            download(url, voronoi_results)

    if Path(voronoi_results).exists():
        with open(voronoi_results, "rb") as f:
            voronoi_density, voronoi_speed = pickle.load(f)

        fig = plot_fundamental_diagram_all_mpl(voronoi_density, voronoi_speed)

        st.pyplot(fig)
        fig.savefig(figname, bbox_inches="tight", pad_inches=0.1)
        # download_file(figname)

        # fig = plots.plot_fundamental_diagram_all(voronoi_density, voronoi_speed)
        # plots.show_fig(fig, figname=figname, html=True, write=True)

    if Path(figname).exists():
        download_file(figname)
    else:
        st.warning(f"File {figname} does not exist yet! You should calculate it first")


def calculate_nt(
    trajectory_data: pedpy.TrajectoryData,
    selected_file: str,
) -> None:
    """Calculate N-T Diagram."""
    pl = st.sidebar.empty()
    distance_to_bounding = st.sidebar.number_input(
        "Distance to border [m]",
        value=0.5,
        min_value=0.1,
        max_value=20.0,
        step=1.0,
        placeholder="Type the ditance to boder.",
        help="Distance of the meansurement lines to the edges of the geometry.",
        format="%.2f",
    )
    directions = get_measurement_lines(trajectory_data, distance_to_bounding)
    docs_expander = st.expander(":orange_book: Documentation (click to expand)", expanded=False)
    with docs_expander:
        flow(directions)

    names = [direction.info.name for direction in directions]
    colors = [direction.info.color for direction in directions]
    selected_names = pl.multiselect("Measurement line", options=names, default=names)
    filename_without_extension = Path(selected_file).stem
    figname = f"NT_distance_{distance_to_bounding:.2f}_{filename_without_extension}"
    fig1, ax1 = plt.subplots()
    nt_stats = {}
    for i, (name, color) in enumerate(zip(selected_names, colors)):
        direction = directions[i]
        nt, _ = pedpy.compute_n_t(traj_data=trajectory_data, measurement_line=direction.line)
        figname += f"_{name}"
        pedpy.plot_nt(
            nt=nt,
            axes=ax1,
            color=color,
            title="",
            label=f"{name}",
        )
        nt_stats[name] = {
            "cumulative pedestrians": nt["cumulative_pedestrians"].iloc[-1],
            # "time / s": nt["time"].iloc[-1],
        }

    ax1.set_xlabel(r"$t\; /\; s$", fontsize=18)
    ax1.set_ylabel(r"$\# pedestrians$", fontsize=18)
    ax1.tick_params(axis="both", which="major", labelsize=14)  # For major ticks
    ax1.legend(loc="best")
    ax1.grid(alpha=0.4)

    c1, c2 = st.columns((0.6, 0.4))
    c1.pyplot(fig1)
    c2.write("**Total number of pedestrians over the observed period.**")
    c2.dataframe(pd.DataFrame(nt_stats))
    figname = Path(figname)
    figname = figname.with_name(figname.stem + ".pdf")
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    logging.info("Check existance %s: %s", data_directory, data_directory.exists())
    figname = data_directory / Path(figname)
    logging.info("Try to save %s", figname)
    fig1.savefig(figname, bbox_inches="tight", pad_inches=0.1)
    download_file(Path(figname))


def calculate_density_profile(
    trajectory_data: pedpy.TrajectoryData,
    walkable_area: pedpy.WalkableArea,
    selected_file: str,
) -> None:
    """Calculate density profiles based on different methods."""
    # chose_method = st.sidebar.radio(
    #     "Method",
    #     ["Gaussian", "Classic"],
    #     help="See [PedPy-documentation](https://pedpy.readthedocs.io/en/latest/user_guide.html#density-profiles).",
    # )
    with st.expander("Documentation"):
        st.write("This profile is using 'Gaussian density profile' from [PedPy]" + "(https://pedpy.readthedocs.io/en/latest/user_guide.html#density-profiles).")
    chose_method = "Gaussian"
    chose_method = str(chose_method)
    method = {
        "Classic": pedpy.DensityMethod.CLASSIC,
        "Gaussian": pedpy.DensityMethod.GAUSSIAN,
    }
    grid_size = st.sidebar.number_input(
        "Grid size",
        value=0.4,
        min_value=0.05,
        max_value=1.0,
        step=0.05,
        placeholder="Type the grid size",
        format="%.2f",
    )
    width = 1.0
    if chose_method == "Gaussian":
        width = float(
            st.sidebar.number_input(
                "fwhm",
                value=0.5,
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                help="full width at half maximum",
                format="%.2f",
            )
        )
    density_profile = pedpy.compute_density_profile(
        data=trajectory_data.data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        density_method=method[chose_method],
        gaussian_width=width,
    )
    rmax = float(
        st.sidebar.number_input(
            "Max density",
            value=3.0,
            min_value=2.0,
            max_value=5.0,
            step=0.1,
            placeholder="Max density for colorbar",
            format="%.2f",
        )
    )
    fig, ax = plt.subplots()
    pedpy.plot_profiles(
        walkable_area=walkable_area,
        profiles=density_profile,
        axes=ax,
        vmax=rmax,
        label="$\\rho$ / 1/$m^2$",
    )
    colorbar_ax = fig.axes[-1]
    colorbar_ax.set_ylabel("$\\rho$ / 1/$m^2$", size=18)
    colorbar_ax.tick_params(labelsize=18)
    # Remove tick marks but keep labels for x-axis
    ax.tick_params(axis="x", length=0)
    # Remove tick marks but keep labels for y-axis
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    base_filename = Path(selected_file).stem
    if chose_method == "Gaussian":
        figname = Path(f"density_profile_method_{chose_method}_width_{width}_grid_{grid_size}_{base_filename}.pdf")
    else:
        figname = Path(f"density_profile_method_{chose_method}_grid_{grid_size}_{base_filename}.pdf")
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    figname = data_directory / figname
    st.pyplot(fig)
    plt.tight_layout()
    fig.savefig(figname, bbox_inches="tight", pad_inches=0.1)
    download_file(figname)


def calculate_speed_profile(
    trajectory_data: pedpy.TrajectoryData,
    walkable_area: pedpy.WalkableArea,
    selected_file: str,
) -> None:
    """Calculate speed profile."""
    with st.expander("Documentation"):
        st.write("This profile is using 'Gaussian speed profile' from [PedPy]" + "(https://pedpy.readthedocs.io/en/latest/user_guide.html#speed-profiles).")
    grid_size = st.sidebar.number_input(
        "Grid size",
        value=0.4,
        min_value=0.05,
        max_value=1.0,
        step=0.05,
        placeholder="Type the grid size",
        format="%.2f",
    )
    # fil = str(
    #     st.sidebar.selectbox(
    #         "How to fil empty cells?",
    #         ["Nan", "0"],
    #     )
    # )
    fil = "Nan"
    # chose_method = st.sidebar.radio(
    #     "Chose method",
    #     ["Gaussian", "Classic"],
    #     help="See [PedPy-documentation](https://pedpy.readthedocs.io/en/latest/user_guide.html#density-profiles).",
    # )
    chose_method = "Gaussian"
    chose_method = str(chose_method)
    if chose_method == "Gaussian":
        fwhm = float(
            st.sidebar.number_input(
                "fwhm",
                value=0.5,
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                help="full width at half maximum",
                format="%.2f",
            )
        )
    if fil == "Nan":
        fil_empty = np.nan
    else:
        fil_empty = 0.0

    individual_speed = pedpy.compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=5,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    combined_data = individual_speed.merge(
        trajectory_data.data,
        on=[pedpy.column_identifier.ID_COL, pedpy.column_identifier.FRAME_COL],
    )
    if chose_method == "Gaussian":
        start = time.time()
        speed_profiles = compute_speed_profile(
            data=combined_data,
            walkable_area=walkable_area,
            grid_size=grid_size,
            fwhm=fwhm,
            speed_method=SpeedMethod.MEAN,
            fill_value=fil_empty,
        )
        end = time.time()
        logging.info("Compute time Gauss speed profiles: %s s", end - start)
    else:
        start = time.time()
        speed_profiles = pedpy.compute_speed_profile(
            data=combined_data,
            walkable_area=walkable_area,
            grid_size=grid_size,
            speed_method=pedpy.SpeedMethod.MEAN,
            fill_value=fil_empty,
        )
        end = time.time()
        logging.info("Compute time Classic: %s s", end - start)

    vmax = float(
        st.sidebar.number_input(
            "Max speed",
            value=0.8,
            min_value=0.5,
            max_value=3.0,
            step=0.1,
            placeholder="Max speed for colorbar",
            format="%.2f",
        )
    )

    fig, ax = plt.subplots()
    pedpy.plot_profiles(
        walkable_area=walkable_area,
        profiles=speed_profiles,
        vmax=vmax,
        axes=ax,
    )
    colorbar_ax = fig.axes[-1]
    colorbar_ax.set_ylabel(r"$v\,/\,m/s$", size=18)
    colorbar_ax.tick_params(labelsize=18)
    # Remove tick marks but keep labels for x-axis
    ax.tick_params(axis="x", length=0)
    # Remove tick marks but keep labels for x-axis
    ax.tick_params(axis="x", length=0)
    # Remove tick marks but keep labels for y-axis
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()

    base_filename = Path(selected_file).stem
    figname = Path(f"speed_profile_fil_{fil_empty}_grid_{grid_size}_{base_filename}.pdf")
    path = Path(__file__)
    data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
    figname = data_directory / figname
    st.pyplot(fig)
    fig.savefig(figname, bbox_inches="tight", pad_inches=0.1)
    download_file(Path(figname))


def ui_tab3_analysis() -> Tuple[str, Optional[int], st_column]:
    """Prepare ui elements."""
    _, c1, _ = st.columns((1, 1, 1))
    if st.sidebar.button(
        ":red_circle: Delete",
        help="Remove pre-loaded files",
    ):
        path: Path = Path(__file__)
        data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
        precalculated_files_pattern = str(data_directory / "*.pkl")
        files_to_delete = glob.glob(precalculated_files_pattern)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                st.toast(f"Deleted {file_path}", icon="✅")
            except Exception as e:
                st.error(f"Error deleting {file_path}: {e}")

    st.sidebar.divider()
    if is_running_locally():
        calculations = str(
            st.radio(
                "**Choose calculation**",
                [
                    "N-T",
                    "Outflow",
                    "Time series",
                    "FD_classical",
                    "FD_voronoi (load)",
                    "FD_voronoi (calculate)",
                    "Density profile",
                    "Speed profile",
                ],
                horizontal=True,
            )
        )
    else:
        calculations = str(
            st.radio(
                "**Choose calculation**",
                [
                    "N-T",
                    "Time series",
                    "Outflow",
                    "FD_classical",
                    "FD_voronoi (load)",
                    "Density profile",
                    "Speed profile",
                ],
                horizontal=True,
            )
        )
    exclude = ["N-T", "Outflow", "Density profile", "Speed profile", "FD_voronoi (load)"]
    if calculations in exclude:
        dv = None
    else:
        st.sidebar.write("**Speed parameter**")
        dv = int(
            st.sidebar.slider(
                r"$\Delta t$",
                1,
                100,
                10,
                5,
                help="To calculate the displacement over a specified number of frames.",
            )
        )

    return calculations, dv, c1


def prepare_data(selected_file: str) -> Tuple[pedpy.TrajectoryData, List[List[float]]]:
    """Load file, setup state_session and get walkable_area."""
    if selected_file != st.session_state.file_changed:
        trajectory_data = load_file(selected_file)
        st.session_state.trajectory_data = trajectory_data
        st.session_state.file_changed = selected_file

    trajectory_data = st.session_state.trajectory_data
    walkable_area = setup_walkable_area(trajectory_data)

    return trajectory_data, walkable_area


def read_and_plot_outflow(filename: str, sigma: float):
    """Read file, calculate flow and plot."""
    st.info(filename.stem)
    # setup and read file
    fig, ax = plt.subplots()
    df = pd.read_csv(filename)

    # Preprocess and calculate time difference
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df["date"] = pd.to_datetime("today").normalize()
    df["datetime"] = pd.to_datetime(df["date"].dt.date.astype(str) + " " + df["time"].dt.time.astype(str))
    df["time_difference"] = df["datetime"] - df["datetime"].iloc[0]
    df["seconds"] = df["time_difference"].dt.total_seconds()
    df["time_difference"] = df["seconds"].diff().fillna(0)
    # Calculate instant flow, handle the first row before dropping it
    df["instant_flow"] = df.people_passed / df.time_difference
    df = df.drop(df.index[0])
    df["instant_flow"] = df["instant_flow"].fillna(0)

    # Smooth the flow using a Gaussian filter
    df["gaussian_flow"] = gaussian_filter(df["instant_flow"], sigma=sigma)

    # plotting
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))  # Format time as hours:minutes
    ax.plot(df["time"], df["instant_flow"], "--.", lw=0.6, color="gray", label="Instant outflow")
    ax.plot(df["time"], df["gaussian_flow"], label="Smoothed outflow", color="black")
    legend = plt.legend(loc="upper right")
    plt.setp(legend.get_texts(), fontweight="bold")
    ax.set_ylim([0.0, 12.5])
    ax.tick_params(axis="x", which="major", labelsize=16)
    ax.tick_params(axis="y", which="major", labelsize=16)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Outflow of pedestrians", fontsize=16)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    # Save and return figure path
    st.pyplot(fig)
    figname = st.session_state.config.processed_directory / (Path(filename).stem + f"_sigma_{sigma}.pdf")
    fig.savefig(figname, bbox_inches="tight", pad_inches=0.1)
    return Path(figname)


def select_file() -> str:
    """Select a file from available options."""
    file_name_to_path = {path.split("/")[-1]: path for path in st.session_state.files}
    filename = str(st.selectbox(":open_file_folder: **Select a file**", file_name_to_path, key="tab3_filename"))
    selected_file = file_name_to_path[filename]
    st.session_state.selected_file = selected_file
    return selected_file


def handle_outflow(sigma: float):
    """Handle outflow calculation and plotting."""
    files = [
        st.session_state.config.flow_directory / "chenavard_2022_1210.csv",
        st.session_state.config.flow_directory / "constantine_2022_1210.csv",
    ]
    for _file in files:
        figname = read_and_plot_outflow(_file, sigma)
        download_file(figname)


def run_tab3() -> None:
    """Run the main logic in tab analysis."""
    calculations, dv, _ = ui_tab3_analysis()

    if not calculations.startswith(("FD", "Outflow")):
        selected_file = select_file()
        trajectory_data, walkable_area = prepare_data(selected_file)
    else:
        selected_file = st.session_state.selected_file
        trajectory_data, walkable_area = prepare_data(selected_file)

    # Dispatch the calculations to the appropriate handler
    calculation_handlers = {
        "Outflow": lambda: handle_outflow(
            float(st.sidebar.slider(r"$\sigma$", value=2.0, max_value=10.0, min_value=0.1, step=1.0, help="Standard deviation for Gaussian kernel used to smooth the curve."))
        ),
        "N-T": lambda: calculate_nt(trajectory_data, selected_file),
        "Density profile": lambda: handle_density_profile(trajectory_data, walkable_area, selected_file),
        "Speed profile": lambda: calculate_speed_profile(trajectory_data, walkable_area, selected_file),
        "Time series": lambda: calculate_time_series(trajectory_data, dv, walkable_area, selected_file),
        "FD_classical": lambda: calculate_fd_classical(dv),
        "FD_voronoi (calculate)": lambda: calculate_fd_voronoi_local(dv),
        "FD_voronoi (load)": download_fd_voronoi,
    }

    if calculations in calculation_handlers:
        calculation_handlers[calculations]()


def handle_density_profile(trajectory_data, walkable_area, selected_file):
    """Handle the logic for density profile calculation."""
    if "Topview" in selected_file:
        calculate_density_profile(trajectory_data, walkable_area, selected_file)
    else:
        run_cctv_analysis(selected_file)
