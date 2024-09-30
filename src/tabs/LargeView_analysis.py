"""This script computes the density and velocity fields from the trajectories of pedestrians."""

import logging
import pickle
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from numba import njit
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from numpy.typing import NDArray

plt.rcParams["font.family"] = "STIXGeneral"


@dataclass
class Parameters:
    """
    Parameters dataclass for CCTV_analysis.

    Attributes:
        FOLDER_TRAJ (Path): Path to the folder containing trajectory data.
        FOLDER_SAVE (Path): Path to the folder where the analysis results will be saved.
        SELECTED_NAME (str): Name of the selected data.
        START_TIME (float): Start time of the analysis.
        DURATION (float): Duration of the analysis.
        X_MIN (int): Minimum value of the x-coordinate.
        X_MAX (int): Maximum value of the x-coordinate.
        Y_MIN (int): Minimum value of the y-coordinate.
        Y_MAX (int): Maximum value of the y-coordinate.
        DT (float): Time step for the analysis.
        XI (float): Parameter for the analysis.
        R_C (float): Parameter for the analysis.
        R_CG (float): Parameter for the analysis.
        CUTOFF (float): Parameter for the analysis.
        DELTA_T (float): Parameter for the analysis.
        QUIVER_STEP (int): Step size for the quiver plot.
        QUIVER_SCALE (float): Scale factor for the quiver plot.
        CUM_TIME (float, optional): Cumulative time for the analysis. Defaults to 0.0.
        NB_CG_X (int, optional): Number of cells in the x-direction for the analysis. Defaults to 0.
        NB_CG_Y (int, optional): Number of cells in the y-direction for the analysis. Defaults to 0.
        DELTA (float, optional): Parameter for the analysis. Defaults to 0.0.
        COLORBAR_MIN (float, optional): Minimum value for the colorbar. Defaults to 0.0.
        COLORBAR_MAX (float, optional): Maximum value for the colorbar. Defaults to 4.0.
        COLORBAR_MAX_V (float, optional): Maximum value for the colorbar (vorticity). Defaults to 0.25.
    """

    FOLDER_TRAJ: Path
    FOLDER_SAVE: Path
    SELECTED_NAME: str
    START_TIME: float
    DURATION: float
    X_MIN: int
    X_MAX: int
    Y_MIN: int
    Y_MAX: int
    DT: float
    XI: float
    R_C: float
    R_CG: float
    CUTOFF: float
    DELTA_T: float
    QUIVER_STEP: int
    QUIVER_SCALE: float
    CUM_TIME: float = 0.0
    NB_CG_X: int = 0
    NB_CG_Y: int = 0
    DELTA: float = 0.0
    COLORBAR_MIN: float = 0.0
    COLORBAR_MAX: float = 4.0
    COLORBAR_MAX_V: float = 0.25


@njit  # type: ignore
def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate the distance between two points.

    Parameters:
    - pos1 (Tuple[float, float]): The coordinates of the first point.
    - pos2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    - float: The distance between the two points.
    """
    return float(np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2))


@njit  # type: ignore
def gaussian_kernel_scalar(r: float, r_c: float, xi: float) -> float:
    """
    Calculate the value of a Gaussian kernel for a given distance.

    Parameters:
        r (float): The distance at which to evaluate the kernel.
        r_c (float): The cutoff distance. If the distance is greater than this value, the kernel value is 0.
        xi (float): The width parameter of the Gaussian kernel.

    Returns:
        float: The value of the Gaussian kernel at the given distance.
    """
    if r > r_c:
        return 0.0
    return float(
        np.exp(-0.5 * (r / xi) ** 2)
    )  # Prefactor is omitted because it cancels out


@njit  # type: ignore
def get_r(
    i_line: int, j_column: int, r_cg: float, x_min: float, y_min: float
) -> Tuple[float, float]:
    """
    Calculate the coordinates (x, y) of a point in a grid based on its indices and grid parameters.

    Parameters:
    - i_line (int): The index of the line in the grid.
    - j_column (int): The index of the column in the grid.
    - r_cg (float): The size of each grid cell.
    - x_min (float): The minimum x-coordinate of the grid.
    - y_min (float): The minimum y-coordinate of the grid.

    Returns:
    - Tuple[float, float]: The coordinates (x, y) of the point in the grid.
    """
    return (i_line * r_cg + 0.5 * r_cg + x_min, j_column * r_cg + 0.5 * r_cg + y_min)


@njit  # type: ignore
def get_cell(
    r: Tuple[float, float], x_min: float, y_min, r_cg: float
) -> Tuple[int, int]:
    """
    Return the cell indices corresponding to the given coordinates.

    Parameters:
    r (tuple): The coordinates (x, y) of the point.
    x_min (float): The minimum x-coordinate of the grid.
    y_min: The minimum y-coordinate of the grid.
    r_cg (float): The size of each cell in the grid.

    Returns:
    Tuple[int, int]: The cell indices (i_line, j_column) corresponding to the given coordinates.
    """
    i_line = int(floor((r[0] - x_min) / r_cg))
    j_column = int(floor((r[1] - y_min) / r_cg))
    return (i_line, j_column)


# This function performs forward and backward filtering to eliminate phase distortion,
# ensuring that the output signal has no phase shift relative to the input.
# The padlen parameter is set to int(1 / delta_t) + 1 to determine the number of samples for edge padding,
# which helps in reducing boundary effects in the filtering process.
def butter_lowpass_filter(
    pd_series: pd.Series, delta_t: float, order: int, cutoff: float
) -> NDArray[np.float64]:
    """
    Apply a Butterworth lowpass filter to a pandas Series.

    Parameters:
        pd_series (pd.Series): The input pandas Series to be filtered.
        delta_t (float): The time interval between samples.
        order (int): The order of the filter.
        cutoff (float): The cutoff frequency of the filter.

    Returns:
        NDArray[np.float64]: The filtered output as a NumPy array.
    """
    nyquist_freq = 0.5 / delta_t  # Nyquist Frequency
    normal_cutoff = cutoff / nyquist_freq  # Normalized cutoff frequency
    # Get the filter coefficients
    b, a = butter(
        order, normal_cutoff, btype="low", analog=False
    )  # Generate the filter coefficients
    filtered_data: NDArray[np.float64] = filtfilt(
        b, a, pd_series, padlen=int(1 / delta_t) + 1
    )  # Apply the filter.

    return filtered_data


def read_and_process_file(filepath: Path) -> pd.DataFrame:
    """
    Read a CSV file from the given filepath and processes it into a pandas DataFrame.

    Parameters:
        filepath (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame containing the data from the CSV file.
    """
    dataframe = pd.read_csv(filepath, sep=" ", comment="#", header=None)
    dataframe.columns = ["id", "frame", "x_m", "y_m", "z_m", "t_s", "x_RGF", "y_RGF"]
    return dataframe


def create_save_folder(pa: Parameters) -> Path:
    """
    Create and returns a save folder path based on the given Parameters object.

    Args:
        pa (Parameters): The Parameters object containing the necessary information.

    Returns:
        Path: The path of the created save folder.

    """
    save_folder = (
        pa.FOLDER_SAVE
        / f"{pa.SELECTED_NAME}_start={pa.START_TIME:.1f}_dur={pa.DURATION:.1f}_dt={pa.DELTA_T:.1f}_xi={pa.XI:.1f}_rcg={pa.R_CG:.1f}"
    )
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder


def process_trajectories(
    all_datas: pd.DataFrame, pa: Parameters
) -> Dict[str, pd.DataFrame]:
    """
    Process trajectories and return a dictionary of processed dataframes.

    Args:
        all_datas (pd.DataFrame): The dataframe containing all trajectory data.
        pa (Parameters): The parameters object containing necessary parameters.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where the keys are trajectory numbers and the values are processed dataframes.

    Raises:
        ValueError: If filtering fails for a trajectory.

    Notes:
        - The function applies a Butterworth low-pass filter to the x and y trajectory data.
        - It computes the start and end time of each trajectory for interpolation.
        - It interpolates the filtered data using alpha values.
        - It filters the data to the desired time interval.
        - It computes cumulative time, min and max coordinates, and velocities for each trajectory.
        - It removes NaN values from the velocities.
        - It returns a dictionary of processed dataframes, where the keys are trajectory numbers.
    """
    # Initialize the dictionary and list
    dict_all_datas = {}
    list_files_skipped = []

    # Reset the time to start from 0
    all_datas["t_s"] = all_datas["t_s"] - all_datas["t_s"].min()

    # Filter trajectories
    unique_ids = all_datas["id"].unique()
    # print(f"Processing {len(unique_ids)} trajectories...")
    for _, traj_nb in enumerate(unique_ids):
        traj_data = all_datas[all_datas["id"] == traj_nb]

        # Skip empty trajectories
        if traj_data.empty or traj_data["t_s"].isnull().all():
            list_files_skipped.append(traj_nb)
            continue

        # Initialize logger
        logging.basicConfig(level=logging.INFO)

        try:
            # Apply Butterworth low-pass filter to x and y trajectory data
            X_bw = butter_lowpass_filter(
                traj_data["x_m"].values, pa.DELTA_T, 2, pa.CUTOFF
            )
            Y_bw = butter_lowpass_filter(
                traj_data["y_m"].values, pa.DELTA_T, 2, pa.CUTOFF
            )
        except ValueError as e:
            # Log the exception with the trajectory number
            logging.error("Filtering failed for trajectory %s: %s", traj_nb, e)

            # Append the trajectory number to the list of skipped files
            list_files_skipped.append(traj_nb)

            # Use unfiltered data as fallback
            X_bw = traj_data["x_m"].values
            Y_bw = traj_data["y_m"].values

        # Compute the start and end time of the trajectory for interpolation
        starting_time_traj = traj_data["t_s"].min(skipna=True)
        end_time = traj_data["t_s"].max(skipna=True)
        t_s_values = traj_data["t_s"].values

        # Compute the alpha values for interpolation
        alpha = np.maximum(
            np.exp(
                np.clip(-4.0 * pa.CUTOFF * (t_s_values - starting_time_traj), -700, 700)
            ),
            np.exp(np.clip(-4.0 * pa.CUTOFF * (end_time - t_s_values), -700, 700)),
        )

        # Interpolate the filtered data
        traj_data.loc[:, "x_m"] = (1.0 - alpha) * X_bw + alpha * traj_data["x_m"].values
        traj_data.loc[:, "y_m"] = (1.0 - alpha) * Y_bw + alpha * traj_data["y_m"].values

        # Filter the data to the desired time interval
        traj_data = traj_data[
            (traj_data["t_s"] >= pa.START_TIME)
            & (traj_data["t_s"] < pa.START_TIME + pa.DURATION + 2.0 * pa.DT)
        ]

        # If the trajectory is empty, skip it
        if traj_data.empty:
            list_files_skipped.append(traj_nb)
            continue

        # Compute the cumulative time
        pa.CUM_TIME += traj_data["t_s"].max(skipna=True) - traj_data["t_s"].min(
            skipna=True
        )

        # Update min and max coordinates
        pa.X_MIN = min(pa.X_MIN, traj_data["x_m"].min(skipna=True))
        pa.X_MAX = max(pa.X_MAX, traj_data["x_m"].max(skipna=True))
        pa.Y_MIN = min(pa.Y_MIN, traj_data["y_m"].min(skipna=True))
        pa.Y_MAX = max(pa.Y_MAX, traj_data["y_m"].max(skipna=True))

        # Compute the velocities
        t_s_diff = np.diff(traj_data["t_s"].values)
        x_m_diff = np.diff(traj_data["x_m"].values)
        y_m_diff = np.diff(traj_data["y_m"].values)

        # Check if the time differences are valid (shannon theorem)
        valid_mask = t_s_diff <= 2.0 * pa.DT
        if not np.all(valid_mask):
            list_files_skipped.append(traj_nb)
            continue

        # Compute the velocities
        traj_data["vx"] = np.concatenate(([np.nan], x_m_diff / t_s_diff))
        traj_data["vy"] = np.concatenate(([np.nan], y_m_diff / t_s_diff))

        # Remove NaN values
        traj_data.dropna(subset=["vx", "vy"], inplace=True)

        # Add the trajectory to the dictionary
        dict_all_datas[traj_nb] = traj_data

    # print(f"{len(list_files_skipped)} trajs have been skipped: {list_files_skipped}")

    return dict_all_datas


def save_data(data_to_save: Any, folder_save: Path, title: str) -> None:
    """
    Save the given data to a file.

    Args:
        data_to_save: The data to be saved.
        folder_save: The folder where the data will be saved.
        title: The title of the file.

    Returns:
        None
    """
    with open(folder_save / title, "wb") as mydumpfile:
        pickle.dump(data_to_save, mydumpfile)


def load_data(folder_save: Path, title: str) -> Any:
    """
    Load data from a file.

    Args:
        folder_save (Path): The folder path where the file is saved.
        title (str): The name of the file.

    Returns:
        Any

    """
    with open(folder_save / title, "rb") as myloadfile:
        return pickle.load(myloadfile)


def calculate_grid_dimensions(pa: Parameters) -> None:
    """
    Calculate the grid dimensions based on the given parameters.

    Args:
        pa (Parameters): The parameters object containing the necessary information.

    Returns:
        None
    """
    # TODO: Oscar: Check conversion of the float Delta to int.
    pa.NB_CG_X = int((pa.X_MAX - pa.X_MIN) / pa.R_CG) + int(pa.DELTA) + 2
    pa.NB_CG_Y = int((pa.Y_MAX - pa.Y_MIN) / pa.R_CG) + int(pa.DELTA) + 2


def initialize_dict(nb_cg_x: int, nb_cg_y: int) -> Dict[str, NDArray[np.float64]]:
    """
    Initialize a dictionary of density and velocity fields.

    Parameters:
        nb_cg_x (int): Number of grid points along the x-axis.
        nb_cg_y (int): Number of grid points along the y-axis.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the initialized density and velocity fields.
    """
    # List of field names
    field_names = ["X", "Y", "rho", "vxs", "vys", "vxs2", "vys2", "var_vs"]

    return {
        name: np.zeros((nb_cg_x, nb_cg_y), dtype=np.float64) for name in field_names
    }


def compute_fields(
    all_trajs: Dict[str, Any],
    df_observables: Dict[str, NDArray[np.float64]],
    pa: Parameters,
    my_progress_bar: Any,
    status_text: Any,
) -> None:
    """
    Compute the density field based on the given trajectories and parameters.

    Args:
        all_trajs (dict): A dictionary containing the trajectories.
        df_observables (Dict[str, NDArray[np.float64]]): A dictionary to store the computed observables.
        pa (Parameters): An object containing the parameters.
    """
    nb_traj = len(all_trajs)
    # Iterate over all trajectories
    for i_traj, traj in tqdm(enumerate(all_trajs.values()), desc="Processing grid..."):
        traj = traj.loc[
            (traj["t_s"] >= pa.START_TIME)
            & (traj["t_s"] < pa.START_TIME + pa.DURATION + 2.0 * pa.DT)
        ]
        if traj.shape[0] == 0:
            continue
        # Iterate over all rows in the trajectory (ie all time steps)
        for row in traj.itertuples():
            pos_ped = (row.x_m, row.y_m)  # position of the pedestrian in meters
            i_rel, j_rel = get_cell(pos_ped, pa.X_MIN, pa.Y_MIN, pa.R_CG)
            # Iterate over the cells in the grid
            for i in range(i_rel - pa.DELTA, i_rel + pa.DELTA + 1):
                for j in range(j_rel - pa.DELTA, j_rel + pa.DELTA + 1):
                    if (
                        i < 0 or i >= pa.NB_CG_X or j < 0 or j >= pa.NB_CG_Y
                    ):  # Check if pedestrian is outside the grid
                        continue

                    # Compute the position of the grid cell
                    pos_grid_cell = get_r(i, j, pa.R_CG, pa.X_MIN, pa.Y_MIN)

                    # Compute the Gaussian kernel
                    phi_r = gaussian_kernel_scalar(
                        calculate_distance(pos_grid_cell, pos_ped), pa.R_C, pa.XI
                    )

                    # Update the fields
                    df_observables["X"][i, j] = pos_grid_cell[0]
                    df_observables["Y"][i, j] = pos_grid_cell[1]
                    df_observables["rho"][i, j] += phi_r
                    df_observables["vxs"][i, j] += phi_r * row.vx
                    df_observables["vys"][i, j] += phi_r * row.vy
                    df_observables["vxs2"][i, j] += phi_r * row.vx**2
                    df_observables["vys2"][i, j] += phi_r * row.vy**2

        # Update progress bar
        percent_complete = int((i_traj / nb_traj) * 100)
        my_progress_bar.progress(percent_complete)
        # Update status text
        progress_text = "Operation in progress. Please wait. ⏳"
        status_text.text(f"{progress_text} {percent_complete}%")

    # Clear status text and progress bar after completion
    status_text.text("Operation complete! ⌛")
    my_progress_bar.empty()

    # Normalize and calculate standard deviation
    nonzero_mask = df_observables["rho"] > 1e-10
    df_observables["vxs"][nonzero_mask] /= df_observables["rho"][nonzero_mask]
    df_observables["vys"][nonzero_mask] /= df_observables["rho"][nonzero_mask]
    df_observables["vxs2"][nonzero_mask] /= df_observables["rho"][nonzero_mask]
    df_observables["vys2"][nonzero_mask] /= df_observables["rho"][nonzero_mask]
    df_observables["vxs2"][nonzero_mask] -= df_observables["vxs"][nonzero_mask] ** 2
    df_observables["vys2"][nonzero_mask] -= df_observables["vys"][nonzero_mask] ** 2
    variance_sum = np.copy(df_observables["vxs2"] + df_observables["vys2"])
    df_observables["var_vs"] = np.maximum(variance_sum, 0.0)

    # Normalize the density field (no need to normalize the variance field because we already devide by the density)
    df_observables["rho"] = normalize_density(np.copy(df_observables["rho"]), pa)


def normalize_density(
    density: NDArray[np.float64], pa: Parameters
) -> NDArray[np.float64]:
    """
    Normalize the density array based on the given parameters.

    Parameters:
    - density (NDArray[np.float64]): The density array to be normalized.
    - pa (Parameters): The parameters object containing the necessary values.

    Returns:
    - NDArray[np.float64]: The normalized density array.
    """
    N_nonrenormalised = pa.R_CG**2 * np.sum(density)
    density *= float(pa.CUM_TIME / pa.DURATION) / float(N_nonrenormalised)
    return density


def update_figure(
    df_observables: Dict[str, NDArray[np.float64]],
    pa: Parameters,
    plot_density: bool,
    zsmooth: bool = False,
) -> go.Figure:
    """
    Update the figure with the given observables and parameters.

    Args:
        df_observables (Dict[str, NDArray[np.float64]]): A dictionary containing the observables data.
            The keys are "X", "Y", "rho", "vxs", and "vys", and the values are numpy arrays.
        pa (Parameters): The parameters object containing the required parameters.

    Returns:
        go.Figure: The updated figure with the heatmap and quiver plot.

    """
    # Create the X and Y axes
    X_axis = np.linspace(
        pa.X_MIN,
        pa.X_MIN + len(df_observables["rho"][0]) * pa.R_CG,
        len(df_observables["rho"][0]),
    )
    Y_axis = np.linspace(
        pa.Y_MIN,
        pa.Y_MIN + len(df_observables["rho"]) * pa.R_CG,
        len(df_observables["rho"]),
    )

    # Create the heatmap
    if plot_density:
        hovertext = np.array(
            [
                [
                    f"x: {X_axis[i]:.2f} m<br>y: {Y_axis[j]:.2f} m<br>Density: {df_observables['rho'][j,i]:.2f} ped/m²"
                    for i in range(df_observables["rho"].shape[1])
                ]
                for j in range(df_observables["rho"].shape[0])
            ]
        )

        # Create heatmap with colorbar limits
        heatmap = go.Heatmap(
            z=df_observables["rho"].T,
            x=X_axis,
            y=Y_axis,
            colorscale="YlOrRd",
            zmin=pa.COLORBAR_MIN,  # Set the minimum value of the colorbar
            zmax=pa.COLORBAR_MAX,  # Set the maximum value of the colorbar
            zsmooth=zsmooth,  # Apply smoothing with 'best' interpolation
            colorbar={"title": "Density [ped/m²]", "titleside": "right"},
            hoverinfo="text",  # Set hoverinfo to display just the hovertext
            hovertext=hovertext,  # Include hovertext
        )

    else:
        hovertext = np.array(
            [
                [
                    f"x: {X_axis[i]:.2f} m<br>y: {Y_axis[j]:.2f} m<br>Var_v: {(df_observables['var_vs'][j,i]):.2f} m²/s²"
                    for i in range(df_observables["var_vs"].shape[1])
                ]
                for j in range(df_observables["var_vs"].shape[0])
            ]
        )

        # Create heatmap with colorbar limits
        heatmap = go.Heatmap(
            z=df_observables["var_vs"].T,
            x=X_axis,
            y=Y_axis,
            colorscale="Blues",
            zmin=pa.COLORBAR_MIN,  # Set the minimum value of the colorbar
            zmax=pa.COLORBAR_MAX_V,  # Set the maximum value of the colorbar
            zsmooth=zsmooth,  # Apply smoothing with 'best' interpolation
            colorbar={"title": "Var_v [m²/s²]", "titleside": "right"},
            hoverinfo="text",  # Set hoverinfo to display just the hovertext
            hovertext=hovertext,  # Include hovertext
        )

    # Create quiver plot
    quiver = ff.create_quiver(
        df_observables["X"][:: pa.QUIVER_STEP, :: pa.QUIVER_STEP],
        df_observables["Y"][:: pa.QUIVER_STEP, :: pa.QUIVER_STEP],
        df_observables["vxs"][:: pa.QUIVER_STEP, :: pa.QUIVER_STEP],
        df_observables["vys"][:: pa.QUIVER_STEP, :: pa.QUIVER_STEP],
        scale=pa.QUIVER_SCALE,
        hoverinfo="skip",
        arrow_scale=0.2,
        line={"width": 1, "color": "black"},
    )

    # Combine both plots in a single figure
    fig = go.Figure(data=list(quiver.data) + [heatmap])

    # Calculate the end point of the arrow annotation
    arrow_velocity = 1  # Velocity represented by the annotation arrow (1 m/s)
    arrow_start_x = 8.0
    arrow_start_y = 3.0
    arrow_end_x = arrow_start_x + (
        arrow_velocity * pa.QUIVER_SCALE
    )  # Adjusted to represent 1 m/s
    arrow_end_y = arrow_start_y

    # Add arrow annotation
    fig.add_annotation(
        x=arrow_end_x,
        y=arrow_end_y,
        xref="x",
        yref="y",
        ax=arrow_start_x,
        ay=arrow_start_y,
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=5,  # Increase the arrow width to make it thicker
        arrowcolor="#009999",
        bgcolor="rgba(0, 0, 0, 0)",
    )

    # Add text annotation for the arrow
    fig.add_annotation(
        x=9.5,
        y=3.8,
        text="<b>1 m/s</b>",
        showarrow=False,
        font={"color": "#009999", "size": 25},
    )

    # Correct layout update with proper domain
    fig.update_layout(
        font={"size": 20},
        title={
            "text": (
                "Velocity field for density heatmap"
                if plot_density
                else "Velocity field for variance velocity heatmap"
            ),
            "font_size": 20,
        },
        xaxis={
            "title": {"text": "x [m]", "font_size": 20},
            "scaleanchor": "y",
            "scaleratio": 1,
            "range": [0.0, 12.0],
            "tickfont_size": 20,
        },
        yaxis={
            "title": {"text": "y [m]", "font_size": 20},
            "scaleanchor": "x",
            "scaleratio": 1,
            "range": [2.2, 23],
            "tickfont_size": 20,
        },
        width=650,
        height=900,
    )

    return fig


def main(selected_file: str) -> None:
    """
    Run main function for the CCTV_analysis module.

    Args:
        selected_file (str): The path to the selected file.

    Returns:
        None
    """
    # Sidebar for settings
    st.sidebar.title("Settings")

    # Add a slider to the sidebar
    slider_value_R_CG = st.sidebar.slider(
        "Select a value for the grid cell size",
        min_value=0.2,
        max_value=1.5,
        value=1.1,
        step=0.1,  # Default value
    )

    path = Path(__file__)

    pa = Parameters(
        FOLDER_TRAJ=Path(
            path.parent.parent.parent.absolute() / "data" / "trajectories"
        ),
        FOLDER_SAVE=Path(path.parent.parent.parent.absolute() / "data" / "pickle"),
        SELECTED_NAME=Path(selected_file).stem,
        START_TIME=0.0,
        DURATION=10.0,
        X_MIN=500,
        X_MAX=-1,
        Y_MIN=500,
        Y_MAX=-1,
        DT=1.0,
        XI=0.75,
        R_C=5.0 * 0.75,
        R_CG=1.1,
        CUM_TIME=0.0,
        CUTOFF=0.25,
        DELTA_T=0.1,
        QUIVER_STEP=1,
        QUIVER_SCALE=3.0,
    )
    pa.R_CG = slider_value_R_CG
    pa.DELTA = (
        int(ceil(pa.R_C / pa.R_CG)) + 1
    )  # Number of cells to consider around the cell containing the point
    folder_save = create_save_folder(pa)

    if (
        not Path(folder_save / "traj_data.pkl").exists()
        or not Path(folder_save / "parameters.pkl").exists()
        or not Path(folder_save / "dictionnary_observables.pkl").exists()
    ):
        # PROGRESS BAR
        title_text = st.text("Progress Bar")
        my_progress_bar = st.progress(0)
        status_text = st.empty()

        # PROCESS TRAJECTORIES
        all_data = read_and_process_file(Path(selected_file))
        all_data["vx"] = np.nan  # Initialize the velocity columns
        all_data["vy"] = np.nan  # Initialize the velocity columns
        all_trajs = process_trajectories(all_data, pa)
        calculate_grid_dimensions(pa)
        save_data(all_trajs, folder_save, "traj_data.pkl")
        save_data(pa, folder_save, "parameters.pkl")

        # FIELDS
        df_observables = initialize_dict(pa.NB_CG_X, pa.NB_CG_Y)
        compute_fields(all_trajs, df_observables, pa, my_progress_bar, status_text)
        save_data(df_observables, folder_save, "dictionnary_observables.pkl")
        title_text.empty()

    # Load the data from the pickle files
    params = load_data(folder_save, "parameters.pkl")
    dict_observables = load_data(folder_save, "dictionnary_observables.pkl")

    # Set the interpolation button initially to False
    st.session_state["zsmooth"] = False

    # Button to toggle interpolation
    toggle_interpolation = st.sidebar.checkbox(
        "Interpolation", value=st.session_state["zsmooth"]
    )

    # Update session state based on checkbox
    st.session_state["zsmooth"] = "best" if toggle_interpolation else False

    # Double column layout with the density and variance heatmaps and velocity field
    col1, col2 = st.columns([1, 1])  # Adjust the ratio to control space allocation
    with col1:
        fig = update_figure(
            df_observables=dict_observables,
            pa=params,
            plot_density=True,
            zsmooth=st.session_state["zsmooth"],
        )
        st.plotly_chart(fig)
        figname = (
            pa.SELECTED_NAME
            + f"_velocity_field_for_density_heatmap_zsmooth{st.session_state['zsmooth']}.pdf"
        )
        img_bytes = fig.to_image(format="pdf")
        st.sidebar.download_button(
            label="Download Density Heatmap", data=img_bytes, file_name=figname
        )
        plt.close()
    with col2:
        fig = update_figure(
            dict_observables, params, False, st.session_state["zsmooth"]
        )
        st.plotly_chart(fig)
        figname = (
            pa.SELECTED_NAME
            + f"_velocity_field_for_density_heatmap_zsmooth{st.session_state['zsmooth']}.pdf"
        )
        img_bytes = fig.to_image(format="pdf")
        st.sidebar.download_button(
            label="Download Variance Heatmap", data=img_bytes, file_name=figname
        )
        plt.close()


def run_cctv_analysis(selected_file: str) -> None:
    """
    Run the animation tab with the selected file.

    Parameters:
        selected_file (str): The path of the selected file.

    Returns:
        None
    """
    main(selected_file)
