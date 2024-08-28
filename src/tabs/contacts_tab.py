"""Map of the gps trajectories coupled with the contacts locations."""

from pathlib import Path
from typing import Tuple

import folium
import gpxpy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from matplotlib import colormaps
from plotly.graph_objects import Figure
from streamlit_folium import st_folium

from ..plotting.plots import download_file


def load_and_process_contacts_data(csv_path: Path, pickle_path: Path) -> None:
    """
    Load and process contacts data from a CSV file.

    Parameters:
    - csv_path (Path): The path to the CSV file containing the contacts data.
    - pickle_path (Path): The path to save the processed data as a pickle file.

    Returns:
    None
    """

    # Load the data
    df = pd.read_csv(csv_path / "Contacts.csv")

    # Convert 'Date' to datetime format (day/month/year)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # Convert 'Duration' and 'Détail' columns to timedelta
    df["Duration"] = df["Duration"].apply(convert_to_timedelta)
    df.iloc[:, 5:] = df.iloc[:, 5:].map(convert_to_timedelta)

    # Convert 'Détail' entries to total seconds
    df.iloc[:, 5:] = df.iloc[:, 5:].apply(lambda col: col.apply(lambda x: x.total_seconds() if pd.notna(x) else None))

    # Save the DataFrame to a pickle file
    df.to_pickle(pickle_path / "contacts_data.pickle")

    # Process the saved pickle file
    all_instant_contacts = pd.read_pickle(pickle_path / "contacts_data.pickle")
    processed_df = process_contacts_data(all_instant_contacts)

    # Save the processed DataFrame to a pickle file
    processed_df.to_pickle(pickle_path / "contacts_data_melted.pickle")


def convert_to_timedelta(time_str: str) -> pd.Timedelta:
    """
    Convert a string representation of time to a pandas Timedelta object.

    Parameters:
        time_str (str): The string representation of time in the format "HH:MM:SS.micros".

    Returns:
        pd.Timedelta: The converted time as a pandas Timedelta object.

    """

    if pd.isna(time_str):
        return pd.NaT
    else:
        hours, minutes, seconds_micros = time_str.split(":")
        seconds, microseconds = seconds_micros.split(".")
        return pd.Timedelta(
            hours=int(hours), minutes=int(minutes), seconds=int(seconds), microseconds=int(microseconds)
        )


def process_contacts_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the contacts data DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing contacts data.

    Returns:
        pd.DataFrame: The processed DataFrame with melted and transformed data.
    """

    # Drop non-numeric 'Détail' columns
    df.drop(columns=["Date", "Time-of-stop", "Total-number-of-collisions", "Duration"], inplace=True)

    # Transpose the DataFrame and rename columns
    df = df.transpose()
    df.columns = ["Subject_" + str(i) for i in range(1, 25)]

    # Drop the 'Name' index
    df.drop(index=["Name"], inplace=True)

    # Use melt to transform the DataFrame
    df_melted = df.melt(var_name="name_subj", value_name="time_seconds")

    # Drop NaN values to only keep rows with time values
    df_melted.dropna(subset=["time_seconds"], inplace=True)

    # Reset the index of the resulting DataFrame
    return df_melted.reset_index(drop=True)


def process_gpx(gpx_path: Path, pickle_path: Path) -> None:
    """
    Process GPX files and save the data to a pickle file.

    Args:
        gpx_path (Path): The path to the folder containing GPX files.
        pickle_path (Path): The path to save the pickle file.

    Returns:
        None
    """

    # Initialize an empty list to collect data
    data = []

    # Loop through all files in the folder # the path is a Path object, loo with glob instead
    for filename in gpx_path.glob("*.gpx"):
        if not str(filename.stem).startswith("."):
            data.extend(parse_gpx_file(filename))

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Save the DataFrame to a pickle file
    df.to_pickle(pickle_path / "all_gps_tracks.pickle")

    # Process the saved pickle file
    process_tracks_data(pickle_path)


def parse_gpx_file(filename: str) -> list[dict]:
    """
    Parses a GPX file and returns a list of dictionaries containing the extracted data.

    Args:
        filename (str): The path to the GPX file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a data point extracted from the GPX file.
            Each dictionary contains the following keys:
                - name_subj (str): The name of the subject.
                - latitude (float): The latitude coordinate.
                - longitude (float): The longitude coordinate.
                - time (datetime): The timestamp of the data point.

    """

    name_subj = str(filename.stem)
    data = []
    with open(filename, "r", encoding="utf-8") as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append(
                        {
                            "name_subj": name_subj,
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                            "time": point.time,
                        }
                    )
    return data


def process_tracks_data(pickle_path: Path) -> None:
    """
    Process the tracks data by performing the following steps:
    1. Load all tracks from a pickle file.
    2. Convert the 'time' column to timedelta by subtracting a reference time.
    3. Convert the 'time_timedelta' column to total seconds.
    4. Adjust the 'time_seconds' column for each track.
    5. Drop unnecessary columns.
    6. Drop duplicate rows where 'time_seconds' is the same but keep the first one.
    7. Save the processed DataFrame to a pickle file.

    Parameters:
    - pickle_path (Path): The path to the pickle file.

    Returns:
    - None
    """
    # Load all tracks
    all_gps_tracks = pd.read_pickle(pickle_path / "all_gps_tracks.pickle")

    # Convert 'time' column to timedelta by subtracting a reference time
    reference_time = all_gps_tracks["time"].min()
    all_gps_tracks["time_timedelta"] = all_gps_tracks["time"] - reference_time

    # Convert 'time_timedelta' to total seconds
    all_gps_tracks["time_seconds"] = all_gps_tracks["time_timedelta"].dt.total_seconds()

    # Adjust time_seconds for each track
    unique_tracks = all_gps_tracks["name_subj"].unique()
    for name_subj in unique_tracks:
        track_df = all_gps_tracks[all_gps_tracks["name_subj"] == name_subj]
        initial_time = track_df.iloc[0]["time_seconds"]
        all_gps_tracks.loc[track_df.index, "time_seconds"] -= initial_time

    # Drop unnecessary columns
    all_gps_tracks.drop(columns=["time", "time_timedelta"], inplace=True)

    # Drop duplicate rows where time_seconds is the same but keep the first one
    all_gps_tracks.drop_duplicates(subset=["name_subj", "time_seconds"], inplace=True)

    # Save the processed DataFrame to a pickle file
    all_gps_tracks.to_pickle(pickle_path / "all_gps_tracks_timeseconds.pickle")


def merge_contacts_and_gps_data(path_pickle: Path) -> None:
    """
    Merge contacts data and GPS data based on the 'time_seconds' column.

    Args:
        contacts_path (Path): The path to the contacts data directory.
        output_path (Path): The path to the output directory.

    Returns:
        None
    """
    # Load all tracks
    df1 = pd.read_pickle(path_pickle / "contacts_data_melted.pickle")
    df2 = pd.read_pickle(path_pickle / "all_gps_tracks_timeseconds.pickle")

    # Convert 'time_seconds' to numeric
    df1["time_seconds"] = pd.to_numeric(df1["time_seconds"], errors="coerce")
    df2["time_seconds"] = pd.to_numeric(df2["time_seconds"], errors="coerce")

    # Initialize an empty list to collect interpolated data
    interpolated_data = []
    # Process GPS data for interpolation
    for name_subj, group in df2.groupby("name_subj"):
        # Interpolate data for the current group
        interpolated_group = interpolate_data(group)
        # Add the 'name_subj' column to the interpolated group
        interpolated_group["name_subj"] = name_subj
        # Append the interpolated group to the list
        interpolated_data.append(interpolated_group)

    # Concatenate all interpolated groups into a single DataFrame
    df2_interpolated = pd.concat(interpolated_data, ignore_index=True)
    df2_interpolated.reset_index(level=0, inplace=True)
    df2 = df2_interpolated.drop(columns=["index"])

    # Sort both DataFrames by 'time_seconds'
    df1.sort_values(by="time_seconds", inplace=True)
    df2.sort_values(by="time_seconds", inplace=True)

    # Convert 'time_seconds' to float64
    df2["time_seconds"] = df2["time_seconds"].astype("float64")

    # Perform an asof merge
    merged_df = pd.merge_asof(df1, df2, on="time_seconds", by="name_subj", direction="nearest")
    # - **Grouping**: The merge operation groups the data by `name_subj`.
    # - **Juxtaposition**: Within each group, it aligns rows from `df1` and `df2` based on the `time_seconds` column.
    # - **Merging**: For each subject (e.g., `subj7`), if the `time_seconds` values are similar (nearest match), the content from both DataFrames is combined into a single row.
    # - **Directionality**: The `direction="nearest"` parameter ensures that the merge operation considers the nearest value in `df2` for each row in `df1`.

    # Drop unnecessary columns and NaN rows
    merged_df.drop(columns=["time_seconds"], inplace=True)
    merged_df.dropna(subset=["latitude", "longitude"], inplace=True)

    # Save the merged DataFrame to a pickle file
    merged_df.to_pickle(path_pickle / "contacts_gps_merged.pickle")


def interpolate_data(group):
    """
    Interpolates missing data in a group of GPS contacts.
    Parameters:
        group (DataFrame): A pandas DataFrame containing GPS contact data.
    Returns:
        DataFrame: A pandas DataFrame with missing data interpolated.
    """

    # Set 'time_seconds' as the index
    group = group.set_index("time_seconds")

    # Remove any duplicated indices, keeping the first occurrence
    group = group.loc[~group.index.duplicated(keep="first")]

    # Reindex to fill missing seconds, ensuring the range covers all possible indices
    if not group.index.empty:
        group = group.reindex(range(int(group.index.min()), int(group.index.max()) + 1))

    # Interpolate latitude and longitude linearly, handling NaN values
    group[["latitude", "longitude"]] = group[["latitude", "longitude"]].interpolate(
        method="linear", limit_direction="both"
    )

    # Reset index to return the DataFrame to its original structure
    group.reset_index(inplace=True)

    return group


def load_data(path_pickle: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load GPS tracks and contact data from pickled files.

    Args:
        gps_path (str): The path to the GPS data directory.
        contacts_path (str): The path to the contacts data directory.

    Returns:
        tuple: A tuple containing all GPS tracks and contact GPS merged data.
    """
    all_gps_tracks = pd.read_pickle(path_pickle / "all_gps_tracks.pickle")
    contact_gps_merged = pd.read_pickle(path_pickle / "contacts_gps_merged.pickle")
    contacts_data = pd.read_pickle(path_pickle / "contacts_data.pickle")
    return all_gps_tracks, contact_gps_merged, contacts_data


def initialize_map() -> folium.Map:
    """
    Initialize the map centered on the middle point of Ludovic-Gardre1's first track.

    Args:
        all_gps_tracks (pd.DataFrame): DataFrame containing all GPS tracks.

    Returns:
        folium.Map: A folium map object.
    """
    map_center = [
        45.76714745916146,
        4.833552178368124,
    ]
    return folium.Map(location=map_center, zoom_start=17.5)


def add_tile_layer(map_object: folium.Map) -> None:
    """
    Add a Google Satellite tile layer to enhance the map visualization.

    Args:
        map_object (folium.Map): The folium map object to which the tile layer will be added.
    """
    google_satellite = folium.TileLayer(
        tiles="http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
        opacity=1.0,
    )
    google_satellite.add_to(map_object)


def plot_gps_tracks(map_object: folium.Map, all_gps_tracks: pd.DataFrame) -> None:
    """
    Plot each GPS track on the map with a unique color.

    Args:
        map_object (folium.Map): The folium map object where tracks will be plotted.
        all_gps_tracks (pd.DataFrame): DataFrame containing all GPS tracks.
    """
    unique_tracks = all_gps_tracks["name_subj"].unique()
    viridis = colormaps.get_cmap("viridis")
    for track_index, name_subj in enumerate(unique_tracks):
        track_df = all_gps_tracks[all_gps_tracks["name_subj"] == name_subj]
        track_points = track_df[["latitude", "longitude"]].values.tolist()
        rgba_color = viridis(track_index / len(unique_tracks))
        hex_color = mcolors.to_hex(rgba_color)
        folium.PolyLine(track_points, color=hex_color, weight=2.5, opacity=1, name=name_subj, popup=name_subj).add_to(
            map_object
        )


def add_contact_markers(map_object: folium.Map, contact_gps_merged: pd.DataFrame, path_icon: str) -> None:
    """
    Add markers for each contact point on the map.

    Args:
        map_object (folium.Map): The folium map object where markers will be added.
        contact_gps_merged (pd.DataFrame): DataFrame containing contact GPS merged data.
    """
    for _, row in contact_gps_merged.iterrows():
        icon_person = folium.features.CustomIcon(icon_image=path_icon + "/contact_icon.png", icon_size=(30, 30))
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=icon_person,
            popup=row["name_subj"],
        ).add_to(map_object)


def plot_histogram(df: pd.DataFrame, bins: int, log_plot: Tuple[bool, bool]) -> plt.Figure:
    """
    Plot a histogram of the total number of collisions.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        bins (int): The number of bins for the histogram.
        log_plot (Tuple[bool, bool]): A tuple indicating whether to use a logarithmic scale for the x-axis and y-axis, respectively.

    Returns:
        plt.Figure: The generated matplotlib Figure object.
    """
    # Remove the rows with zero collisions if log_plot is True
    if log_plot[0]:
        data = df["Total-number-of-collisions"].replace(0, np.nan)
    else:
        data = df["Total-number-of-collisions"]

    # Create the histogram
    fig, ax = plt.subplots(figsize=(10, 6), dpi=1800)
    sns.histplot(data, bins=bins, kde=True, log_scale=log_plot, ax=ax)
    plt.xlabel("Number of contacts along the path")
    plt.ylabel("Number of people")
    plt.title("Histogram of the total number of collisions")
    plt.savefig(Path(__file__).parent.parent.parent.absolute() / "data" / "processed" / f"histogram_{bins}.pdf")
    plt.close(fig)

    return fig


def plot_cumulative_contacts(df: pd.DataFrame) -> Figure:
    """
    Plot the cumulative number of contacts as a function of time.
    Args:
        df (pd.DataFrame): The input DataFrame containing contact data.
    Returns:
        Figure: The generated plot.
    """

    # Drop the non-numeric 'Détail' columns
    detail_data = df.drop(
        columns=[
            "Date",
            "Time-of-stop",
            "Total-number-of-collisions",
            "Duration",
        ],
        inplace=False,
    )

    # Initialize an empty figure
    fig = go.Figure()
    # Loop through the DataFrame and plot each person's contact times
    for index, row in detail_data.iterrows():
        times = row.dropna().values  # Get the 'Détail' times for the person
        if len(times) > 0:
            values = np.cumsum(np.concatenate(([0], np.ones(len(times), dtype="int"))))  # type: ignore
            edges = np.concatenate((times, [df["Duration"].iloc[index].total_seconds()]))
            # Add a trace for each person
            fig.add_trace(go.Scatter(x=edges, y=values, mode="lines+markers", name=f"Subject {row.name}"))

    # Update layout of the figure
    fig.update_layout(
        title="Cumulative number of contacts as a function of time",
        xaxis_title="Time [s]",
        yaxis_title="Cumulative number of contacts",
        width=800,
        height=600,
    )

    return fig


def main() -> None:
    """
    Main function for generating a map of GPS trajectories coupled with contacts locations.

    This function performs the following steps:
    1. Sets the title of the map.
    2. Defines the paths to the data directories.
    3. Loads GPS tracks and contact data.
    4. Initializes the map and adds layers.
    5. Displays the map in the Streamlit app.
    6. Initializes the session state variable if it doesn't exist.
    7. Creates a slider for selecting the number of bins.
    8. Creates a button for toggling the session state boolean variable.
    9. Displays the current value of the session state boolean variable.
    10. Plots a histogram of the contacts data.
    11. Saves the histogram as a PDF file.
    12. Displays the histogram in the Streamlit app.
    13. Downloads the histogram PDF file.
    14. Plots a cumulative contacts chart.
    15. Displays the cumulative contacts chart in the Streamlit app.
    16. Removes the histogram files created in the processed directory.
    """

    st.title("Map of GPS Trajectories coupled with contacts locations.")

    # Paths to the data directories
    path = Path(__file__)
    path_csv = path.parent.parent.parent.absolute() / "data" / "other_datasets"
    path_pickle = path.parent.parent.parent.absolute() / "data" / "pickle"
    path_gpx = path.parent.parent.parent.absolute() / "data" / "other_datasets" / "GPSTracks"
    path_icon = str(path.parent.parent.parent.absolute() / "data" / "assets" / "logo_contact")

    # If "contacts_gps_merged.pickle" does not exist, run the following code
    if not Path(path_pickle / "contacts_gps_merged.pickle").exists():
        load_and_process_contacts_data(path_csv, path_pickle)
        process_gpx(path_gpx, path_pickle)
        merge_contacts_and_gps_data(path_pickle)

    # Load GPS tracks and contact data
    all_gps_tracks, contact_gps_merged, contacts_data = load_data(path_pickle)

    # Initialize map and add layers
    my_map = initialize_map()
    add_tile_layer(my_map)
    plot_gps_tracks(my_map, all_gps_tracks)
    add_contact_markers(my_map, contact_gps_merged, path_icon)

    # Display the map in the Streamlit app
    st_folium(my_map, width=825, height=700)

    col1, col2 = st.columns([1, 1])  # Adjust the ratio to control space allocation
    with col1:
        # Initialize the session state variable if it doesn't exist
        if "bool_var" not in st.session_state:
            st.session_state["bool_var"] = True

        # Slider for selecting the number of bins
        plt = st.empty()
        bins = int(st.slider("Select number of bins:", min_value=5, max_value=11, value=6, step=1))

        # Create a button in the Streamlit app
        if st.button("log-x-scale"):
            # When the button is clicked, toggle the session state boolean variable
            st.session_state["bool_var"] = not st.session_state["bool_var"]

        # Display the current value of the session state boolean variable
        st.write(f'Current value of boolean variable: {st.session_state["bool_var"]}')

        # Plot a histogram of the contacts data
        fig = plot_histogram(contacts_data, bins, (st.session_state["bool_var"], False))
        figname = Path(f"histogram_{bins}.pdf")
        path = Path(__file__)
        data_directory = path.parent.parent.parent.absolute() / "data" / "processed"
        figname = data_directory / Path(figname)
        st.pyplot(fig)
        download_file(figname)

    with col2:
        fig = plot_cumulative_contacts(contacts_data)
        st.plotly_chart(fig)

    # remove the histogram files created in the processed directory
    for file in data_directory.glob("histogram_*.pdf"):
        file.unlink()


def run_tab_contact() -> None:
    """
    Function to run the contact tab.

    This function calls the main function to execute the contact tab.

    Parameters:
    None

    Returns:
    None
    """
    main()
