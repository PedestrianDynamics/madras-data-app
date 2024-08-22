"""Map of the gps trajectories coupled with the contacts locations."""

from pathlib import Path
from typing import Tuple

import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import colormaps
from plotly.graph_objects import Figure
from streamlit_folium import st_folium

from ..plotting.plots import download_file


def load_data(gps_path: str, contacts_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load GPS tracks and contact data from pickled files.

    Args:
        gps_path (str): The path to the GPS data directory.
        contacts_path (str): The path to the contacts data directory.

    Returns:
        tuple: A tuple containing all GPS tracks and contact GPS merged data.
    """
    all_gps_tracks = pd.read_pickle(gps_path + "/" + "all_gps_tracks.pickle")
    contact_gps_merged = pd.read_pickle(contacts_path + "/" + "contacts_gps_merged.pickle")
    contacts_data = pd.read_pickle(contacts_path + "/" + "contacts_data.pickle")
    return all_gps_tracks, contact_gps_merged, contacts_data


def initialize_map(all_gps_tracks: pd.DataFrame) -> folium.Map:
    """
    Initialize the map centered on the middle point of Ludovic-Gardre1's first track.

    Args:
        all_gps_tracks (pd.DataFrame): DataFrame containing all GPS tracks.

    Returns:
        folium.Map: A folium map object.
    """
    map_center = [45.76714745916146, 4.833552178368124] # first_track_df.iloc[len(first_track_df) // 2][["latitude", "longitude"]].tolist()
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
        folium.PolyLine(track_points, color=hex_color, weight=2.5, opacity=1).add_to(map_object)


def add_contact_markers(map_object: folium.Map, contact_gps_merged: pd.DataFrame, path_icon: str) -> None:
    """
    Add markers for each contact point on the map.

    Args:
        map_object (folium.Map): The folium map object where markers will be added.
        contact_gps_merged (pd.DataFrame): DataFrame containing contact GPS merged data.
    """
    for index, row in contact_gps_merged.iterrows():
        icon_person = folium.features.CustomIcon(icon_image=path_icon + "/contact_icon.png", icon_size=(30, 30))
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=icon_person,
            popup=row["name_subj"],
        ).add_to(map_object)


def plot_histogram(df: pd.DataFrame, bins: int, log_plot: Tuple[bool, bool]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=1800)
    sns.histplot(df['Total-number-of-collisions'], bins=bins, kde=True, log_scale=log_plot, ax=ax)
    plt.xlabel('Number of contacts along the path')
    plt.ylabel('Number of people')
    plt.title('Histogram of the total number of collisions')
    plt.savefig(Path(__file__).parent.parent.parent.absolute() / "data" / "processed" / f"histogram_{bins}.pdf")
    plt.close(fig)  # Close the figure
    return fig


def plot_cumulative_contacts(df: pd.DataFrame) -> Figure:
    """To plot cumulative contacts as a function of time using Plotly"""
    # Initialize an empty figure
    # Drop the non-numeric 'Détail' columns
    detail_data = df.drop(columns=["Name", "Date", "Time-of-stop", "Total-number-of-collisions", "Duration"], inplace=False)

    fig = go.Figure()
    # Loop through the DataFrame and plot each person's contact times
    for index, row in detail_data.iterrows():
        times = row.dropna().values  # Get the 'Détail' times for the person
        if len(times) > 0:
            values = np.cumsum(np.concatenate(([0], np.ones(len(times), dtype="int"))))  # type: ignore
            edges = np.concatenate((times, [df["Duration"].iloc[index].total_seconds()]))
            # Add a trace for each person
            fig.add_trace(go.Scatter(x=edges, y=values, mode="lines+markers"))

    # Update layout of the figure
    fig.update_layout(
        title="Cumulative Contacts as a Function of Time",
        xaxis_title="Time [microseconds]",
        yaxis_title="Cumulative Number of Contacts",
        width=800, height=800
    )

    return fig


def main() -> None:
    """
    Main function to orchestrate the map creation process and integrate with Streamlit.
    """
    st.title("Map of GPS Trajectories coupled with contacts locations.")

    # Paths to the data directories
    path = Path(__file__)
    path_data = str(path.parent.parent.parent.absolute() / "data" / "contacts")
    path_icon = str(path.parent.parent.parent.absolute() / "data" / "assets" / "logo_contact")
    # Load GPS tracks and contact data
    all_gps_tracks, contact_gps_merged, contacts_data = load_data(path_data, path_data)

    # Initialize map and add layers
    my_map = initialize_map(all_gps_tracks)
    add_tile_layer(my_map)
    plot_gps_tracks(my_map, all_gps_tracks)
    add_contact_markers(my_map, contact_gps_merged, path_icon)

    # Display the map in the Streamlit app
    st_folium(my_map, width=825, height=700)

    # Initialize the session state variable if it doesn't exist
    if 'bool_var' not in st.session_state:
        st.session_state['bool_var'] = True


    col1, col2 = st.columns([1, 1])  # Adjust the ratio to control space allocation
    with col1:
        # Slider for selecting the number of bins
        plt = st.empty()
        bins = int(st.slider("Select number of bins:", min_value=5, max_value=11, value=6, step=1))

        # Create a button in the Streamlit app
        if st.button('log-x-scale'):
            # When the button is clicked, toggle the session state boolean variable
            st.session_state['bool_var'] = not st.session_state['bool_var']

        # Display the current value of the session state boolean variable
        st.write(f'Current value of boolean variable: {st.session_state["bool_var"]}')

        fig = plot_histogram(contacts_data, bins, (st.session_state["bool_var"],False))
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
    main()
