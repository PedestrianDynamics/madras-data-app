""" Map of the gps trajectories coupled with the contacts locations. """

from typing import Tuple

import folium
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib import colormaps
from plotly.graph_objects import Figure
from streamlit_folium import st_folium


def load_data(
    gps_path: str, contacts_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load GPS tracks and contact data from pickled files.

    Args:
        gps_path (str): The path to the GPS data directory.
        contacts_path (str): The path to the contacts data directory.

    Returns:
        tuple: A tuple containing all GPS tracks and contact GPS merged data.
    """
    all_gps_tracks = pd.read_pickle(gps_path + "all_gps_tracks.pickle")
    contact_gps_merged = pd.read_pickle(contacts_path + "contacts_gps_merged.pickle")
    contacts_data = pd.read_pickle(contacts_path + "contacts_data.pickle")
    return all_gps_tracks, contact_gps_merged, contacts_data


def initialize_map(all_gps_tracks: pd.DataFrame) -> folium.Map:
    """
    Initialize the map centered on the middle point of Ludovic-Gardre1's first track.

    Args:
        all_gps_tracks (pd.DataFrame): DataFrame containing all GPS tracks.

    Returns:
        folium.Map: A folium map object.
    """
    first_track_df = all_gps_tracks[all_gps_tracks["name_subj"] == "Ludovic-Gardre1"]
    map_center = first_track_df.iloc[len(first_track_df) // 2][
        ["latitude", "longitude"]
    ].tolist()
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
        folium.PolyLine(
            track_points, color=hex_color, weight=2.5, opacity=1, popup=name_subj
        ).add_to(map_object)


def add_contact_markers(
    map_object: folium.Map, contact_gps_merged: pd.DataFrame, path_icon: str
) -> None:
    """
    Add markers for each contact point on the map.

    Args:
        map_object (folium.Map): The folium map object where markers will be added.
        contact_gps_merged (pd.DataFrame): DataFrame containing contact GPS merged data.
    """
    for index, row in contact_gps_merged.iterrows():
        icon_person = folium.features.CustomIcon(
            icon_image=path_icon + "/contact_icon.png", icon_size=(30, 30)
        )
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=icon_person,
            popup=row["name_subj"],
        ).add_to(map_object)


def plot_histogram(df: pd.DataFrame) -> Figure:
    """
    Creates an interactive bar chart using Plotly to visualize the total number of collisions.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'Total-number-of-collisions' column.

    Returns:
        Figure: The Plotly figure object for the histogram.
    """
    # Slider for selecting the number of bins
    bins = st.slider(
        "Select number of bins:", min_value=5, max_value=11, value=10, step=3
    )
    fig = px.histogram(
        df["Total-number-of-collisions"], x="Total-number-of-collisions", nbins=bins
    )
    print(df["Total-number-of-collisions"])
    fig.update_layout(
        xaxis_title="Number of contacts along the path", yaxis_title="Number of people"
    )  # Set the range for the log scale

    return fig


def plot_cumulative_contacts(df: pd.DataFrame) -> Figure:
    """To plot cumulative contacts as a function of time using Plotly"""
    # Initialize an empty figure
    fig = go.Figure()

    # Loop through the DataFrame and plot each person's contact times
    for index, row in df.iterrows():
        times = row.dropna().values  # Get the 'Détail' times for the person
        if len(times) > 0:
            values = np.cumsum(np.concatenate(([0], np.ones(len(times), dtype="int"))))  # type: ignore
            edges = np.concatenate(
                (times, [df["Duration"].iloc[index].total_seconds()])
            )
            # Add a trace for each person
            fig.add_trace(
                go.Scatter(x=edges, y=values, mode="lines+markers", name=row["Name"])
            )

    # Update layout of the figure
    fig.update_layout(
        title="Cumulative Contacts as a Function of Time",
        xaxis_title="Time [microseconds]",
        yaxis_title="Cumulative Number of Contacts",
        drawstyle="steps-pre",
    )

    return fig


def main() -> None:
    """
    Main function to orchestrate the map creation process and integrate with Streamlit.
    """
    st.title("Map of GPS Trajectories coupled with contacts locations.")

    # Paths to the data directories
    path_data = "./App_data_contacts/"
    path_icon = "./logo_contact/"

    # Load GPS tracks and contact data
    all_gps_tracks, contact_gps_merged, contacts_data = load_data(path_data, path_data)

    # Initialize map and add layers
    my_map = initialize_map(all_gps_tracks)
    add_tile_layer(my_map)
    plot_gps_tracks(my_map, all_gps_tracks)
    add_contact_markers(my_map, contact_gps_merged, path_icon)

    # Display the map in the Streamlit app
    st_folium(my_map, width=825, height=700)

    st.title("Histogram of the total number of collisions")
    fig = plot_histogram(contacts_data)
    st.plotly_chart(fig, use_container_width=True)


def call_main() -> None:
    main()
