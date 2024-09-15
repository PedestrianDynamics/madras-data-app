"""Streamlit app to create an animation of pedestrian movements."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.wkt import loads


def load_data(pickle_name: str) -> pd.DataFrame:
    """
    Load pedestrian trajectory data from a pickle file.

    Args:
        pickle_name (str): The name of the pickle file to load.

    Returns:
        pd.DataFrame: DataFrame containing the pedestrian trajectory data.
    """
    pd_trajs = pd.read_pickle(pickle_name)
    return pd_trajs


def transform_polygon(polygon):
    """
    Transforms the coordinates of a polygon from RGF93 (EPSG:2154) to WGS84 (EPSG:4326).

    Args:
        polygon (Polygon): The polygon to be transformed.

    Returns:
        Polygon: The transformed polygon.

    """
    # Initialize the transformer from RGF93 (EPSG:2154) to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    if len(list(polygon.exterior.coords[0])) < 3:
        new_exterior = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
    else:
        new_exterior = [transformer.transform(x, y) for (x, y, z) in polygon.exterior.coords]
    return Polygon(new_exterior)


def trajs_from_rgf93_to_wgs84(trajs: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the coordinates in the 'trajs' DataFrame from RGF93 (EPSG:2154) to WGS84 (EPSG:4326).

    Parameters:
    trajs (pd.DataFrame): DataFrame containing trajectory data with RGF93 coordinates.

    Returns:
    pd.DataFrame: DataFrame with converted coordinates in WGS84 format.
    """
    # Initialize the transformer from RGF93 (EPSG:2154) to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    # Apply the transformation to each row
    trajs[["lon_wgs84", "lat_wgs84"]] = trajs.apply(
        lambda row: pd.Series(transformer.transform(row["x_RGF"], row["y_RGF"])), axis=1
    )

    return trajs


def create_animation_plotly(
    pd_trajs: pd.DataFrame,
    pd_geometry: pd.DataFrame,
    show_polygons: bool,
    frame_duration: float,
    min_velocity: float = 0.0,
    max_velocity: float = 1.0,
) -> go.Figure:
    """
    Create a Plotly animation of pedestrian movements.

    Args:
        pd_trajs (pd.DataFrame): DataFrame containing the pedestrian trajectory data.
        pd_geometry (pd.DataFrame): DataFrame containing geometric data for obstacles.
        show_polygons (bool): Flag to show polygons on the map.

    Returns:
        Figure: Plotly figure object with the pedestrian movement animation.
    """

    # Create the scatter plot with velocity as the color using log scale
    fig = px.scatter_geo(
        pd_trajs,
        lon="lon_wgs84",
        lat="lat_wgs84",
        animation_frame="t/s",
        animation_group="id",
        color="velocity",
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=(min_velocity, max_velocity),
        projection="mercator",
    )
    # Update traces to customize hovertemplate
    fig.update_traces(
        hovertemplate="<b>Pedestrian</b><br><br>"
        + "lon_wgs84: %{lon:.7f}°<br>"
        + "lat_wgs84: %{lat:.7f}°<br>"
        + "velocity: %{marker.color:.4f} m/s<extra></extra>"
    )
    # change the legend of the slider to show the time in seconds
    if len(fig.layout.sliders) > 0:
        fig.layout.sliders[0].currentvalue.prefix = "Time [s]: "

    # Set the scale of the color axis to logarithmic
    fig.update_coloraxes(
        colorbar=dict(title="Velocity [m/s]"),
        colorscale="Viridis",
        cmin=min_velocity,
        cmax=max_velocity,
    )

    if len(fig.layout.sliders) > 0:
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True},
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

    # Add filled geometric obstacles to the map
    if show_polygons:
        for _, row in pd_geometry.iterrows():
            polygon_coords = mapping(row["geometry"])["coordinates"][0]
            center_of_mass = Polygon(polygon_coords).centroid
            fig.add_trace(
                go.Scattergeo(
                    lon=[coord[0] for coord in polygon_coords],
                    lat=[coord[1] for coord in polygon_coords],
                    fill="toself",
                    mode="lines",
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    line=dict(width=1),
                    hoverinfo="text",  # Display hover text
                    hovertext=f"<b>Obstacle {row['Type']}</b><br><br>center of mass<br>lon_wgs84={center_of_mass.x:.7f}°<br>lat_wgs84={center_of_mass.y:.7f}°",
                )
            )
        fig.update_layout(
            height=800,  # Set the height of the figure
            width=800,  # Set the width of the figure
        )
    else:
        fig.update_layout(
            height=800,
            width=800,
        )

    # Updating layout for geographic centering
    fig.update_geos(center=dict(lat=45.76751, lon=4.833584), projection_scale=200000.0, showland=False)
    fig.update_layout(xaxis_title="Longitude [WGS84]", yaxis_title="Latitude [WGS84]", showlegend=False)

    return fig


def compute_pedestrian_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the velocity of pedestrians based on their movement data.

    Args:
        df (pd.DataFrame): DataFrame containing the movement data of pedestrians.

    Returns:
        pd.DataFrame: DataFrame with additional columns for calculated velocities.

    """
    # Calculate velocities
    # Convert the coordinates from degrees to meters
    df[["x_meters", "y_meters"]] = df.apply(
        lambda row: degrees_to_meters(row["lat_wgs84"], row["lon_wgs84"]), axis=1, result_type="expand"
    )

    # Sort the DataFrame by 'id' and 't/s' to ensure correct calculation
    df = df.sort_values(by=["id", "t/s"])

    # Calculate differences in longitude, latitude, and time
    df["delta_x"] = df.groupby("id")["x_meters"].diff()
    df["delta_y"] = df.groupby("id")["y_meters"].diff()
    df["delta_t"] = df.groupby("id")["t/s"].diff()

    # Calculate distance using Euclidean distance
    df["distance"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)

    # Calculate velocity (distance/time)
    df["velocity"] = df["distance"] / df["delta_t"]

    # Round the velocity values to 3 decimal places
    df["velocity"] = df["velocity"].round(3)

    # Fill NaN values in velocity with 0 (first frame for each pedestrian)
    df["velocity"] = df["velocity"].fillna(0)

    # Fill inf values with 0
    df["velocity"] = df["velocity"].replace([np.inf, -np.inf], 0)

    # Round lat and lon to 4 decimal places
    df["lat_wgs84"] = df["lat_wgs84"].round(7)
    df["lon_wgs84"] = df["lon_wgs84"].round(7)

    # Add a column for pedestrian name
    df["Pedestrian"] = "Pedestrian " + df["id"].astype(str)

    return df


def adjust_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the time values in the DataFrame to start from 0.

    Args:
        df (pd.DataFrame): DataFrame containing the pedestrian trajectory data.

    Returns:
        pd.DataFrame: DataFrame with adjusted time values.
    """
    # Subtract the initial time value from all time values
    df["t/s"] = df["t/s"] - df["t/s"][0]
    df["t/s"] = df["t/s"].round(1)

    return df


def degrees_to_meters(lat, lon):
    """
    Converts latitude and longitude coordinates from degrees to meters.
    Parameters:
    lat (float): Latitude coordinate in degrees.
    lon (float): Longitude coordinate in degrees.
    Returns:
    tuple: A tuple containing the converted x and y coordinates in meters.
    """

    R = 6371000  # Radius of Earth in meters
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * lon_rad * np.cos(lat_rad)
    y = R * lat_rad
    return x, y


def prepare_data(traj_path: Path, geometry_path: Path, selected_traj_file: Path) -> None:
    """
    Prepare and convert trajectory data from RGF93 to WGS84 coordinates.

    Args:
        traj_path (Path): The path to the directory containing trajectory files.

    Returns:
        None
    """
    selected_pickle = str(traj_path.parent / "pickle" / (str(Path(selected_traj_file).stem) + "_converted.pkl"))
    if not Path(selected_pickle).exists():
        # Loop over files in trajectories that start with Topview or LargeView
        if str(selected_traj_file.stem).startswith("LargeView"):
            # Assuming the data starts at line 8 (adjust this as necessary)
            df = pd.read_csv(
                selected_traj_file,
                sep=" ",
                header=None,
                skiprows=7,
                names=["id", "frame", "x/m", "y/m", "z/m", "t/s", "x_RGF", "y_RGF"],
            )
            # Convert the coordinates from RGF93 to WGS84
            df_converted = trajs_from_rgf93_to_wgs84(df)
            # Subtract the initial time value from all time values
            df_converted["t/s"] = adjust_time(df_converted)["t/s"]
            # Add a column for pedestrian velocity
            df_converted = compute_pedestrian_velocity(df_converted)
            # Save the converted DataFrame to a pickle file
            PICKLE_SAVE_PATH = str(traj_path.parent / "pickle" / (selected_traj_file.stem + "_converted.pkl"))
            df_converted.to_pickle(PICKLE_SAVE_PATH)

        if str(selected_traj_file.stem).startswith("Topview"):
            # Assuming the data starts at line 8 (adjust this as necessary)
            df = pd.read_csv(
                selected_traj_file,
                sep=" ",
                header=None,
                skiprows=7,
                names=["id", "frame", "x/m", "y/m", "z/m", "id_global", "t/s", "x_RGF", "y_RGF"],
            )
            df = df.drop(columns=["id"])
            df = df.rename(columns={"id_global": "id"})
            # Convert the coordinates from RGF93 to WGS84
            df_converted = trajs_from_rgf93_to_wgs84(df)
            # Subtract the initial time value from all time values
            df_converted["t/s"] = adjust_time(df_converted)["t/s"]
            # Add a column for pedestrian velocity
            df_converted = compute_pedestrian_velocity(df_converted)
            # Save the converted DataFrame to a pickle file
            PICKLE_SAVE_PATH = str(traj_path.parent / "pickle" / (selected_traj_file.stem + "_converted.pkl"))
            df_converted.to_pickle(PICKLE_SAVE_PATH)

    # Geometry data
    geometry_pickle = geometry_path.parent / "pickle" / "geometry_converted.pkl"
    if not geometry_pickle.exists():
        pd_geometry_converted = extract_gps_data_from_csv_geometry(geometry_path / "WKT.csv")
        pd_geometry_converted.to_pickle(geometry_pickle)


def extract_gps_data_from_csv_geometry(file_path: Path) -> pd.DataFrame:
    """
    Extract GPS data from a CSV file and prepare it for Plotly animation.

    Args:
        file_path (str): Path to the CSV file containing GPS coordinates.

    Returns:
        pd.DataFrame: DataFrame containing the GPS data with necessary columns for animation.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract coordinates from the WKT column and transform them into Shapely geometries
    df["geometry"] = df["WKT"].apply(loads)

    # convert from rgf93 to wgs84
    df["geometry"] = df["geometry"].apply(transform_polygon)

    return df


def main(selected_file: str) -> None:
    """
    Main function to run the Streamlit app.
    """
    path = Path(__file__)

    TRAJ_PATH = Path(path.parent.parent.parent.absolute() / "data" / "trajectories")
    GEOMETRY_PATH = Path(path.parent.parent.parent.absolute() / "data" / "geometry")

    # is topview or largeview
    is_topview = str(Path(selected_file).stem).startswith("Topview")

    # select the pickle file
    selected_pickle = str(TRAJ_PATH.parent / "pickle" / (str(Path(selected_file).stem) + "_converted.pkl"))
    geometry_pickle = str(GEOMETRY_PATH.parent / "pickle" / "geometry_converted.pkl")

    # prepare the data
    prepare_data(TRAJ_PATH, GEOMETRY_PATH, Path(selected_file))

    # Load the pedestrian trajectory data
    pd_trajs = load_data(selected_pickle)
    pd_geometry = load_data(geometry_pickle)

    # Title for the sidebar
    st.sidebar.title("Animation settings")

    # Checkbox to toggle the display of geometric polygons
    show_polygons = st.sidebar.checkbox("Show Obstacles", value=True)

    # Calculate minimum and maximum velocities
    min_velocity = pd_trajs["velocity"].min()
    max_velocity = pd_trajs["velocity"].max()

    # Streamlit slider for frame duration
    freq_topview = 30
    freq_largeview = 10
    frame_duration = st.sidebar.slider(
        "Select frame duration (ms)",
        min_value=int(1000 / (2 * freq_topview)),
        max_value=int(1000 / (0.5 * freq_largeview)),
        value=int(1000 / freq_topview) if is_topview else int(1000 / freq_largeview),
        step=5,
    )
    fig = create_animation_plotly(pd_trajs, pd_geometry, show_polygons, frame_duration, min_velocity, max_velocity)
    st.plotly_chart(fig, use_container_width=True)


def run_tab_animation(selected_file: str) -> None:
    """
    Run the animation tab with the selected file.

    Parameters:
        selected_file (str): The path of the selected file.

    Returns:
        None
    """
    main(selected_file)
