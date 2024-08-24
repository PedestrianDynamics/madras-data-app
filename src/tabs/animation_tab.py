"""Streamlit app to create an animation of pedestrian movements."""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from pyproj import Transformer
from shapely.wkt import loads
import plotly.graph_objects as go
from shapely.geometry import Polygon
from shapely.geometry import mapping


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
    pd_trajs: pd.DataFrame, pd_geometry: pd.DataFrame, show_polygons: bool, is_topview: bool
) -> go.Figure:
    """
    Create a Plotly animation of pedestrian movements.

    Args:
        pd_trajs (pd.DataFrame): DataFrame containing the pedestrian trajectory data.
        pd_geometry (pd.DataFrame): DataFrame containing geometric data for obstacles.
        show_polygons (bool): Flag to show polygons on the map.
        is_topview (bool): Flag to adjust settings for top view.

    Returns:
        Figure: Plotly figure object with the pedestrian movement animation.
    """
    fig = px.scatter(
        pd_trajs,
        x="lon_wgs84",
        y="lat_wgs84",
        animation_frame="frame",
        animation_group="id",
        color="id",
        hover_name="id",
    )

    fig.update_layout(
        xaxis_title="Longitude [WGS84]",
        yaxis_title="Latitude [WGS84]",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1.4,
    )

    if show_polygons:
        for _, row in pd_geometry.iterrows():
            polygon_coords = mapping(row["geometry"])["coordinates"][0]
            fig.add_trace(
                go.Scatter(
                    x=[coord[0] for coord in polygon_coords],
                    y=[coord[1] for coord in polygon_coords],
                    fill="toself",
                    mode="lines",
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    line=dict(width=1),
                    name=f"Obstacle {row['Type']}",
                )
            )
        fig.update_layout(
            height=1550,
            width=1200,
            xaxis=dict(range=[4.8325, 4.8346]),
            yaxis=dict(range=[45.767, 45.7679]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-1.8,
                xanchor="center",
                x=0.5,
            ),
        )
    else:
        fig.update_layout(
            height=700,
            width=900,
            xaxis_range=[pd_trajs["lon_wgs84"].min(), pd_trajs["lon_wgs84"].max()],
            yaxis_range=[pd_trajs["lat_wgs84"].min(), pd_trajs["lat_wgs84"].max()],
        )

    if len(pd_trajs["frame"].unique()) > 1:
        frame_duration = 1000 / 30.0 if is_topview else 1000.0 / 12.0
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_duration
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 1e-7

    return fig


def visualize_map(pd_trajs: pd.DataFrame, pd_geometry: pd.DataFrame, show_polygons: bool) -> go.Figure:
    """
    Visualizes a map with a scatter plot of initial pedestrian positions and optional filled geometric obstacles.

    Args:
        pd_trajs (pd.DataFrame): DataFrame containing pedestrian trajectories.
        pd_geometry (pd.DataFrame): DataFrame containing geometric obstacle information.
        show_polygons (bool): Flag indicating whether to show filled geometric obstacles on the map.

    Returns:
        Figure: Scatter mapbox figure displaying the map visualization.
    """
    # Filter data for pedestrians with frame 0
    initial_pedestrians = pd_trajs[pd_trajs["frame"] == 0]

    # Create the scatter_mapbox figure
    fig = px.scatter_mapbox(
        initial_pedestrians,
        lat="lat_wgs84",
        lon="lon_wgs84",
        hover_name="id",
        color="id",
        mapbox_style="open-street-map",
    )
    fig.update_layout(
        mapbox_bounds={
            "west": 4.831,  # Minimum longitude
            "east": 4.836,  # Maximum longitude
            "south": 45.7665,  # Minimum latitude
            "north": 45.7684,  # Maximum latitude
        },
    )
    if show_polygons:
        # Add filled geometric obstacles to the map
        for _, row in pd_geometry.iterrows():
            try:
                polygon_coords = mapping(row["geometry"])["coordinates"][0]
                fig.add_trace(
                    go.Scattermapbox(
                        lon=[coord[0] for coord in polygon_coords],
                        lat=[coord[1] for coord in polygon_coords],
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        line=dict(width=1),
                        name=f"Obstacle {row['Type']}",
                    )
                )
            except Exception as e:
                print(f"Error processing polygon: {e}")

        # Adjust the layout size of the figure and position the legend below the map
        fig.update_layout(
            height=1200,
            width=900,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-1.2,
                xanchor="center",
                x=0.5,
            ),
        )
    else:
        # Adjust the layout size of the figure
        fig.update_layout(
            height=900,
            width=1100,
        )

    return fig


def prepare_data(traj_path: Path, GEOMETRY_PATH: Path, selected_traj_file: Path) -> None:
    """
    Prepare and convert trajectory data from RGF93 to WGS84 coordinates.

    Args:
        traj_path (Path): The path to the directory containing trajectory files.

    Returns:
        None
    """

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

        # Save the converted DataFrame to a pickle file
        PICKLE_SAVE_PATH = str(traj_path.parent / "pickle" / (selected_traj_file.stem + "_converted.pkl"))
        df_converted.to_pickle(PICKLE_SAVE_PATH)

    # Geometry data
    geometry_pickle = GEOMETRY_PATH.parent.parent / "pickle" / "geometry_converted.pkl"
    if not geometry_pickle.exists():
        pd_geometry_converted = extract_gps_data_from_csv_geometry(GEOMETRY_PATH / "WKT.csv")
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

    TRAJ_PATH = path.parent.parent.parent.absolute() / "data" / "trajectories"
    GEOMETRY_PATH = Path(path.parent.parent.parent.absolute() / "data" / "other_datasets" / "geometry")

    # is topview or largeview
    is_topview = str(Path(selected_file).stem).startswith("Topview")

    # select the pickle file
    selected_pickle = str(TRAJ_PATH.parent / "pickle" / (str(Path(selected_file).stem) + "_converted.pkl"))
    geometry_pickle = str(GEOMETRY_PATH.parent.parent / "pickle" / "geometry_converted.pkl")

    # if selected_pickle does not exist, prepare the data
    if not Path(selected_pickle).exists():
        prepare_data(TRAJ_PATH, GEOMETRY_PATH, Path(selected_file))

    # Load the pedestrian trajectory data
    pd_trajs = load_data(selected_pickle)
    pd_geometry = load_data(geometry_pickle)

    # Checkbox to toggle the display of geometric polygons
    show_polygons = st.checkbox("Show Obstacles", value=True)

    # Create the Streamlit app
    col1, col2 = st.columns([1, 1])  # Adjust the ratio to control space allocation

    with col1:
        st.title("Initial position of pedestrians")
        fig = visualize_map(pd_trajs, pd_geometry, show_polygons)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.title("Animation")
        fig = create_animation_plotly(pd_trajs, pd_geometry, show_polygons, is_topview)
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
