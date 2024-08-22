"""Streamlit app to create an animation of pedestrian movements."""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.graph_objs import Figure
from pyproj import Transformer


def load_data(pickle_name: str) -> pd.DataFrame:
    """
    Load pedestrian trajectory data from a pickle file.

    Args:
        pickle_name (str): The name of the pickle file to load.

    Returns:
        pd.DataFrame: DataFrame containing the pedestrian trajectory data.
    """
    pd_trajs = pd.read_pickle(pickle_name)
    print(pd_trajs.head())
    return pd_trajs

# Define a function to apply the transformation
def transform_rgf93_to_wgs84(row, transformer):
    """
    Transforms coordinates from RGF93 to WGS84 using a transformer object.

    Parameters:
    - row: A pandas Series containing the coordinates in RGF93 format.
    - transformer: A transformer object used for the coordinate transformation.

    Returns:
    - A pandas Series containing the transformed coordinates in WGS84 format.
    """
    x_wgs84, y_wgs84 = transformer.transform(row["x_RGF"], row["y_RGF"])
    return pd.Series({"lon_wgs84": x_wgs84, "lat_wgs84": y_wgs84})


def trajs_from_rgf93_to_wgs84(trajs: pd.DataFrame):
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
    trajs[["lon_wgs84", "lat_wgs84"]] = trajs.apply(transform_rgf93_to_wgs84, args=(transformer,), axis=1)

    return trajs

def create_animation_plotly(pd_trajs: pd.DataFrame, selected_file: str) -> Figure:
    """
    Create a Plotly animation of pedestrian movements.

    Args:
        pd_trajs (pd.DataFrame): DataFrame containing the pedestrian trajectory data.

    Returns:
        Figure: Plotly figure object with the pedestrian movement animation.
    """
    fig: Figure = px.scatter(
        pd_trajs,
        x="lon_wgs84",
        y="lat_wgs84",
        animation_frame="frame",
        animation_group="id",
        color="id",
        hover_name="id",
        range_x=[pd_trajs["lon_wgs84"].min(), pd_trajs["lon_wgs84"].max()],
        range_y=[pd_trajs["lat_wgs84"].min(), pd_trajs["lat_wgs84"].max()],
    )
    fig.update_layout(title="Pedestrian Movements Over Time", xaxis_title="X Coordinate", yaxis_title="Y Coordinate")

    # Adjust layout size if needed
    fig.update_layout(height=700, width=900)
    # Adjust animation speed
    if Path(selected_file).stem.startswith("LargeView_zoom"):
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 10**(-1)
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 10**(-6)

    return fig

def visualize_map(pd_trajs: pd.DataFrame) -> Figure:
    # Filter data for pedestrians with frame 0
    initial_pedestrians = pd_trajs[pd_trajs['frame'] == 0]
    # Create the scatter_mapbox figure
    fig = px.scatter_mapbox(
        initial_pedestrians,
        lat="lat_wgs84",  # Ensure this column contains latitude values
        lon="lon_wgs84",  # Ensure this column contains longitude values
        hover_name="id",
        color="id",
        # color_discrete_sequence=["fuchsia"],
        zoom=17.5,
        mapbox_style="open-street-map",
    )
    # Adjust layout size if needed
    fig.update_layout(height=700, width=900)

    return fig

def prepare_data(traj_path: Path) -> None:
    """
    Prepare and convert trajectory data from RGF93 to WGS84 coordinates.

    Args:
        traj_path (Path): The path to the directory containing trajectory files.

    Returns:
        None
    """
    # Loop over files in trajectories that start with Topview
    for selected_traj_file in traj_path.glob("LargeView*"):
        # Assuming the data starts at line 8 (adjust this as necessary)
        df = pd.read_csv(selected_traj_file, sep=' ', header=None, skiprows=7, names=["id", "frame", "x/m", "y/m", "z/m", "t/s", "x_RGF", "y_RGF"])

        # Convert the coordinates from RGF93 to WGS84
        df_converted = trajs_from_rgf93_to_wgs84(df)

        # Save the converted DataFrame to a pickle file
        PICKLE_SAVE_PATH = str(traj_path.parent / "processed" / (selected_traj_file.stem + "_converted.pkl"))
        df_converted.to_pickle(PICKLE_SAVE_PATH)


def main(selected_file: str) -> None:
    """
    Main function to run the Streamlit app.
    """
    path = Path(__file__)

    TRAJ_PATH = path.parent.parent.parent.absolute() / "data" / "trajectories"
    # prepare_data(TRAJ_PATH)

    # select the pickle file
    selected_pickle = str(TRAJ_PATH.parent / "processed" /  (str(Path(selected_file).stem) + "_converted.pkl"))

    pd_trajs = load_data(selected_pickle)

    col1, col2 = st.columns([1, 1])  # Adjust the ratio to control space allocation
    with col1:
        st.title("Animation")
        fig = create_animation_plotly(pd_trajs, selected_file)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.title("Initial position of pedestrians")
        fig = visualize_map(pd_trajs)
        st.plotly_chart(fig, use_container_width=True)


def run_tab_animation(selected_file: str) -> None:
    main(selected_file)

