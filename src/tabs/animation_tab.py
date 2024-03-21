"""Streamlit app to create an animation of pedestrian movements."""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.graph_objs import Figure


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


def create_animation(pd_trajs: pd.DataFrame) -> Figure:
    """
    Create a Plotly animation of pedestrian movements.

    Args:
        pd_trajs (pd.DataFrame): DataFrame containing the pedestrian trajectory data.

    Returns:
        Figure: Plotly figure object with the pedestrian movement animation.
    """
    # Create the scatter_mapbox figure
    fig = px.scatter_mapbox(
        pd_trajs,
        lat="lat_wgs84",  # Ensure this column contains latitude values
        lon="lon_wgs84",  # Ensure this column contains longitude values
        hover_name="id",
        animation_frame="frame",
        animation_group="id",
        color="id",
        color_discrete_sequence=["fuchsia"],
        zoom=17.5,
        mapbox_style="open-street-map",
    )

    # Adjust animation speed
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 1

    # Adjust layout size if needed
    fig.update_layout(height=700, width=900)

    return fig


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    path = Path(__file__)
    PICKLE_NAME = str(path.parent.parent.parent.absolute() / "data" / "contacts" / "dataframe_trajectories_oscar_WGS84.pkl")
    
    pd_trajs = load_data(PICKLE_NAME)
    # Drop columns 'x' and 'y' if they exist to reduce the size
    pd_trajs.drop(columns=["x", "y"], inplace=True, errors='ignore')

    
    st.title("Animation Pedestrian on place des Terreaux")
    fig = create_animation(pd_trajs)
    st.plotly_chart(fig, use_container_width=True)


def run_tab_animation() -> None:
    main()

