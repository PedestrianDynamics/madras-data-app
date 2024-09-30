import pickle
from pathlib import Path

import pandas as pd
import pedpy
import pygwalker as pyg
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

from ..classes.datafactory import load_file
from ..helpers.utilities import setup_walkable_area


def prepare_data(selected_file: str, delta_frame: int) -> pd.DataFrame:
    if selected_file != st.session_state.file_changed:
        trajectory_data = load_file(selected_file)
        st.session_state.trajectory_data = trajectory_data
        st.session_state.file_changed = selected_file

    result_file = Path(selected_file).stem + "_results.pkl"
    if not Path(result_file).exists():
        with st.status("Preparing data ..."):
            trajectory_data = st.session_state.trajectory_data
            walkable_area = setup_walkable_area(trajectory_data)
            speed = pedpy.compute_individual_speed(
                traj_data=trajectory_data,
                frame_step=delta_frame,
                speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
            )
            individual = pedpy.compute_individual_voronoi_polygons(
                traj_data=trajectory_data,
                walkable_area=walkable_area,
                cut_off=pedpy.Cutoff(radius=1.0, quad_segments=3),
            )
            density_voronoi, intersecting = pedpy.compute_voronoi_density(
                individual_voronoi_data=individual, measurement_area=walkable_area
            )
            voronoi_speed = pedpy.compute_voronoi_speed(
                traj_data=trajectory_data,
                individual_voronoi_intersection=intersecting,
                individual_speed=speed,
                measurement_area=walkable_area,
            )
            data_with_speed = voronoi_speed.merge(
                trajectory_data.data, on=["frame"], how="left"
            )
            data_with_speed_density = density_voronoi.merge(
                data_with_speed, on=["frame"], how="left"
            )
            with open(result_file, "wb") as f:
                pickle.dump(data_with_speed_density, f)
    else:
        with open(result_file, "rb") as f:
            data_with_speed_density = pickle.load(f)

    return data_with_speed_density


def run_walker(df: pd.DataFrame) -> None:
    """You should cache your pygwalker renderer, if you don't want your memory to explode."""

    @st.cache_resource
    def get_pyg_renderer(df: pd.DataFrame) -> "StreamlitRenderer":
        # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
        return StreamlitRenderer(
            df,
            spec="./gw_config.json",
            spec_io_mode="rw",
            field_specs={"frame": pyg.FieldSpec(analyticType="dimension")},
        )

    renderer = get_pyg_renderer(df)
    renderer.render_explore()


def run_explorer() -> None:
    """Call explorer woth dataframe from selected file."""
    file_name_to_path = {path.split("/")[-1]: path for path in st.session_state.files}
    filename = str(
        st.selectbox(
            ":open_file_folder: **Select a file**",
            file_name_to_path,
            key="explorer_filename",
        )
    )
    selected_file = file_name_to_path[filename]
    st.session_state.selected_file = selected_file
    df = prepare_data(selected_file, delta_frame=10)
    run_walker(df)
