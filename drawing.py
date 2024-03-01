"""Collection of drawing functionalities."""

import logging
from typing import Any, List, Tuple

import pedpy
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import plots
import utilities


def drawing_canvas(trajectory_data: pedpy.TrajectoryData, walkable_area: pedpy.WalkableArea) -> Tuple[Any, float, float, float]:
    """Draw trajectories as img and prepare canvas."""
    drawing_mode = st.sidebar.radio(
        "**Measurement:**",
        ("Area", "Transform"),
    )
    # fig = plots.plot_trajectories(trajectory_data, framerate=50, walkable_area=walkable_area)
    # st.sidebar.plotly_chart(fig)
    if drawing_mode == "Area":
        drawing_mode = "rect"

    if drawing_mode == "Transform":
        drawing_mode = "transform"

    stroke_width = st.sidebar.slider("**Stroke width:**", 1, 25, 3)
    if st.session_state.bg_img is None:
        logging("START new canvas")
        data = trajectory_data.data
        min_x = trajectory_data.data["x"].min()
        max_x = trajectory_data.data["x"].max()
        min_y = trajectory_data.data["y"].min()
        max_y = trajectory_data.data["y"].max()

        bg_img, img_width, img_height, dpi, scale = plots.bg_img(data, min_x, max_x, min_y, max_y)
        st.session_state.scale = scale
        st.session_state.dpi = dpi
        st.session_state.img_width = img_width
        st.session_state.img_height = img_height
        st.session_state.bg_img = bg_img
    else:
        bg_img = st.session_state.bg_img
        scale = st.session_state.scale
        dpi = st.session_state.dpi
        img_height = st.session_state.img_height
        img_width = st.session_state.img_width

    canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="#060EE8",
        background_color="#eee",
        background_image=bg_img,
        update_streamlit=True,
        width=img_width,
        height=img_height,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    return canvas, dpi, scale, img_height


def get_measurement_area(
    trajectory_data: pedpy.TrajectoryData,
    canvas: Any,
    dpi: float,
    scale: float,
    img_height: float,
) -> List[pedpy.MeasurementArea]:
    """Return a list of drawn measurement areas."""
    min_x = trajectory_data.data["x"].min()
    max_x = trajectory_data.data["x"].max()
    min_y = trajectory_data.data["y"].min()
    max_y = trajectory_data.data["y"].max()
    boundaries = (min_x, max_x, min_y, max_y)
    measurement_areas = []
    rects = plots.draw_rects(
        canvas,
        img_height,
        dpi,
        scale,
        boundaries,
    )
    if not rects:
        measurement_areas.append(utilities.setup_measurement_area(min_x, max_x, min_y, max_y))
        return measurement_areas

    for ir, _ in enumerate(rects):
        from_x = rects[ir]["x"][0]
        to_x = rects[ir]["x"][1]
        from_y = rects[ir]["y"][3]
        to_y = rects[ir]["y"][0]
        measurement_areas.append(utilities.setup_measurement_area(from_x, to_x, from_y, to_y))

    return measurement_areas
