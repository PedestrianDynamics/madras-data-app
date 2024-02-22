""" Show general results, including ploting, animation, ..."""

import time
import pedpy
import streamlit as st
from shapely import Polygon

import datafactory

# import docs
import plots
from anim import animate

# import cProfile
# import pstats
# import io


def run_tab2(selected_file, msg):
    """First tab. Plot original data, animatoin, neighborhood."""

    c1, c2 = st.columns((1, 1))
    # todo
    msg.write("")
    if selected_file != st.session_state.file_changed:
        trajectory_data = datafactory.load_file(selected_file)
        st.session_state.trajectory_data = trajectory_data
        st.session_state.file_changed = selected_file

    trajectory_data = st.session_state.trajectory_data
    min_x = trajectory_data.data["x"].min()
    max_x = trajectory_data.data["x"].max()
    min_y = trajectory_data.data["y"].min()
    max_y = trajectory_data.data["y"].max()
    rectangle_coords = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]
    rectangle_polygon = Polygon(rectangle_coords)
    walkable_area = pedpy.WalkableArea(rectangle_polygon)
    data_with_speed = pedpy.compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=5,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    data_with_speed = data_with_speed.merge(
        trajectory_data.data, on=["id", "frame"], how="left"
    )

    ids = trajectory_data.data["id"].unique()
    start_time = time.time()
    rc0, rc1, rc2, rc3 = st.columns((1, 1, 1, 1))
    st.write("---------")
    columns_to_display = ["id", "frame", "x", "y"]
    display = rc0.checkbox("Data", value=False, help="Display data table")
    do_plot_trajectories = rc1.checkbox("Plot", value=False, help="Plot trajectories")
    do_animate = rc2.checkbox(
        "Animation",
        value=False,
        help="Visualise movement of trajecories (Slow, so patience!)",
    )

    # Logic

    if display:
        st.dataframe(trajectory_data.data.loc[:, columns_to_display])

    if do_plot_trajectories:
        c1, c2 = st.columns((0.8, 0.25))
        sample_frame = c2.slider(
            "Every nth frame",
            1,
            1000,
            200,
            10,
            help="plot every_nth_frame",
        )
        uid = c2.number_input(
            "Insert id of pedestrian",
            value=None,
            min_value=int(min(ids)),
            max_value=int(max(ids)),
            placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
            format="%d",
        )
        show_direction = c2.number_input(
            "Choose direction to show",
            value=None,
            min_value=1,
            max_value=4,
            placeholder="Type a number in [1, 4]",
            format="%d",
        )

        fig = plots.plot_trajectories(
            trajectory_data, sample_frame, uid, show_direction, walkable_area
        )
        c1.plotly_chart(fig)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to plot trajectories: {elapsed_time:.2f} seconds")

    if do_animate:
        #        pr = cProfile.Profile()
        #        pr.enable()

        anm = animate(
            data_with_speed,
            walkable_area,
            width=800,
            height=800,
            radius=0.1,  # 0.75
            # title_note="(<span style='color:green;'>M</span>, <span style='color:blue;'>F</span>)",
        )
        #       pr.disable()
        #       s = io.StringIO()
        #       sortby = "cumulative"
        #       ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #        ps.print_stats()
        #       profiling_results = s.getvalue()

        # Display the profiling results in the app
        # st.text_area("Profiling Results", profiling_results, height=300)

        st.plotly_chart(anm)
