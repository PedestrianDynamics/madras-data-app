"""Plot functionalities for the app."""

import numpy as np
import pandas as pd
import pedpy
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


def plot_trajectories(
    trajectory_data: pedpy.TrajectoryData,
    framerate: int,
    uid: int,
    show_direction: int,
    walkable_area: pedpy.WalkableArea,
) -> go.Figure:
    fig = go.Figure()
    c1, c2, c3 = st.columns((1, 1, 1))
    data = trajectory_data.data
    num_agents = len(np.unique(data["id"]))
    colors = {
        1: "magenta",  # Assuming 1 is for female
        2: "green",  # Assuming 2 is for male
        3: "black",  # non binary
        4: "blue",
    }
    x_exterior, y_exterior = walkable_area.polygon.exterior.xy
    x_exterior = list(x_exterior)
    y_exterior = list(y_exterior)

    directions = assign_direction_number(data)
    # For each unique id, plot a trajectory
    if uid is not None:
        df = data[data["id"] == uid]
        direction = directions.loc[directions["id"] == uid, "direction_number"].iloc[0]

        color_choice = colors[direction]
        fig.add_trace(
            go.Scatter(
                x=df["x"][::framerate],
                y=df["y"][::framerate],
                line={"color": color_choice},
                marker={"color": color_choice},
                mode="lines",
                name=f"ID {uid}",
            )
        )
    else:
        for uid, df in data.groupby("id"):

            direction = directions.loc[
                directions["id"] == uid, "direction_number"
            ].iloc[0]

            if show_direction is None:
                color_choice = colors[direction]
                fig.add_trace(
                    go.Scatter(
                        x=df["x"][::framerate],
                        y=df["y"][::framerate],
                        line={"color": color_choice},
                        marker={"color": color_choice},
                        mode="lines",
                        name=f"ID {uid}",
                    )
                )
            else:
                if direction == show_direction:
                    color_choice = colors[direction]
                    fig.add_trace(
                        go.Scatter(
                            x=df["x"][::framerate],
                            y=df["y"][::framerate],
                            line={"color": color_choice},
                            marker={"color": color_choice},
                            mode="lines",
                            name=f"ID {uid}",
                        )
                    )

    # geometry
    fig.add_trace(
        go.Scatter(
            x=x_exterior,
            y=y_exterior,
            mode="lines",
            line={"color": "red"},
            name="geometry",
        )
    )
    count_direction = ""
    for direction in [1, 2, 3, 4]:
        count = directions[directions["direction_number"] == direction].shape[0]
        count_direction += "Direction: " + str(direction) + ": " + str(count) + ". "
    fig.update_layout(
        title=f" Trajectories: {num_agents}. {count_direction}",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis={"scaleanchor": "y"},  # , range=[xmin, xmax]),
        yaxis={"scaleratio": 1},  # , range=[ymin, ymax]),
        showlegend=False,
    )
    return fig


def plot_time_series(density: pd.DataFrame, speed: pd.DataFrame, fps: int) -> go.Figure:

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            rf"$\mu= {np.mean(density.density):.2f}\; \pm {np.std(density.density):.2f}\; 1/m^2$",
            rf"$\mu= {np.mean(speed):.2f}\;\pm {np.std(speed):.2f}\; m/s$",
        ),
        horizontal_spacing=0.2,
    )

    fig.add_trace(
        go.Scatter(
            x=density.index / fps,
            y=density.density,
            line={"color": "blue"},
            marker={"color": "blue"},
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=speed.index / fps,
            y=speed,
            line={"color": "blue"},
            marker={"color": "blue"},
            mode="lines",
        ),
        row=1,
        col=2,
    )

    rmin = np.min(density.density) - 0.1
    rmax = np.max(density.density) + 0.1
    vmax = np.max(speed) + 0.1
    vmin = np.min(speed) - 0.1
    fig.update_layout(
        showlegend=False,
    )
    fig.update_yaxes(
        range=[rmin, rmax],
        title_text=r"$\rho\; /\; 1/m^2$",
        title_font={"size": 20},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[vmin, vmax],
        title_text=r"$v\; /\; m/s$",
        title_font={"size": 20},
        # scaleanchor="x",
        scaleratio=1,
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text=r"$t\; / s$", title_font={"size": 20}, row=1, col=2)
    fig.update_xaxes(title_text=r"$t\; / s$", title_font={"size": 20}, row=1, col=1)

    return fig


def plot_fundamental_diagram_all(density_dict, speed_dict) -> go.Figure:
    fig = go.Figure()

    rmax = -1
    vmax = -1

    colors_const = [
        "blue",
        "red",
        "green",
        "magenta",
        "black",
        "cyan",
        "yellow",
        "orange",
        "purple",
    ]
    marker_shapes = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x-thin",
        "triangle-up",
        "hexagon",
        "star",
        "pentagon",
    ]

    colors = []
    filenames = []
    for filename, color in zip(density_dict.keys(), colors_const):
        colors.append(color)
        filenames.append(filename)

    for i, (density, speed) in enumerate(
        zip(density_dict.values(), speed_dict.values())
    ):
        fig.add_trace(
            go.Scatter(
                x=density.density,
                y=speed.speed,
                marker={
                    "color": colors[i % len(color)],
                    "opacity": 0.5,
                    "symbol": marker_shapes[i % len(marker_shapes)],
                },
                mode="markers",
                name=f"{filenames[i%len(filenames)]}",
                showlegend=True,
            )
        )
        rmax = max(rmax, np.max(density))
        vmax = max(vmax, np.max(speed))
        vmin = min(vmax, np.min(speed))

    vmax += 0.05
    rmax += 0.05
    vmin -= 0.05

    # vmax = 2.0
    fig.update_yaxes(
        # range=[vmin, vmax],
        title_text=r"$v\; / \frac{m}{s}$",
        title_font={"size": 20},
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_xaxes(
        title_text=r"$\rho / m^{-2}$",
        title_font={"size": 20},
    )

    return fig


def plot_x_y(x, y, title, xlabel, ylabel, color, threshold=0):

    x = np.unique(x)
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"<b>{title}</b>"],
        x_title=xlabel,
        y_title=ylabel,
    )

    trace = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        showlegend=True,
        name=title,
        line={"width": 3, "color": color},
        fill="none",
    )

    fig.append_trace(trace, row=1, col=1)
    return trace, fig


def assign_direction_number(agent_data):
    """
    Assigns a direction number to each agent based on their main direction of motion.

    Parameters:
    - agent_data (DataFrame): A DataFrame with columns 'id', 'frame', 'x', 'y', representing
      agent IDs, frame numbers, and their positions at those frames.

    Returns:
    - A DataFrame with an additional 'direction_number' column.
    """
    # Group by agent ID and calculate the difference in position
    direction_numbers = []
    for agent_id, group in agent_data.groupby("id"):
        start_pos = group.iloc[0]  # Starting position
        end_pos = group.iloc[-1]  # Ending position

        delta_x = end_pos["x"] - start_pos["x"]
        delta_y = end_pos["y"] - start_pos["y"]

        # Determine primary direction of motion
        if abs(delta_x) > abs(delta_y):
            # Motion is primarily horizontal
            direction_number = (
                3 if delta_x > 0 else 4
            )  # East if delta_x positive, West otherwise
        else:
            # Motion is primarily vertical
            direction_number = (
                1 if delta_y > 0 else 2
            )  # North if delta_y positive, South otherwise

        direction_numbers.append((agent_id, direction_number))

    # Create a DataFrame from the direction numbers
    return pd.DataFrame(direction_numbers, columns=["id", "direction_number"])

    # Merge the direction DataFrame with the original agent_data DataFrame
    # result_df = pd.merge(agent_data, direction_df, on='id')



def show_fig(
    fig: Figure,
    figname: str,
    html: bool = False,
    write: bool = False,
    height: int = 500,
) -> None:
    """Workaround function to show figures having LaTeX-Code.

    Args:
        fig (Figure): A Plotly figure object to display.
        html (bool, optional): Flag to determine if the figure should be shown as HTML. Defaults to False.
        write (bool, optional): Flag to write the fig as a file and make a download button
        height (int, optional): Height of the HTML component if displayed as HTML. Defaults to 500.

    Returns:
        None
    """
    if not html:
        st.plotly_chart(fig)
    else:
        st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=height)

    fig.write_image(figname)


def download_file(figname, col=None):
    with open(figname, "rb") as pdf_file:
        if col is None:
            st.download_button(
                label="Download",
                data=pdf_file,
                file_name=figname,
                mime="application/octet-stream",
                help=f"Download {figname}",
            )
        else:
            col.download_button(
                label="Download",
                data=pdf_file,
                file_name=figname,
                mime="application/octet-stream",
                help=f"Download {figname}",
            )
