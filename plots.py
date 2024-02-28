"""Plot functionalities for the app."""

import collections
import io
from typing import Dict, Optional, Tuple, TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedpy
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.graph_objs import Figure, Scatter
from plotly.subplots import make_subplots

plt.rcParams["text.usetex"] = True
st_column: TypeAlias = st.delta_generator.DeltaGenerator


def plot_trajectories(
    trajectory_data: pedpy.TrajectoryData,
    framerate: int,
    walkable_area: pedpy.WalkableArea,
    uid: Optional[float] = None,
    show_direction: Optional[float] = None,
) -> go.Figure:
    """Plot trajectories and geometry.

    framerate: sampling rate of the trajectories.
    """
    fig = go.Figure()
    c1, c2, c3 = st.columns((1, 1, 1))
    data = trajectory_data.data
    num_agents = len(np.unique(data["id"]))
    colors = {
        1: "green",
        2: "purple",
        3: "red",
        4: "blue",
    }
    dnames = {1: "North", 2: "South", 3: "East", 4: "West"}
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
        fig.add_trace(
            go.Scatter(
                x=[df["x"].iloc[0]],
                y=[df["y"].iloc[0]],
                marker={"color": "red", "symbol": "circle"},
                mode="markers",
                name=f"Start ID {uid}",
            )
        )

        for other, df in data.groupby("id"):
            if other != uid:
                fig.add_trace(
                    go.Scatter(
                        x=df["x"][::framerate],
                        y=df["y"][::framerate],
                        line={"color": "gray", "width": 0.1},
                        marker={"color": "gray"},
                        mode="lines",
                        name=f"ID {uid}",
                    )
                )

    else:
        for uid, df in data.groupby("id"):
            direction = directions.loc[directions["id"] == uid, "direction_number"].iloc[0]

            if show_direction is None:
                color_choice = colors[direction]
                fig.add_trace(
                    go.Scatter(
                        x=df["x"][::framerate],
                        y=df["y"][::framerate],
                        line={"color": color_choice, "width": 0.1},
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
                            line={"color": color_choice, "width": 0.3},
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
            line={"color": "black"},
            name="geometry",
        )
    )
    count_direction = ""
    ymin = -6
    ymax = 4
    for direction in [1, 2, 3, 4]:
        count = directions[directions["direction_number"] == direction].shape[0]
        # count_direction += f"<span style='color:{colors[direction]};'>Direction</span> " + str(direction) + ": " + str(count) + ". "
        count_direction += f"<span style='color:{colors[direction]};'> {dnames[direction]} {count}</span>."

    fig.update_layout(
        title=f" Trajectories: {num_agents}. {count_direction}",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis={"scaleanchor": "y"},
        yaxis={"scaleratio": 1},
        showlegend=False,
    )
    return fig


# mpl
def plot_trajectories_figure_mpl(
    trajectory_data: pedpy.TrajectoryData,
    walkable_area: pedpy.WalkableArea,
    with_colors: bool,
):
    """Plot trajectories and geometry mpl version.

    framerate: sampling rate of the trajectories.
    """
    fig, ax = plt.subplots()
    data = trajectory_data.data
    num_agents = len(np.unique(data["id"]))
    colors = {
        1: "green",
        2: "purple",
        3: "red",
        4: "blue",
    }
    dnames = {1: "North", 2: "South", 3: "East", 4: "West"}
    x_exterior, y_exterior = walkable_area.polygon.exterior.xy
    x_exterior = list(x_exterior)
    y_exterior = list(y_exterior)

    directions = assign_direction_number(data)
    for uid, df in data.groupby("id"):
        direction = directions.loc[directions["id"] == uid, "direction_number"].iloc[0]
        if with_colors:
            color_choice = colors[direction]
        else:
            color_choice = "gray"
        ax.plot(
            df["x"],
            df["y"],
            color=color_choice,
            lw=0.1,
            alpha=0.6,
        )
    # geometry
    ax.plot(
        x_exterior,
        y_exterior,
        color="black",
    )
    title_text = f"Trajectories: {num_agents}."
    for direction in [1, 2, 3, 4]:
        count = directions[directions["direction_number"] == direction].shape[0]
        title_text += f" {dnames[direction]} {count}."

    ax.set_title(title_text)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    ax.set_aspect("equal", "box")
    return fig


def plot_time_series(density: pd.DataFrame, speed: pd.DataFrame, fps: int) -> go.Figure:
    """Plot density and speed time series side-byside."""
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
    fig.update_xaxes(
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


def plt_plot_time_series(density: pd.DataFrame, speed: pd.DataFrame, fps: int) -> Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Plot density and speed time series side-byside."""
    # density
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(density.index / fps, density.density, color="blue", lw=1)
    title1 = rf"$\mu= {np.mean(density.density):.2f}\; \pm {np.std(density.density):.2f}\; 1/m^2$"
    ax1.set_title(title1)
    ax1.set_ylabel(r"$\rho\; /\; 1/m^2$")
    ax1.set_xlabel(r"$t\; /\;s$")
    rmin = np.min(density.density) - 0.1
    rmax = np.max(density.density) + 0.1
    ax1.set_ylim([rmin, rmax])
    ax1.grid(alpha=0.4)
    # speed
    ax2.plot(speed.index / fps, speed, color="gray", lw=1)
    title2 = rf"$\mu= {np.mean(speed):.2f}\; \pm {np.std(speed):.2f}\; m/s$"
    ax2.set_title(title2)
    ax2.set_ylabel(r"$v\; /\; m/s$")
    ax2.set_xlabel(r"$t\; /\;s$")
    vmax = np.max(speed) + 0.1
    vmin = np.min(speed) - 0.1
    ax2.set_ylim([vmin, vmax])
    ax2.grid(alpha=0.4)

    return fig1, fig2


def plot_fundamental_diagram_all(density_dict: Dict[str, pd.DataFrame], speed_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Plot fundamental diagram of all files."""
    fig = go.Figure()
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
        "triangle-down",
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

    for i, (density, speed) in enumerate(zip(density_dict.values(), speed_dict.values())):
        if isinstance(speed, pd.Series):
            y = speed
        else:
            y = speed.speed
        fig.add_trace(
            go.Scatter(
                x=density.density,
                y=y,
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


def plot_x_y(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, color: str) -> Tuple[Scatter, Figure]:
    """Plot two arrays and return trace and fig."""
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


def plot_fundamental_diagram_all_mpl(density_dict: dict, speed_dict: dict):
    """Plot fundamental diagram of all files using Matplotlib."""
    # Define colors and marker styles
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
        "o",
        "s",
        "D",
        "x",
        "^",
        "v",
        "h",
        "*",
        "p",
    ]  # Matplotlib marker styles

    fig, ax = plt.subplots()
    for i, ((filename, density), (_, speed)) in enumerate(zip(density_dict.items(), speed_dict.items())):
        if isinstance(speed, pd.Series):
            y = speed
        else:
            y = speed["speed"]  # Adjust this if 'speed' DataFrame structure is different

        ax.plot(
            density["density"],
            y,
            color=colors_const[i % len(colors_const)],
            alpha=0.5,
            linestyle="",
            marker=marker_shapes[i % len(marker_shapes)],
            label=filename,
        )

    ax.set_xlabel(r"$\rho / m^{-2}$", fontsize=20)
    ax.set_ylabel(r"$v\; / \frac{m}{s}$", fontsize=20)
    ax.legend(loc="best")

    return fig


def assign_direction_number(agent_data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a direction number to each agent based on their main direction of motion.

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
            direction_number = 3 if delta_x > 0 else 4  # East if delta_x positive, West otherwise
        else:
            # Motion is primarily vertical
            direction_number = 1 if delta_y > 0 else 2  # North if delta_y positive, South otherwise

        direction_numbers.append((agent_id, direction_number))

    return pd.DataFrame(direction_numbers, columns=["id", "direction_number"])


def show_fig(
    fig: Figure,
    figname: str = "default.pdf",
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
        st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=height)  # type: ignore
    if write:
        fig.write_image(figname)


def download_file(figname: str, col: Optional[st_column] = None, label: str = "") -> None:
    """Make download button for file."""
    with open(figname, "rb") as pdf_file:
        if col is None:
            st.download_button(
                type="primary",
                label=f"Download {label}",
                data=pdf_file,
                file_name=figname,
                mime="application/octet-stream",
                help=f"Download {figname}",
            )
        else:
            col.download_button(
                type="primary",
                label=f"Download {label}",
                data=pdf_file,
                file_name=figname,
                mime="application/octet-stream",
                help=f"Download {figname}",
            )


def get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY):
    """Return with, height and scale for background image."""
    scale = np.amin((geomaxX - geominX, geomaxY - geominY))
    scale_max = 20
    scale = min(scale_max, scale)
    scale = (1 - scale / scale_max) * 0.9 + scale / scale_max * 0.1
    # scale = 0.3
    w = (geomaxX - geominX) * scale
    h = (geomaxY - geominY) * scale
    return w, h, scale


def plot_trajectories_mpl(ax, data, scale: int = 1, shift_x: int = 0, shift_y: int = 0):
    """Plot data and update axis with matplotlib."""

    pid = data["id"].unique()
    for ped in pid:
        pedd = data[data["id"] == ped]
        ax.plot(
            (pedd["x"] - shift_x) * scale,
            (pedd["y"] - shift_y) * scale,
            "-",
            color="black",
            lw=0.1,
        )


def bg_img(data, geominX: float, geomaxX: float, geominY: float, geomaxY: float):
    """Plot trajectories and create a background image."""
    print("enter bg_img")
    width, height, scale = get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.set_dpi(100)
    ax.set_xlim((0, width))
    ax.set_ylim((0, height))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # inv = ax.transData.inverted()
    plot_trajectories_mpl(ax, data, scale, geominX, geominY)
    major_ticks_top_x = np.linspace(0, width, 5)
    major_ticks_top_y = np.linspace(0, height, 5)
    minor_ticks_top_x = np.linspace(0, width, 40)
    minor_ticks_top_y = np.linspace(0, height, 40)
    major_ticks_bottom_x = np.linspace(0, width, 20)
    major_ticks_bottom_y = np.linspace(0, height, 20)
    ax.set_xticks(major_ticks_top_x)
    ax.set_yticks(major_ticks_top_y)
    ax.set_xticks(minor_ticks_top_x, minor=True)
    ax.set_yticks(minor_ticks_top_y, minor=True)
    ax.grid(which="major", alpha=0.6)
    ax.grid(which="minor", alpha=0.3)
    ax.set_xticks(major_ticks_bottom_x)
    ax.set_yticks(major_ticks_bottom_y)
    ax.grid()
    bg_img = fig2img(fig)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    img_width, img_height = bbox.width * fig.dpi, bbox.height * fig.dpi
    # inv = ax.transData.inverted()
    return bg_img, img_width, img_height, fig.dpi, scale


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def draw_rects(canvas, img_height, dpi, scale, boundaries):
    geominX, geomaxX, geominY, geomaxY = boundaries
    rect_points_xml = collections.defaultdict(dict)
    if canvas.json_data is not None:
        objects = pd.json_normalize(canvas.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")

        if not objects.empty:
            rects = objects[objects["type"].values == "rect"]
            if not rects.empty:
                (
                    rfirst_x,
                    rfirst_y,
                    rsecond_x,
                    rsecond_y,
                    rthird_x,
                    rthird_y,
                    rfirth_x,
                    rfirth_y,
                ) = process_rects(rects, img_height)
                i = 0
                for x1, x2, x3, x4, y1, y2, y3, y4 in zip(
                    rfirst_x,
                    rsecond_x,
                    rthird_x,
                    rfirth_x,
                    rfirst_y,
                    rsecond_y,
                    rthird_y,
                    rfirth_y,
                ):
                    rect_points_xml[i]["x"] = [
                        x1 / scale / dpi + geominX,
                        x2 / scale / dpi + geominX,
                        x3 / scale / dpi + geominX,
                        x4 / scale / dpi + geominX,
                    ]
                    rect_points_xml[i]["y"] = [
                        y1 / scale / dpi + geominY,
                        y2 / scale / dpi + geominY,
                        y3 / scale / dpi + geominY,
                        y4 / scale / dpi + geominY,
                    ]
                    i += 1

    return rect_points_xml


def process_rects(rects, h_dpi):
    """Transform rect's points to world coordinates.

    :param rects: the object for rectangle
    :param h_dpi: height of image in dpi
    :returns: 4 points a 2 coordinates -> 8 values

    """
    left = np.array(rects["left"])
    top = np.array(rects["top"])
    scale_x = np.array(rects["scaleX"])
    scale_y = np.array(rects["scaleY"])
    height = np.array(rects["height"]) * scale_y
    width = np.array(rects["width"]) * scale_x
    angle = -np.array(rects["angle"]) * np.pi / 180
    # angle = -np.radians(rects["angle"])
    # center_x = left + width / 2
    # center_y = top - height / 2
    # first
    first_x = left
    first_y = h_dpi - top

    # second
    x1, y1 = rotate(width, 0, angle)
    second_x = first_x + x1  # width
    second_y = first_y + y1
    # third
    x1, y1 = rotate(0, -height, angle)
    third_x = first_x + x1
    third_y = first_y + y1  # - height
    # forth
    x1, y1 = rotate(width, -height, angle)
    firth_x = first_x + x1  # width
    firth_y = first_y + y1
    # rotate

    return (
        first_x,
        first_y,
        second_x,
        second_y,
        third_x,
        third_y,
        firth_x,
        firth_y,
    )


def rotate(x, y, angle):
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)
