"""Init ui."""

from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_option_menu import option_menu


def setup_app() -> None:
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Madras Project",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/PedestrianDynamics/madras-data-app",
            "Report a bug": "https://github.com/PedestrianDynamics/madras-data-app//issues",
            "About": "# Field observation for Madras project.\n This is a tool to analyse "
            + "and visualise several field data of pedestrian dynamics during the festival of lights in 2022:\n\n"
            + ":flag-fr: - :flag-de: Germany.",
        },
    )


def init_app_looks() -> None:
    """
    Initializes the appearance of the application.

    This function sets up the sidebar with a GitHub repository badge, a DOI badge,
    and a logo image. It constructs the paths and URLs required for these elements
    and uses Streamlit's sidebar components to display them.

    - Displays a GitHub repository badge with a link to the repository.
    - Displays a DOI badge with a link to the DOI.
    - Displays a logo image from the assets directory.
    """
    path = Path(__file__)
    ROOT_DIR = path.parent.parent.parent.absolute()
    logo_path = ROOT_DIR / "data" / "assets" / "logo.png"
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/PedestrianDynamics/madras-data-app"
    repo_name = f"[![Repo]({gh})]({repo})"
    c1, c2 = st.sidebar.columns((1.2, 0.5))
    c2.markdown(repo_name, unsafe_allow_html=True)
    c1.write(
        "[![DOI](https://zenodo.org/badge/760394097.svg)](https://zenodo.org/doi/10.5281/zenodo.10694866)"
    )
    st.sidebar.image(str(logo_path), use_column_width=True)


def init_sidebar() -> Any:
    """Init sidebar and 5 tabs.

    To add more tabs, add the name of the tab and add an icon from
    https://icons.getbootstrap.com/
    """
    # Custom CSS to handle multi-line text alignment and indentation
    st.markdown(
        """
        <style>
        .nav-link {
            display: flex;
            align-items: center;
            white-space: pre-wrap; /* Allows text to wrap */
            text-align: left;
        }
        .nav-link div {
            margin-left: 10px; /* Adjust margin to align text with icon */
        }
        .nav-link div span {
            display: block;
            padding-left: 20px; /* Simulate tab space */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return option_menu(
        "Multi-agent modelling of dense crowd dynamics: Predict & Understand",
        [
            "About",
            "Map",
            "Trajectories",
            "Analysis",
            "Contacts",
            "Surveys",
            # "Explorer",
            "Geometry",
        ],
        icons=[
            "info-square",
            "pin-map",
            "people",
            "bar-chart-line",
            "exclamation-triangle",
            # "graph-up-arrow",
            "bi bi-clipboard2-data",
            "camera-reels-fill",
        ],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "gray", "font-size": "15px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
        },
    )
