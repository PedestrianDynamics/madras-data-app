"""Init ui and download AppData if not existing."""

from pathlib import Path

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
            "About": "# Field observation for Madras project.\n This is a tool to analyse and visualise several field data of pedestrian dynamics during the festival of lights in 2022:\n\n :flag-fr: - :flag-de: Germany.",
        },
    )
    # st.title("Madras project: Festival of light 2022")


def init_app_looks() -> None:
    path = Path(__file__)
    ROOT_DIR = path.parent.absolute()

    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/PedestrianDynamics/madras-data-app"
    repo_name = f"[![Repo]({gh})]({repo})"
    c1, c2 = st.sidebar.columns((1.2, 0.5))
    c2.markdown(repo_name, unsafe_allow_html=True)
    c1.write("[![DOI](https://zenodo.org/badge/760394097.svg)](https://zenodo.org/doi/10.5281/zenodo.10694866)")
    st.sidebar.image(f"{ROOT_DIR}/logo.png", use_column_width=True)


def init_sidebar():
    """Init sidebar and 4 tabs."""
    return option_menu(
        "Multi-agent modelling of dense crowd dynamics: Predict & Understand",
        ["About", "Map", "Trajectories", "Analysis"],
        icons=["info-square", "pin-map", "people", "bar-chart-line"],
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
