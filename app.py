"""Main entry point to the data visualisator for MADRAS project."""

import streamlit as st

import analysis_tab
import datafactory
import docs
import map_tab
import traj_tab
import ui
from utilities import is_running_locally
import logging

# Basic configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
if __name__ == "__main__":
    ui.setup_app()
    selected = ui.init_sidebar()
    ui.init_app_looks()
    datafactory.init_session_state()

    if selected == "About":
        docs.about()

    if selected == "Map":
        map_tab.call_main()

    if selected == "Trajectories":
        msg = st.empty()
        filename = str(
            st.selectbox(":open_file_folder: **Select a file**", st.session_state.files)
        )
        st.session_state.selected_file = filename
        traj_tab.run_tab2(filename, msg)

    if selected == "Analysis":
        analysis_tab.run_tab3()
