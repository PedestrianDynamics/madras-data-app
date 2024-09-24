"""Main entry point to the data visualisator for MADRAS project."""

import streamlit as st

from src.classes.datafactory import init_session_state
from src.docs import docs
from src.helpers.log_config import setup_logging
from src.tabs.analysis_tab import run_tab3
from src.tabs.contacts_tab import run_tab_contact
from src.tabs.geometry_tab import run_tab_animation

# from src.tabs.explorer import run_explorer
from src.tabs.map_tab import run_tab_map
from src.tabs.survey_tab import run_tab_survey
from src.tabs.traj_tab import run_tab2
from src.ui.ui import init_app_looks, init_sidebar, setup_app

setup_logging()
if __name__ == "__main__":
    setup_app()
    selected_tab = init_sidebar()
    init_app_looks()
    init_session_state()

    if selected_tab == "About":
        docs.about()

    if selected_tab == "Map":
        run_tab_map()

    if selected_tab == "Trajectories":
        msg = st.empty()
        file_name_to_path = {path.split("/")[-1]: path for path in st.session_state.files}
        filename = str(st.selectbox(":open_file_folder: **Select a file**", file_name_to_path))
        st.session_state.selected_file = file_name_to_path[filename]
        run_tab2(file_name_to_path[filename], msg)

    if selected_tab == "Analysis":
        run_tab3()

    if selected_tab == "Contacts":
        run_tab_contact()

    if selected_tab == "Surveys":
        run_tab_survey()

    # if selected_tab == "Explorer":
    #     run_explorer()

    if selected_tab == "Geometry":
        file_name_to_path = {path.split("/")[-1]: path for path in st.session_state.files}
        filename = str(st.selectbox(":open_file_folder: **Select a file**", file_name_to_path))
        st.session_state.selected_file = file_name_to_path[filename]
        run_tab_animation(file_name_to_path[filename])
