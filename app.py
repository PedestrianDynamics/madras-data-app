"""Main entry point to the data visualisator for MADRAS project."""

import streamlit as st

from src.docs import docs
from src.classes.datafactory import init_session_state
from src.helpers.log_config import setup_logging 
from src.ui.ui import setup_app, init_sidebar, setup_app, init_app_looks

from src.tabs.traj_tab import run_tab2
from src.tabs.map_tab import call_main
from src.tabs.analysis_tab import run_tab3
setup_logging()
if __name__ == "__main__":
    setup_app()
    selected_tab = init_sidebar()
    init_app_looks()
    init_session_state()

    if selected_tab == "About":
        docs.about()

    if selected_tab == "Map":
        call_main()

    if selected_tab == "Trajectories":
        msg = st.empty()
        file_name_to_path = {path.split('/')[-1]: path for path in st.session_state.files}
        filename = str(st.selectbox(":open_file_folder: **Select a file**", file_name_to_path))
        st.session_state.selected_file = file_name_to_path[filename]
        run_tab2(file_name_to_path[filename], msg)

    if selected_tab == "Analysis":
        run_tab3()

    # if selected_tab == "Contacts":
    #     contacts_tab.call_main()
