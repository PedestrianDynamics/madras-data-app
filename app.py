"""Main entry point to the data visualisator for MADRAS project."""

import streamlit as st
import ui


import map_tab
import datafactory
import traj_tab
import analysis_tab
import docs

if __name__ == "__main__":
    ui.setup_app()
    ui.init_app_looks()
    datafactory.init_session_state()

    tab4, tab1, tab2, tab3 = ui.init_sidebar()

    # Map
    with tab1:
        map_tab.call_main()

    # Trajectories
    with tab2:
        msg = st.empty()
        activate_tab2 = st.toggle("Activate", key="tab2", value=False)
        if activate_tab2:
            filename = str(st.selectbox("Select a file:", st.session_state.files))
            st.session_state.selected_file = filename
            traj_tab.run_tab2(filename, msg)

    # Analysis
    with tab3:
        activate_tab3 = st.toggle("Activate", key="tab3", value=False)
        if activate_tab3:
            analysis_tab.run_tab3()
    # Info
    with tab4:
        docs.about()
