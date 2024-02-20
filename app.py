"""Main entry point to the data visualisator for MADRAS project."""

import streamlit as st
import ui


import map_tab
import datafactory
import show_data

if __name__ == "__main__":
    ui.setup_app()
    ui.init_app_looks()
    datafactory.init_session_state()

    tab1, tab2, tab3, tab4 = ui.init_sidebar()

    # Map
    with tab1:
        map_tab.call_main()

    # Trajectories
    with tab2:
        msg = st.empty()
        activate_tab2 = st.toggle("Activate", value=False)
        if activate_tab2:
            filename = st.selectbox("Select a file:", st.session_state.files)
            st.session_state.selected_file = filename
            show_data.run_tab2(filename, msg)

    # Analysis
    with tab3:
        st.warning("in progress ...")
        st.info(st.session_state.selected_file)

    # Info
    with tab4:
        st.info("More info later....")
