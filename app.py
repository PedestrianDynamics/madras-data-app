"""Main entry point to the data visualisator for MADRAS project."""

import streamlit as st
import ui


import map_tab
import datafactory

if __name__ == "__main__":
    ui.setup_app()
    ui.init_app_looks()
    datafactory.init_session_state()

    tab0, tab1, tab2, tab3 = ui.init_sidebar()
    with tab0:
        st.info("More info later....")

    with tab1:
        map_tab.call_main()

    with tab2:
        activate_tab2 = st.toggle("Activate", value=False)
        if activate_tab2:
            filename = st.selectbox("Select a file:", st.session_state.files)
            st.session_state.selected_file = filename
            with st.status(f"Loading {filename}"):
                data = datafactory.load_file(filename)

            st.dataframe(data.data)

    with tab3:
        st.warning("in progress ...")
        st.info(st.session_state.selected_file)
