"""Main entry point to the data visualisator for MADRAS project."""

import map_tab
import streamlit as st
import ui

if __name__ == "__main__":
    ui.setup_app()
    ui.init_app_looks()
    tab1, tab2, tab3 = ui.init_sidebar()

    with tab1:
        map_tab.call_main()
