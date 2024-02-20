"""Main entry point to the data visualisator for MADRAS project."""

import map_tab
import streamlit as st
from utilities import setup_app

if __name__ == "__main__":
    setup_app()
    map_tab.call_main()
