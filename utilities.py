import streamlit as st


def setup_app() -> None:
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Madras Project",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Interactive Map with Multiple Layers")
    st.markdown(
        """
    **Layer Selection:**
    Use the layer control button in the top right corner of the map to toggle different layers. 
    You can select video overlays, camera markers, and other features from this control panel.
    """,
        unsafe_allow_html=True,
    )
