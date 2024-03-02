"""Documentation texts for the app."""

from typing import List

import streamlit as st

from datafactory import Direction


def flow(directions: List[Direction]) -> None:
    """Write documentation text for NT-diagram."""
    st.write(
        r"""
        The N-t diagram shows how many pedestrian have crossed the measurement line at a specific time.

        PedPy Computes the frame-wise cumulative number of pedestrians passing the line. For more information see [PedPy-documentation](https://pedpy.readthedocs.io/en/latest/api/methods.html#flow_calculator.compute_n_t).

        **Measurement lines:**
        """
    )

    table_data = {"Line": [], "Point": [], "Coordinates": []}  # type: ignore
    for direction in directions:
        line = direction.line.line
        for point_number, coordinate in enumerate(line.coords, start=1):
            table_data["Line"].append(f"{direction.info.name}")
            table_data["Point"].append(f"Point {point_number}")
            formatted_coordinate = f"({coordinate[0]:.2f}, {coordinate[1]:.2f})"
            table_data["Coordinates"].append(f"{formatted_coordinate}")

    # Display as a table
    st.table(data=table_data)


def density_speed() -> None:
    """Write documentation text for density-speed calculations."""
    st.write(
        r"""
            ## Density:
            The measurement method is as follows:
            $$
            \rho = \frac{N}{A},
            $$
            where $N$ is the number of agents in the actual frame and $A$ the size of the observed area.
            See :point_right: [pedpy compute_classic_density](https://pedpy.readthedocs.io/en/stable/api/methods.html#density_calculator.compute_classic_density)
            ## Speed
            The calculation of speed is based on the displacement in the $x$ and $y$ directions over time.
            """
    )
    st.latex(
        r"""
        \begin{equation}
        v = \frac{\tilde X(t + \Delta t) - \tilde X(t-\Delta t)}{2\Delta t}.
        \end{equation}
        """
    )
    st.write(
        """
        See:point_right: [pedpy individual speed](https://pedpy.readthedocs.io/en/stable/user_guide.html#individual-speed).
        """
    )


def about() -> None:
    """Write About text."""
    text = """


    ## Overview
    The [MADRAS-project](https://www.madras-crowds.eu/) is a collaborative cooperation funded by [ANR](https://anr.fr) :flag-fr: and [DFG](htpps://dfg.de) :flag-de:, aims to develop innovative agent-based models to predict and understand dense crowd dynamics and to apply these models in a large-scale case study.
    This app offers a visualisation of data collection of the festival of lights in 2022, a distinguished open-air event that draws nearly two million visitors over four days.
    """
    st.markdown(text)
    st.image("images/fcaym-FdL22.png", caption="Festival of Lights in Lyon 2022.")

    text2 = """
    This app is part of the MADRAS project, which focuses on collecting and analyzing videos of crowded scenes during the festival. The primary goal is to extract valuable pedestrian dynamics measurements to enhance our understanding of crowd behaviors during such large-scale events.

    ## Data Extraction and Analysis
    The app provides an intuitive interface for users to interactively explore the collected data, understand crowd dynamics, and extract insights on pedestrian behaviors.


    - **Trajectory Plotting**: Allows users to plot and visualize the trajectories of visitors moving through the event space.
    - **Density Calculation**: Interactive tools to calculate and analyze crowd density in different areas of the festival.
    - **Speed and Flow Measurement**: Capabilities to measure and understand the average speed and flow of the crowd, aiding in the calibration and testing of print()edestrian models.
    - **Map Visualization**: An interactive map of the event, enabling users to visually explore the areas of interest and the locations of cameras.
    """
    st.markdown(text2)
    text3 = """
    Selected scenes of the Festival of Lights are also used as reference scenarios for numerical simulations. The collection of crowd videos is done in the strict respect of the privacy and personal data protection of the filmed visitors. The videos are processed anonymously, without distinguishing the filmed persons by any criteria. All pedestrian dynamics data (as well as the models and simulation software) will be publicly available at the end of the project.
    """
    st.markdown(text3)
    st.image(
        "images/fbppj-FestivalOfLights2-min.png",
        caption="Emplacement of cameras for the video recording during the Festival of Lights 2022.",
    )
