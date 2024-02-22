import streamlit as st


def flow(measurement_lines):
    st.write(
        rf"""
        The N-t diagram shows how many pedestrian have crossed the measurement line at a specific time.

        Measurementlines are:
        - Left: {measurement_lines[0].line}
        - Top:  {measurement_lines[1].line}
        - Right:  {measurement_lines[2].line}
        - Buttom:  {measurement_lines[3].line}
        """
    )


def density_speed():
    st.write(
        r"""
            ## Density:
            The measurement method is as follows:
            $$
            \rho = \frac{N}{A},
            $$
            where $N$ is the number of agents in the actual frame and $A$ the size of the observed area.
        
            ## Speed
            The calculation of speed is based on the displacement in the $x$ and $y$ directions over time. The method involves the following steps:
            - Calculate Displacements: The displacement in both the $x$ and $y$ directions is calculated as the difference between successive positions, accounting for displacement over a specified number of frames ($\Delta t$). This is done separately for each entity, identified by its $id$. The mathematical expressions for these displacements are:
            """
    )
    st.latex(
        r"""
        \begin{align*}
        \Delta x &= x(t + \Delta t) - x(t). \\
        \Delta y &= y(t + \Delta t) - y(t).
        \end{align*}
        """
    )
    st.write(
        """
        where $\Delta x$ and $\Delta y$ represent the displacements in the $x$ and $y$ directions, respectively, and $\Delta t$ is the difference in frame indices used for the calculation.

        - Compute Distance Traveled: The distance traveled between the frames is computed using the Pythagorean theorem, which combines the displacements in both directions:
        $$
        \\text{distance} = \sqrt{\Delta x^2 + \Delta y^2}.
        $$
        - Calculate Speed: Finally, the speed is calculated as the ratio of the distance traveled to the time
        """
    )
    st.latex(
        r"""
        \begin{equation}
        \text{speed} = \frac{\text{distance}}{\Delta t}
        \end{equation}
        """
    )
    st.write(
        """
        This yields the speed of each entity between the specified frames, taking into account the displacements in both spatial dimensions.
        """
    )


def about():
    text = """
    # Multi-agent modelling of dense crowd dynamics: Predict & Understand (MADRAS)
    
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
