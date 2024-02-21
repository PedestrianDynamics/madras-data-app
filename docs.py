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
