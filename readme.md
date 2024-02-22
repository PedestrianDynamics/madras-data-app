# MADRAS Project Streamlit App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://madras-data-app.streamlit.app/)

This application presents an interactive visualization of field observations conducted by 
the MADRAS team during the Festival of Lights in Lyon, 2022, on December 8th. 
It provides users with an engaging means to explore and analyze the data collected.



## Overview
The **MADRAS Project Streamlit App** is analyse the field observations performed in the [Festival of Lights](https://www.madras-crowds.eu/Festival-of-Lights-Lyon-.h.htm) in Lyon, a distinguished open-air event that draws nearly two million visitors over four days. 
This app is part of the [MADRAS project](https://www.madras-crowds.eu/), which focuses on developing models and on collecting and analyzing videos of crowded scenes during the festival. The primary goal is to extract valuable pedestrian dynamics measurements to enhance our understanding of crowd behaviors during such large-scale events.

## Features

### Data Extraction and Analysis
- **Trajectory Plotting**: Allows users to plot and visualize the trajectories of visitors moving through the event space.
- **Density Calculation**: Interactive tools to calculate and analyze crowd density in different areas of the festival.
- **Speed and Flow Measurement**: Capabilities to measure and understand the average speed and flow of the crowd, aiding in the calibration and testing of pedestrian models.
- **Map Visualization**: An interactive map of the event, enabling users to visually explore the areas of interest and the locations of cameras.


## Local Execution Guide

The app can be used by clicking on this link [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://madras-data-app.streamlit.app/).

However, for optimal performance, especially for tasks that demand significant computing resources, consider operating the app on your local machine.

To set up, follow these steps after downloading the repository:


1. Environment Setup (Highly Recommended)

Create and activate a virtual environment to manage dependencies efficiently:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Dependencies

Ensure all required packages are installed:


```bash
pip install -r requirements.txt
```

3. Launch the App

Start the app with the following command:

```bash
streamlit run app.py
```

This process establishes a local environment tailored for running the app's intensive computations efficiently.
