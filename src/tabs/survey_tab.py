"""This module contains the functionality for the survey tab in the application."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objects import Figure


def histogram_survey(df_survey: pd.DataFrame, remove_outlier: bool) -> Figure:
    """
    Generate a histogram visualization of group sizes from a survey DataFrame.

    Parameters:
        df_survey (pd.DataFrame): The survey data containing group sizes.
        remove_outlier (bool): Flag to indicate whether to remove the maximum value as an outlier.
    Returns:
        Figure: A Plotly Figure object containing the histogram visualization.
    The function performs the following steps:
        1. Extracts the 'Children' and 'total' columns from the DataFrame.
        2. Optionally removes the maximum value from the 'total' column if outliers should be excluded.
        3. Initializes a Plotly Figure and adds histogram traces for 'Children' and 'total' categories.
        4. Updates the hover template with custom data for better interactivity.
        5. Configures the layout for improved visualization, including titles and axis labels.
    """
    # Extract values from the DataFrame
    values_children = df_survey["Children"].fillna(0).tolist()
    values_both = df_survey["total"].tolist()

    # Remove the maximum value if outliers should be excluded
    if remove_outlier and values_both:
        max_value = max(values_both)
        values_both = [x for x in values_both if x != max_value]

    # Initialize the figure
    fig = go.Figure()

    # Add histogram traces for each category
    fig.add_trace(go.Histogram(x=values_children, name="Children", marker_color="pink"))
    fig.add_trace(
        go.Histogram(x=values_both, name="Adults and children", marker_color="blue")
    )

    # Create a DataFrame for custom data in hover templates
    data = pd.DataFrame(
        {
            "Categories": ["Adults and children"] * len(values_both)
            + ["Children"] * len(values_children)
        }
    )

    # Update hover template with custom data
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "<b>Counts</b>: %{y}",
                "<b>Group Size</b>: %{x}",
                "<b>Category</b>: %{customdata[0]}",
                "<extra></extra>",
            ]
        ),
        customdata=data[["Categories"]].values,
    )

    # Update layout for better visualization
    fig.update_layout(
        title={"text": "Distribution of Group Sizes", "font_size": 28},
        width=1000,
        height=700,
        xaxis={
            "title": {"text": "Group Size", "font_size": 30, "font_color": "black"},
            "tickfont_size": 30,
            "tickfont_color": "black",
        },
        yaxis={
            "title": {"text": "Counts", "font_size": 30, "font_color": "black"},
            "tickfont_size": 30,
            "tickfont_color": "black",
        },
        legend={"font_size": 30, "font_color": "black"},
    )

    return fig


def main() -> None:
    """
    Visualize survey results.

    This function performs the following steps:
    1. Determines the path to the survey results CSV file and the pickle directory.
    2. Checks if a pickle file of the survey results exists:
       - If it exists, loads the survey results from the pickle file.
       - If it does not exist, reads the survey results from the CSV file, fills missing values,
         and saves it as a pickle file.
    3. Provides a sidebar option to remove outliers from the survey results.
    4. Generates and displays a histogram of the survey results.
    5. Provides a sidebar button to download the histogram as a PDF file.
    """
    path = Path(__file__).resolve()
    survey_path = (
        path.parent.parent.parent.absolute() / "data" / "surveys" / "survey_results.csv"
    )
    path_pickle = path.parent.parent.parent.absolute() / "data" / "pickle"

    # Check if survey.pkl exists, if not create it, else load it
    pickle_survey_path = path_pickle / "survey_results.pkl"
    if Path(path_pickle / "survey_results.pkl").exists():
        df_survey = pd.read_pickle(pickle_survey_path)
    else:
        df_survey = pd.read_csv(survey_path, sep=";")
        df_survey["Adults"] = df_survey["Adults"].fillna(0)
        df_survey["Children"] = df_survey["Children"].fillna(0)
        df_survey.to_pickle(pickle_survey_path)

    # Sidebar remove outlier button for the survey
    remove_outlier = st.sidebar.checkbox("Remove outlier", value=True)
    # Histogram of the survey results
    fig = histogram_survey(df_survey, remove_outlier)
    st.plotly_chart(fig)
    # Streamlit button in the sidebar to download the graph in PDF format
    st.sidebar.download_button(
        label="Download Survey Histogram",
        data=fig.to_image(format="pdf"),
        file_name="survey_results.pdf",
    )


def run_tab_survey() -> None:
    """
    Execute the main function for the survey tab.

    This function serves as the entry point for running the survey tab
    functionality within the application. It calls the main() function
    to initiate the necessary processes.
    """
    main()
