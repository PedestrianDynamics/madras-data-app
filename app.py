"""Map visualisator for Madras project."""
import json
from dataclasses import dataclass

import folium
import streamlit as st
from streamlit_folium import st_folium
from typing import Dict, Tuple


@dataclass
class Camera:
    """Data class representing a camera with its location and video URL."""

    location: tuple
    url: str
    name: str
    field: list


tile_layers = {
    "Open Street Map": "openstreetmap",
    "CartoDB Positron": "CartoDB positron",
    "CartoDB Dark_Matter": "CartoDB dark_matter",
}


def load_cameras_from_json(file_path: str) -> Dict[str, Camera]:
    """
    Load camera data from a JSON file and return a dict. of Camera objects.

    Args:
    file_path (str): The path to the JSON file.

    Returns:
    Dict[str, Camera]: A dictionary mapping camera names to Camera objects.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {file_path}")
        return {}

    cameras = {}
    for key, info in data.items():
        try:
            # Ensure the data structure is as expected
            location = tuple(info["location"])
            url = info["url"]
            name = info["name"]
            field = info["field"]
            cameras[key] = Camera(location=location, url=url, name=name, field=field)

        except KeyError as e:
            # Handle missing keys in the data
            st.error(f"Missing key in camera data: {e}")
            continue  # Skip this camera and continue with the next
        except Exception as e:
            # Catch any other unexpected errors
            st.error(f"Error processing camera data: {e}")
            continue

    return cameras


def create_map(center: Tuple[float, float], tile_layer, zoom: int = 16) -> folium.Map:
    """Create a folium map with camera markers and polygon layers.

    Args:
    center (Tuple[float, float]): The center of the map (latitude, longitude).
    zoom_start (int): The initial zoom level of the map.

    Returns:
    folium.Map: A folium map object.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles=tile_layer)

    camera_layers = []
    for name in cameras.keys():
        camera_layers.append(
            folium.FeatureGroup(name=name, show=True).add_to(m),
        )

    polygon_layers = [
        folium.FeatureGroup(name="Place des Taurraux", show=True, overlay=True).add_to(
            m
        ),
        folium.FeatureGroup(
            name="Place Saint-Jean",
            show=True,
            overlay=True,
        ).add_to(m),
    ]
    polygons = [
        folium.PolyLine(
            locations=[
                [45.760673, 4.826871],
                [45.761122, 4.827051],
                [45.761350, 4.825956],
                [45.760983, 4.825772],
                [45.760673, 4.826871],
            ],
            tooltip="Place Saint-Jean",
            fill_color="blue",
            color="red",
            fill_opacity=0.2,
            fill=True,
        ),
        folium.PolyLine(
            locations=[
                [45.767407, 4.834173],
                [45.767756, 4.834065],
                [45.767658, 4.83280],
                [45.767146, 4.832846],
                [45.767407, 4.834173],
            ],
            tooltip="Place des Terraux",
            fill_color="blue",
            color="red",
            fill_opacity=0.2,
            fill=True,
        ),
    ]

    vision_fields = {}
    for name, camera in cameras.items():
        vision_fields[name] = folium.PolyLine(
            locations=camera.field,
            tooltip="field " + name,
            fill_color="blue",
            color="red",
            fill_opacity=0.1,
            fill=True,
        )

    for polygon, layer in zip(polygons, polygon_layers):
        polygon.add_to(layer)

    for (key, camera), layer in zip(cameras.items(), camera_layers):
        coords = camera.location
        tooltip = f"{key}: {camera.name}"
        folium.Marker(location=coords, tooltip=tooltip).add_to(layer)
        vision_fields[key].add_to(layer)

    # folium.FitOverlays().add_to(m)
    folium.LayerControl().add_to(m)
    return m


def setup() -> None:
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


def main(cameras: Dict[str, Camera], selected_layer) -> None:
    """Implement the main logic of the app.

    Args:
    cameras (Dict[str, Camera]): A dictionary of Camera objects.
    """
    center = [45.76322690683106, 4.83001470565796]  # Coordinates for Lyon, France
    m = create_map(center, tile_layer=tile_layers[selected_layer])
    map_data = st_folium(m, width=1600, height=1200)
    st.info(map_data)
    st.sidebar.info(
        f"[{map_data['last_clicked']['lat']}, {map_data['last_clicked']['lng']}]"
    )
    placeholder = st.sidebar.empty()
    video_name = map_data.get("last_object_clicked_tooltip")
    if video_name:
        placeholder.info(f"Selected Camera: {video_name}")
        camera = cameras.get(video_name.split(":")[0])
        if camera:
            st.sidebar.video(camera.url)
        else:
            st.sidebar.error(f"No video linked to {video_name}.")
    else:
        st.sidebar.error("No camera selected.")


if __name__ == "__main__":
    setup()
    cameras = load_cameras_from_json("cameras.json")
    selected_layer = st.selectbox("Choose a Map Style:", list(tile_layers.keys()))
    main(cameras, selected_layer)
