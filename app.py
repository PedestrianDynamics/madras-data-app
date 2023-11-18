import json
from dataclasses import dataclass

import folium
import streamlit as st
from streamlit_folium import st_folium


@dataclass
class Camera:
    location: tuple
    url: str


def load_cameras_from_json(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
        cameras = {}
        for name, info in data.items():
            cameras[name] = Camera(location=tuple(info["location"]), url=info["url"])
        return cameras


def create_map(center, zoom_start=16):
    m = folium.Map(location=center, zoom_start=zoom_start)
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

    for polygon, layer in zip(polygons, polygon_layers):
        polygon.add_to(layer)

    for (name, camera), layer in zip(cameras.items(), camera_layers):
        print(name, camera)
        coords = camera.location
        tooltip = name
        folium.Marker(location=coords, tooltip=tooltip).add_to(layer)

    folium.LayerControl().add_to(m)
    return m


def setup():
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


def main(cameras):
    center = [45.76322690683106, 4.83001470565796]  # Coordinates for Lyon, France
    m = create_map(center)
    map_data = st_folium(m, width=800, height=800)
    placeholder = st.sidebar.empty()
    video_name = map_data.get("last_object_clicked_tooltip")
    if video_name:
        placeholder.info(f"Selected Camera: {video_name}")
        camera = cameras.get(video_name)
        if camera:
            st.sidebar.video(camera.url)
        else:
            st.sidebar.error(f"No video linked to {video_name}.")
    else:
        st.sidebar.error("No camera selected.")


if __name__ == "__main__":
    setup()
    cameras = load_cameras_from_json("cameras.json")
    main(cameras)
