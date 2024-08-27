import streamlit as st
import cv2
import torch
import numpy as np
import random
from math import radians, cos, sin, sqrt, atan2

# YOLOv5 model load
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# Haversine function to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Generate random locations
def generate_random_location():
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon

# Parking areas (coordinates for both videos)
areas_parking1 = {
    'area1': [(52,364),(30,417),(73,412),(88,369)],
    'area2': [(105,353),(86,428),(137,427),(146,358)],
    'area3': [(159,354),(150,427),(204,425),(203,353)],
    'area4': [(217,352),(219,422),(273,418),(261,347)],
    'area5': [(274,345),(286,417),(338,415),(321,345)],
    'area6': [(336,343),(357,410),(409,408),(382,340)],
    'area7': [(396,338),(426,404),(479,399),(439,334)],
    'area8': [(458,333),(494,397),(543,390),(495,330)],
    'area9': [(511,327),(557,388),(603,383),(549,324)],
    'area10': [(564,323),(615,381),(654,372),(596,315)],
    'area11': [(616,316),(666,369),(703,363),(642,312)],
    'area12': [(674,311),(730,360),(764,355),(707,308)]
}

areas_easy1 = {
    'area11': [(616,316),(666,369),(703,363),(642,312)],
    'area12': [(674,311),(730,360),(764,355),(707,308)]
}

def process_frame(frame, areas):
    img = cv2.resize(frame, (1020, 500))
    results = model(img)
    detections = results.pandas().xyxy[0]

    area_occupied = {area: False for area in areas}

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = row['name']

        if class_name == 'car':
            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
            for area_name, area_coords in areas.items():
                if cv2.pointPolygonTest(np.array(area_coords, np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                    area_occupied[area_name] = True

    for area_name, area_coords in areas.items():
        color = (0, 0, 255) if area_occupied[area_name] else (0, 255, 0)
        cv2.polylines(img, [np.array(area_coords, np.int32)], True, color, 2)
        cv2.putText(img, area_name, area_coords[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    occupied_spaces = sum(area_occupied.values())
    available_spaces = len(areas) - occupied_spaces
    cv2.putText(img, f'Available: {available_spaces}/{len(areas)}', (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return img, available_spaces, len(areas)

# Streamlit UI
st.title("Parking Lot Availability and Closest Location Finder")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "Parking Lot 1 Video", "Parking Lot 2 Video"])

if page == "Home":
    # Generate random locations for the user and two parking lots
    user_location = generate_random_location()
    parking_lot1 = generate_random_location()
    parking_lot2 = generate_random_location()

    # Calculate distances
    distance1 = haversine(user_location[0], user_location[1], parking_lot1[0], parking_lot1[1])
    distance2 = haversine(user_location[0], user_location[1], parking_lot2[0], parking_lot2[1])

    # Display user location
    st.write(f"User Location: {user_location}")

    # Create buttons for parking lots
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"Parking Lot 1 (Distance: {distance1:.2f} km)"):
            st.write(f"Parking Lot 1 Location: {parking_lot1}")
            st.write(f"Distance: {distance1:.2f} km")

    with col2:
        if st.button(f"Parking Lot 2 (Distance: {distance2:.2f} km)"):
            st.write(f"Parking Lot 2 Location: {parking_lot2}")
            st.write(f"Distance: {distance2:.2f} km")

    # Determine and display closest parking lot
    if distance1 < distance2:
        closest_parking_lot = parking_lot1
        closest_distance = distance1
        parking_lot_name = "Parking Lot 1"
    else:
        closest_parking_lot = parking_lot2
        closest_distance = distance2
        parking_lot_name = "Parking Lot 2"

    st.write(f"Closest Parking Lot: {parking_lot_name} (Distance: {closest_distance:.2f} km)")

elif page == "Parking Lot 1 Video":
    st.header("Processed Video: parking1.mp4")
    video = cv2.VideoCapture('parking1.mp4')
    stframe = st.empty()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        processed_frame, available, total = process_frame(frame, areas_parking1)
        stframe.image(processed_frame, channels="BGR", use_column_width=True)
        st.text(f"Available parking spaces: {available}/{total}")

    video.release()

elif page == "Parking Lot 2 Video":
    st.header("Processed Video: easy1.mp4")
    video = cv2.VideoCapture('easy1.mp4')
    stframe = st.empty()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        processed_frame, available, total = process_frame(frame, areas_easy1)
        stframe.image(processed_frame, channels="BGR", use_column_width=True)
        st.text(f"Available parking spaces: {available}/{total}")

    video.release()
