import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Vehicle Counter", layout="wide")
st.title("🚗 Vehicle Counting and Classification System")

# --- Session State for History Table ---
if 'history' not in st.session_state:
    st.session_state.history = []

# Load YOLOv8 model (Medium size for better accuracy)
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt") 

model = load_model()

uploaded_video = st.file_uploader("Choose a video file (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Original Video")
        st.video(uploaded_video)
        
    with col2:
        st.subheader("🎯 Counting Results (Real-time)")
        
        stframe = st.empty() 
        m1, m2, m3 = st.columns(3)
        car_metric = m1.empty()
        moto_metric = m2.empty()
        truck_metric = m3.empty()
        
        car_metric.metric(label="🚗 Car", value="0")
        moto_metric.metric(label="🛵 Motorcycle", value="0")
        truck_metric.metric(label="🚚 Truck", value="0")

        if st.button("🚀 Start Detection and Counting"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            line_y = int(height / 2)
            
            track_history = {}
            counts = {"car": 0, "motorcycle": 0, "truck": 0}
            
            class_mapping = {2: "car", 3: "motorcycle", 7: "truck"}
            classes_to_track = [2, 3, 7]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # AI tuning: conf=0.4 to filter out low-confidence detections
                results = model.track(frame, classes=classes_to_track, conf=0.4, persist=True, tracker="botsort.yaml", verbose=False)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()

                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        class_name = class_mapping[int(class_id)]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name.capitalize()} ID:{int(track_id)}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                        if track_id in track_history:
                            prev_cy = track_history[track_id]
                            if (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy):
                                counts[class_name] += 1
                        
                        track_history[track_id] = cy

                cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 3)
                cv2.putText(frame, "Counting Line", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                
                car_metric.metric(label="🚗 Car", value=str(counts['car']))
                moto_metric.metric(label="🛵 Motorcycle", value=str(counts['motorcycle']))
                truck_metric.metric(label="🚚 Truck", value=str(counts['truck']))

            cap.release()
            os.remove(video_path) 
            st.success("Processing Complete! 🎉")
            
            # --- 📊 Save data to history table ---
            total_vehicles = counts['car'] + counts['motorcycle'] + counts['truck']
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            new_record = {
                "Date/Time": current_time,
                "Car": counts['car'],
                "Motorcycle": counts['motorcycle'],
                "Truck": counts['truck'],
                "Total": total_vehicles
            }
            
            st.session_state.history.append(new_record)

# --- Display History Table ---
st.markdown("---")
st.subheader("📋 Vehicle Counting History Table")

if len(st.session_state.history) > 0:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, hide_index=True, use_container_width=True)
    
    if st.button("🗑️ Clear All History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No counting history available. Please upload a video and start processing.")
