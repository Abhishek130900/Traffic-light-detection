import streamlit as st
from ultralytics import YOLO
import pyttsx3
import cv2
import tempfile
import os
from PIL import Image

# ========== App Config ==========
st.set_page_config(page_title="Traffic Sign Detector", layout="wide")

# ========== Voice Engine ==========
engine = pyttsx3.init()

# ========== Load YOLOv8 Model ==========
model_path = "G:/Gehu Materials/Gehu self projects/Traffic light detection/best (1).pt"
model = YOLO(model_path)

# ========== Custom CSS Styling ==========
st.markdown("""
    <style>
        .title { font-size:40px; font-weight:bold; color:#00BFFF; text-align:center; margin-bottom:0px; }
        .subtitle { font-size:18px; text-align:center; color:#cccccc; margin-top:0px; margin-bottom:30px; }
        .section-title { font-size:20px; font-weight:bold; margin-top:30px; color:#FF6347; }
        .result-box { padding:10px; border-radius:10px; background-color:#f0f2f6; margin-top:10px; }
        .footer-note { text-align:center; color:gray; font-size:14px; margin-top:50px; }
    </style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=96)
st.sidebar.title("🚦 Traffic Sign Detection")
st.sidebar.markdown("""
**🔎 Description:**  
This mini-project detects traffic signs from images or videos using a custom-trained YOLOv8 model. It provides real-time voice feedback for the recognized traffic signs.

**👨‍💻 Created By:** Abhishek Kamboj
**📁 Project:** Traffic Sign Detection  
**📧 Contact:** abhishekkamboj542@gmail.com
""")

# ========== Title Section ==========
st.markdown("<div class='title'>Traffic Sign Detection using YOLOv8</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload any image or video containing traffic signs. The system will detect and speak out the recognized signs.</div>", unsafe_allow_html=True)

# ========== File Upload ==========
uploaded_file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# ========== Main Logic ==========
if uploaded_file is not None:
    ext = uploaded_file.name.split('.')[-1].lower()
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown(f"<div class='section-title'>📁 Uploaded File: {uploaded_file.name}</div>", unsafe_allow_html=True)

    # ========== If image ==========
    if ext in ['jpg', 'jpeg', 'png']:
        st.image(file_path, caption="Uploaded Image", use_container_width=True)
        results = model.predict(source=file_path, save=True)
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_label = model.names[int(box.cls)]
                detected_objects.append(class_label)

        st.markdown("<div class='section-title'>🔍 Detection Results</div>", unsafe_allow_html=True)

        if detected_objects:
            message = "Detected: " + ", ".join(set(detected_objects))
            st.success(message)
            engine.say(message)
            engine.runAndWait()
        else:
            message = "No traffic signs detected."
            st.warning(message)
            engine.say(message)
            engine.runAndWait()

        output_dir = os.path.join("runs", "detect")
        latest_run_dir = max([os.path.join(output_dir, d) for d in os.listdir(output_dir)], key=os.path.getmtime)
        processed_image_path = os.path.join(latest_run_dir, os.path.basename(file_path))

        if os.path.exists(processed_image_path):
            st.image(processed_image_path, caption="Detected Image", use_container_width=True)
        else:
            st.error("Processed image not found.")

    # ========== If video ==========
    elif ext in ['mp4', 'avi', 'mov']:
        st.video(file_path)
        cap = cv2.VideoCapture(file_path)
        video_result_path = os.path.join(temp_dir.name, "result_" + uploaded_file.name)
        writer = None
        detected_objects = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame)
            for result in results:
                for box in result.boxes:
                    class_label = model.names[int(box.cls)]
                    detected_objects.add(class_label)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width, _ = frame.shape
                writer = cv2.VideoWriter(video_result_path, fourcc, 20.0, (width, height))
            writer.write(frame)

        cap.release()
        if writer:
            writer.release()

        st.markdown("<div class='section-title'>🔍 Detection Summary</div>", unsafe_allow_html=True)
        if detected_objects:
            message = "Detected: " + ", ".join(detected_objects)
            st.success(message)
            engine.say(message)
            engine.runAndWait()
        else:
            message = "No traffic signs detected."
            st.warning(message)
            engine.say(message)
            engine.runAndWait()

        st.video(video_result_path)

    else:
        st.error("❌ Unsupported file format!")

# ========== Footer ==========
st.markdown("<div class='footer-note'>🎓 A Mini Project for Academic Purpose | Powered by YOLOv8 & Streamlit</div>", unsafe_allow_html=True)

# For running this project use this command 
# streamlit run "g:\Gehu Materials\Gehu self projects\Traffic light detection\trafficSignDetect.py"
