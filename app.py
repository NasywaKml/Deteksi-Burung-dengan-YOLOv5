import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Memuat model YOLOv5 secara lokal
device = select_device('cpu')  # Gunakan 'cuda' jika tersedia
model = attempt_load("models/best.pt", map_location=device)
model.eval()

# UI Aplikasi Streamlit
st.title("Deteksi Burung dengan YOLOv5")
st.sidebar.title("Pilihan")

# Pilihan mode deteksi
mode = st.sidebar.radio("Pilih mode:", ["Upload Video", "Upload Gambar"])

# Fungsi deteksi objek pada gambar
def detect_objects(img):
    img_resized = cv2.resize(img, (640, 640))  # Resize gambar ke ukuran model
    img_tensor = torch.from_numpy(img_resized).to(device).float()
    img_tensor /= 255.0  # Normalisasi
    img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # Format ke [batch, channel, height, width]

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)[0]  # Filter prediksi

    detected_img = img.copy()
    if pred is not None:
        for det in pred:
            x1, y1, x2, y2, conf, cls = det
            label = f"{conf:.2f}"
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return detected_img

# Mode Upload Gambar
if mode == "Upload Gambar":
    st.write("Mode Upload Gambar")
    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        img = np.array(Image.open(uploaded_image))
        detected_image = detect_objects(img)
        st.image(detected_image, caption="Objek yang Terdeteksi", use_container_width=True)

# Mode Upload Video
elif mode == "Upload Video":
    st.write("Mode Upload Video")
    uploaded_video = st.file_uploader("Unggah file video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file)
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            st_frame.image(detected_frame)

        cap.release()
