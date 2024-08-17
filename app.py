import streamlit as st
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# โหลดโมเดลที่เทรนไว้แล้ว
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # เปลี่ยน output layer สำหรับ 5 คลาส
    model.load_state_dict(torch.load("resnet50.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# ฟังก์ชันสำหรับการพยากรณ์
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(image).unsqueeze(0)  # เพิ่ม batch dimension
    with torch.no_grad():
        outputs = model(img)
    _, predicted = outputs.max(1)
    return predicted.item()

# ชื่อคลาส (แก้ไขให้ตรงกับคลาสของคุณ)
class_names = ['คลาสที่ 1', 'คลาสที่ 2', 'คลาสที่ 3', 'คลาสที่ 4', 'คลาสที่ 5']

# สร้าง Streamlit app
st.title("Real-Time Object Detection with ResNet50")

# ตัวเลือกเปิดปิดกล้อง
run = st.checkbox('Run camera')

# โหลดโมเดล
model = load_model()

# ฟังก์ชันสำหรับเปิดกล้อง
if run:
    # เปิดการใช้งานกล้อง
    video_capture = cv2.VideoCapture(0)
    
    stframe = st.empty()  # ตัวแปรสำหรับแสดงผลวิดีโอ
    
    while run:
        ret, frame = video_capture.read()
        if not ret:
            break

        # เปลี่ยนสีจาก BGR (ของ OpenCV) เป็น RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # ทำนายผล
        class_id = predict(img_pil, model)
        class_name = class_names[class_id]

        # แสดงชื่อคลาสบนวิดีโอ
        cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # แสดงผลวิดีโอใน Streamlit
        stframe.image(frame, channels="BGR")

    video_capture.release()

# ถ้าไม่เปิดกล้อง ให้แสดงข้อความปิดกล้อง
else:
    st.write("Camera is off. Check 'Run camera' to start.")

