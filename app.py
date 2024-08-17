import streamlit as st
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

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
st.title("Real-time Object Detection with ResNet50")

# โหลดโมเดล
model = load_model()

# เริ่มการทำงานของกล้อง
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
    
    # แปลงภาพจาก BGR (ที่ opencv ใช้) เป็น RGB (ที่ PIL ใช้)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # แปลงจาก numpy array เป็น PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    # ทำนายผล
    class_id = predict(pil_img, model)
    
    # วาดข้อความผลลัพธ์ลงบนภาพ
    cv2.putText(frame, f'Predicted: {class_names[class_id]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # แสดงภาพ
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    cap.release()
    cv2.destroyAllWindows()
