import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # ปรับเป็น 5 คลาส
    model.load_state_dict(torch.load('resnet50.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# กำหนดคลาส
class_names = ['คลาส1', 'คลาส2', 'คลาส3', 'คลาส4', 'คลาส5']

# ฟังก์ชันสำหรับทำนาย
def predict(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Streamlit app
st.title('การจำแนกภาพด้วย ResNet50')

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# สร้าง placeholder สำหรับแสดงภาพจากกล้องและผลลัพธ์การทำนาย
camera_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("ไม่สามารถเปิดกล้องได้")
        break
    
    # แปลงภาพให้เป็นรูปแบบที่ถูกต้อง
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # ทำนายผล
    prediction = predict(pil_image)
    
    # แสดงภาพจากกล้องและผลลัพธ์การทำนาย
    camera_placeholder.image(frame_rgb, caption=f"Predicted class: {prediction}", channels="RGB", use_column_width=True)
    
    # ตรวจสอบการกดปุ่ม 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
