import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np

# สร้างคลาสโมเดล ResNet50 ที่กำหนดเอง
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=101)  # กำหนดจำนวนคลาสที่ตรงกับข้อมูล

    def forward(self, x):
        x = self.resnet(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50()
    try:
        model.load_state_dict(torch.load('resnet50.pth', map_location=device), strict=False)  # ใช้ strict=False ในการโหลดโมเดล
    except RuntimeError as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    model.to(device)
    model.eval()
    return model, device

# ฟังก์ชันสำหรับเตรียมภาพก่อนส่งให้โมเดล
def preprocess_image(image, input_height=224, input_width=224):
    transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

# กำหนดป้ายกำกับคลาส (ปรับตามความเหมาะสม)
class_labels = ["กระดาษ ราคา/กก. 1-4 บาท", "ขวดแก้ว ราคา/กก. 0.25-3 บาท", "พลาสติกรวม ราคา/กก. 5-8 บาท", "พลาสติกใส ราคา/กก. 5-10 บาท", "เศษเหล็ก ราคา/กก. 6-12 บาท"]

# โหลดโมเดล
model, device = load_model()

# แสดงหัวข้อแอป
st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น โดยแสดงผลประเภทขยะรีไซเคิลและช่วงราคาต่อกิโลกรัม')

# ปุ่มเริ่มต้นการใช้งานกล้อง
start_button = st.button("เริ่มต้นการใช้งานกล้อง", key="start_button")

# ปุ่มหยุดการใช้งานกล้อง
stop_button = st.button("หยุดการใช้งานกล้อง", key="stop_button")

# หากกดปุ่มเริ่มต้นการใช้งานกล้อง
if start_button:
    cap = cv2.VideoCapture(0)  # เริ่มการจับภาพจากกล้อง
    video_placeholder = st.empty()  # พื้นที่แสดงวิดีโอ
    prediction_placeholder = st.empty()  # พื้นที่แสดงผลการทำนาย

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("ไม่สามารถจับภาพจากกล้องได้")
            break
        
        # แปลงจาก BGR เป็น RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # แปลงภาพเป็น PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # แสดงวิดีโอ
        video_placeholder.image(pil_image, channels="RGB", use_column_width=True)
        
        # เตรียมภาพก่อนส่งให้โมเดล
        input_tensor = preprocess_image(pil_image)
        
        # ทำการทำนาย
        with torch.no_grad():
            prediction = model(input_tensor.to(device))
            _, predicted_class = torch.max(prediction, 1)
        
        # รับผลลัพธ์ที่ทำนายได้
        predicted_label = class_labels[predicted_class.item()]
        
        # แสดงผลลัพธ์พร้อมป้ายกำกับ
        prediction_placeholder.success(f'คลาสที่ทำนายได้: {predicted_class.item()} - {predicted_label}')
        
        # หยุดการทำงานหากกดปุ่มหยุด
        if stop_button:
            cap.release()
            break

    # ปล่อยกล้องเมื่อหยุดการทำงาน
    if not stop_button:
        cap.release()

# ปล่อยกล้องเมื่อแอปสิ้นสุดการทำงานหรือกดปุ่มหยุด
if cap.isOpened():
    cap.release()
