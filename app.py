import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import time

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=101)
        
    def forward(self, x):
        x = self.resnet(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50()
    model.load_state_dict(torch.load('resnet50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image, input_height=224, input_width=224):
    transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
    ])
    tensor = transform(Image.fromarray(image)).unsqueeze(0)
    return tensor

# สร้างลิสต์ของ label สำหรับ 101 คลาส
class_labels = ["กระดาษ ราคา/กก. 1-4 บาท", "ขวดแก้ว ราคา/กก. 0.25-3 บาท", "พลาสติกรวม ราคา/กก.  5-8 บาท", "พลาสติกใส ราคา/กก. 5-10 บาท", "เศษเหล็ก ราคา/กก. 6-12 บาท"]

model, device = load_model()

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น โดยแสดงผลประเภทขยะรีไซเคิลและช่วงราคาต่อกิโลกรัม')

# เริ่มสตรีมวิดีโอ
video_stream = st.empty()

# สร้าง placeholder สำหรับแสดงผลการคาดการณ์
prediction_text = st.empty()

# เริ่มการจับภาพวิดีโอ
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # แปลง BGR เป็น RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ประมวลผลภาพ
    input_tensor = preprocess_image(frame_rgb)
    
    # ทำการคาดการณ์
    with torch.no_grad():
        prediction = model(input_tensor.to(device))
        _, predicted_class = torch.max(prediction, 1)
    
    # รับ label ที่คาดการณ์
    predicted_label = class_labels[predicted_class.item()]
    
    # วาดข้อความบนเฟรม
    cv2.putText(frame, f'Class: {predicted_class.item()} - {predicted_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # แสดงเฟรมในสตรีมวิดีโอ
    video_stream.image(frame, channels="BGR")
    
    # อัปเดตข้อความการคาดการณ์
    prediction_text.text(f'คลาสที่คาดการณ์: {predicted_class.item()} - {predicted_label}')
    
    # รอสักครู่เพื่อลดการใช้ทรัพยากร
    time.sleep(0.1)

# ปิดการจับภาพเมื่อเสร็จสิ้น
cap.release()