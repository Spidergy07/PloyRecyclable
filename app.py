import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# โหลดโมเดล ResNet50 จากไฟล์ที่ดาวน์โหลดไว้แล้ว
model = models.resnet50()
model.load_state_dict(torch.load("resnet50.pth"))
model.eval()

# การเตรียมข้อมูลก่อนป้อนให้โมเดล
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# กำหนดคลาส (ตามที่ ImageNet ใช้)
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# ฟังก์ชันในการทำนาย
def predict(image):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# ฟังก์ชันสตรีมวิดีโอจากเว็บแคม
def video_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # แปลงภาพจาก BGR เป็น RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ทำการทำนาย
        label = predict(image)

        # แสดงผลการทำนายบนเฟรม
        cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # แสดงผลเฟรมใน Streamlit
        st.image(frame, channels="BGR")

        # ตรวจสอบว่า Streamlit ได้รับการหยุดหรือไม่
        if st.button('Stop'):
            break

    cap.release()

# ส่วนติดต่อผู้ใช้
st.title("Real-time Object Detection with ResNet50")
st.write("สตรีมวิดีโอจากเว็บแคมและทำนายวัตถุแบบเรียลไทม์")

video_stream()
