import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import requests
import os

model_path = 'resnet50.pth'

# โหลดโมเดล
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # ปรับจำนวนคลาสให้ตรงกับโมเดลของคุณ
model.load_state_dict(torch.load(model_path))
model.eval()

# กำหนด transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ฟังก์ชันสำหรับทำ inference
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Map index to class name
class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']  # แทนที่ด้วยชื่อคลาสของคุณ

# Streamlit UI
st.title("Object Detection with ResNet50")
run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    prediction = predict(image)
    label = class_names[prediction]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(frame)

cap.release()
