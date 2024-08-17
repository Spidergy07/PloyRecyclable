import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np

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
    tensor = transform(image).unsqueeze(0)
    return tensor

# Example class labels (adjust as needed)
class_labels = ["กระดาษ ราคา/กก. 1-4 บาท", "ขวดแก้ว ราคา/กก. 0.25-3 บาท", "พลาสติกรวม ราคา/กก.  5-8 บาท", "พลาสติกใส ราคา/กก. 5-10 บาท", "เศษเหล็ก ราคา/กก. 6-12 บาท"]

model, device = load_model()

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น โดยแสดงผลประเภทขยะรีไซเคิลและช่วงราคาต่อกิโลกรัม')

# Start capturing video
cap = cv2.VideoCapture(0)
stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the video frame in real-time
    stframe.image(rgb_frame, channels="RGB", use_column_width=True)

    # Capture a frame when the "Capture" button is pressed
    if st.button("Capture Frame"):
        pil_image = Image.fromarray(rgb_frame)
        
        # Preprocess the image
        input_tensor = preprocess_image(pil_image)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor.to(device))
            _, predicted_class = torch.max(prediction, 1)
        
        # Get the predicted label
        predicted_label = class_labels[predicted_class.item()]
        
        # Display the prediction with label
        st.success(f'Predicted class: {predicted_class.item()} - {predicted_label}')

cap.release()
