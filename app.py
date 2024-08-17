import streamlit as st
import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=5)
        
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

class_labels = ["กระดาษ", "ขวดแก้ว", "พลาสติกรวม", "พลาสติกใส", "เศษเหล็ก"]

model, device = load_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.device = device

    def transform(self, frame):
        img = frame.to_ndarray(format="rgb24")
        pil_image = Image.fromarray(img)
        
        input_tensor = preprocess_image(pil_image)
        
        with torch.no_grad():
            prediction = self.model(input_tensor.to(self.device))
            _, predicted_class = torch.max(prediction, 1)
        
        predicted_label = class_labels[predicted_class.item()]
        
        cv2.putText(img, f"Predicted class: {predicted_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น')

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)