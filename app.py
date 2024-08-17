import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

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

class_labels = ["กระดาษ ราคา/กก. 1-4 บาท", "ขวดแก้ว ราคา/กก. 0.25-3 บาท", "พลาสติกรวม ราคา/กก.  5-8 บาท", "พลาสติกใส ราคา/กก. 5-10 บาท", "เศษเหล็ก ราคา/กก. 6-12 บาท"]

model, device = load_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.device = device

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        input_tensor = preprocess_image(pil_image)
        
        with torch.no_grad():
            prediction = self.model(input_tensor.to(self.device))
            _, predicted_class = torch.max(prediction, 1)
        
        predicted_label = class_labels[predicted_class.item()]
        
        cv2.putText(img, f'Class: {predicted_class.item()} - {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น โดยแสดงผลประเภทขยะรีไซเคิลและช่วงราคาต่อกิโลกรัม')

ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

if ctx.video_transformer:
    st.write(f"Last prediction: {ctx.video_transformer.last_prediction}")