import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
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
    tensor = transform(Image.fromarray(image)).unsqueeze(0)
    return tensor

# สร้างลิสต์ของ label สำหรับ 101 คลาส
class_labels = ["กระดาษ ราคา/กก. 1-4 บาท", "ขวดแก้ว ราคา/กก. 0.25-3 บาท", "พลาสติกรวม ราคา/กก.  5-8 บาท", "พลาสติกใส ราคา/กก. 5-10 บาท", "เศษเหล็ก ราคา/กก. 6-12 บาท"]

model, device = load_model()

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น โดยแสดงผลประเภทขยะรีไซเคิลและช่วงราคาต่อกิโลกรัม')

# ใช้ streamlit เพื่อนำเข้าภาพจากกล้องเว็บแคม
image = st.camera_input("Take a picture")

if image:
    # แปลงภาพเป็น NumPy array
    image_np = np.array(Image.open(image))
    
    # แสดงภาพ
    st.image(image_np, caption='Captured Image', use_column_width=True)
    
    # ประมวลผลภาพ
    input_tensor = preprocess_image(image_np)
    
    # ทำการคาดการณ์
    with torch.no_grad():
        prediction = model(input_tensor.to(device))
        _, predicted_class = torch.max(prediction, 1)
    
    # รับ label ที่คาดการณ์
    predicted_label = class_labels[predicted_class.item()]
    
    # แสดงผลลัพธ์
    st.success(f'Predicted class: {predicted_class.item()} - {predicted_label}')
