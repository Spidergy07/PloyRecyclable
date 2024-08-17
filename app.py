import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

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
st.title("Image Classification with ResNet50")

# ตัวอัปโหลดไฟล์สำหรับอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # โหลดโมเดล
    model = load_model()
    
    # ทำนายผล
    class_id = predict(image, model)
    
    # แสดงผลลัพธ์
    st.write(f"Predicted class: {class_names[class_id]}")
