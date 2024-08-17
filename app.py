import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# โหลดโมเดล ResNet50
model = models.resnet50()
model.load_state_dict(torch.load("resnet50.pth", map_location=torch.device('cpu')))
model.eval()

# การเตรียมข้อมูลก่อนป้อนให้โมเดล
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# กำหนดคลาส (คุณสามารถแก้ไขได้ตามความเหมาะสม)
classes = ['แมว', 'สุนัข', 'ม้า', 'ช้าง', 'นก', 'แมลง', 'ปลา', 'เต่า']

# ฟังก์ชันในการทำนาย
def predict(image):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# ส่วนติดต่อผู้ใช้
st.title("Image Classification with ResNet50")
st.write("อัปโหลดภาพและรับผลการทำนาย")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    label = predict(image)
    st.write(f"Predicted class: {label}")
