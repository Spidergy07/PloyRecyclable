import streamlit as st
import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn

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

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น')

cap = cv2.VideoCapture(0)

stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Cannot read from webcam. Please check your camera.")
        break

    # Convert OpenCV BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Preprocess the image
    input_tensor = preprocess_image(pil_image)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor.to(device))
        _, predicted_class = torch.max(prediction, 1)
    
    predicted_label = class_labels[predicted_class.item()]
    
    # Add prediction text to frame
    cv2.putText(frame, f"Predicted class: {predicted_label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    stframe.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()