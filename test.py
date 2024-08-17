import streamlit as st
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import torch.nn as nn

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

def read_tensor_from_image_url(url, input_height=224, input_width=224):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor

model, device = load_model()

st.title('Image Classification with ResNet50')

image_url = st.text_input('Enter image URL:')

if image_url:
    st.image(image_url, caption='Input Image', use_column_width=True)
    
    if st.button('Predict'):
        image_tensor = read_tensor_from_image_url(image_url)
        
        with torch.no_grad():
            prediction = model(image_tensor.to(device))
            _, predicted_class = torch.max(prediction, 1)
        
        st.success(f'Predicted class: {predicted_class.item()}')