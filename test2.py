import streamlit as st
import base64
import cv2
from PIL import Image

st.title('การแยกประเภทขยะรีไซเคิลเบื้องต้น โดยแสดงผลประเภทขยะรีไซเคิลและช่วงราคาต่อกิโลกรัม')

st.markdown("""
<script>
async function startCamera() {
    const video = document.createElement('video');
    video.width = 640;
    video.height = 480;
    document.body.appendChild(video);

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();

    const canvas = document.createElement('canvas');
    canvas.width = video.width;
    canvas.height = video.height;
    document.body.appendChild(canvas);

    const context = canvas.getContext('2d');

    function captureFrame() {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/png');
        document.getElementById('capturedImage').src = dataURL;
        setTimeout(captureFrame, 1000); // Capture frame every second
    }
    captureFrame();
}

window.onload = startCamera;
</script>
<img id="capturedImage" src="" style="display:none;">
""", unsafe_allow_html=True)

# Display the captured image
captured_image = st.image("", caption="Captured Image", use_column_width=True)
