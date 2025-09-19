import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=True)   # ✅ same as training
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
    model.load_state_dict(torch.load("resnet34_quality.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Class names
classes = ["Bad_Blur", "Bad_Dark", "Bad_Dull", "Good"]

# Streamlit UI
st.title("📷 Image Quality Analyzer")
st.write("Upload an image to check if it's Good or Bad (Blur, Dark, Dull).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

    label = classes[preds.item()]
    confidence = conf.item() * 100

    st.markdown(f"### ✅ Prediction: **{label} ({confidence:.2f}% confidence)**")
