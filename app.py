import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import logging

# Configuration
MODEL_PATH = "resnet34_quality.pth"
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["Bad_Blur", "Bad_Dark", "Bad_Dull", "Good"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


@st.cache_resource
def load_model():
    """
    Load the ResNet34 model with pretrained ImageNet weights and custom classification head.
    
    Returns:
        torch.nn.Module: Loaded model on the appropriate device
        
    Raises:
        FileNotFoundError: If model file is not found
        Exception: If model loading or state dict loading fails
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Initialize model with modern weights parameter
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Replace final layer for custom classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
        # Load trained weights
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model loading error: {e}")
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        st.error(f"❌ Error loading model: {str(e)}")
        raise


# Load model once at startup
try:
    model = load_model()
except Exception:
    st.stop()


# Streamlit UI Configuration
st.set_page_config(
    page_title="Image Quality Analyzer",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Image Quality Analyzer")
st.write("Upload an image to analyze its quality (Good, Blur, Dark, or Dull)")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # File size validation
        file_size = len(uploaded_file.getbuffer())
        if file_size > MAX_FILE_SIZE:
            st.error(f"❌ File size exceeds {MAX_FILE_SIZE / (1024*1024):.0f} MB limit")
            st.stop()
        
        # Open and display image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Image dimension validation
        width, height = image.size
        st.image(image, caption=f"Uploaded Image ({width}×{height}px)", use_container_width=True)
        
        # Show processing status
        with st.spinner("🔍 Analyzing image..."):
            # Preprocess image
            img_tensor = transform(image).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            # Model inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            confidence, prediction_idx = torch.max(probs, 1)
            confidence_pct = confidence.item() * 100
            predicted_label = CLASS_NAMES[prediction_idx.item()]
        
        # Display main prediction
        st.markdown(f"### ✅ Prediction: **{predicted_label}** ({confidence_pct:.2f}% confidence)")
        
        # Display confidence scores for all classes
        st.subheader("Confidence Scores")
        
        # Create a bar chart of all class probabilities
        probs_dict = {
            CLASS_NAMES[i]: float(probs[0, i].cpu().numpy()) * 100
            for i in range(NUM_CLASSES)
        }
        
        # Sort by confidence descending
        sorted_probs = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(sorted_probs)
        
        with col2:
            st.metric("Top Confidence", f"{confidence_pct:.2f}%")
        
        # Display detailed breakdown
        st.subheader("Detailed Breakdown")
        for class_name, prob in sorted_probs.items():
            st.write(f"**{class_name}**: {prob:.2f}%")
        
        logger.info(f"Prediction made: {predicted_label} ({confidence_pct:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Model: ResNet34 | Device: {} | Classes: {}<br>
    </div>
    """.format(
        str(device).upper(),
        ", ".join(CLASS_NAMES)
    ),
    unsafe_allow_html=True
)
