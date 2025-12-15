import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import yaml
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ==================== NEW GRAD-CAM IMPORTS ====================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# ==============================================================

# ==================== LOAD MODEL ARCHITECTURE ====================
class CustomResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding='same')
        self.conv2 = torch.nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding='same')
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels[1], kernel_size=1, stride=1, padding='valid'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels[1])
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels[0])
        self.bn2 = torch.nn.BatchNorm2d(out_channels[1])
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.bn1(self.relu(self.conv1(input)))
        input = self.bn2(self.relu(self.conv2(input)))
        input = torch.cat([input, shortcut], dim=1)
        return input

class LightTBNet_4blocks(torch.nn.Module):
    def __init__(self, in_channels, outputs):
        super().__init__()
        layer0 = torch.nn.Sequential(
            CustomResBlock(in_channels, [16, 16]),
            torch.nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer1 = torch.nn.Sequential(
            CustomResBlock(32, [32, 32]),
            torch.nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer2 = torch.nn.Sequential(
            CustomResBlock(64, [64, 64]),
            torch.nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer3 = torch.nn.Sequential(
            CustomResBlock(128, [128, 128]),
            torch.nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, out_channels=16, kernel_size=1, padding='valid'),
            torch.nn.ReLU(),
        )
        self.features = torch.nn.Sequential()
        self.features.add_module('layer0', layer0)
        self.features.add_module('layer1', layer1)
        self.features.add_module('layer2', layer2)
        self.features.add_module('layer3', layer3)
        self.features.add_module('layer4', layer4)

        # 4*4*16 is the correct flattened size for 64x64 input
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 16, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, outputs),
        )

    def forward(self, input):
        input = self.features(input)
        input = self.classifier(input)
        return input

# ==================== CONFIGURE PAGE ====================
st.set_page_config(
    page_title="TB Detection",
    page_icon="🫁",
    layout="centered"
)

# ==================== LOAD CONFIG ====================
@st.cache_resource
def load_config():
    """Load configuration from YAML"""
    try:
        with open("config_train.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error("config_train.yaml not found! Make sure it's in the same directory.")
        return None
    
# ==================== GLOBAL CONFIG AND DEVICE SETUP (CRITICAL) ====================
# Define these globally for clean access across all functions
CONFIG_GLOBAL = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================================================

# ==================== LOAD MODEL (MODIFIED) ====================
@st.cache_resource
def load_model(): 
    """Load trained model using global CONFIG_GLOBAL and DEVICE"""
    
    # Access the global variables defined above
    config = CONFIG_GLOBAL
    device = DEVICE
    
    try:
        if config is None:
            st.error("Config not loaded")
            return None

        # Initialize model
        model = LightTBNet_4blocks(in_channels=1, outputs=2)
        model_path = Path("./models/model_best.pth")
        
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            st.info("Please ensure the model path is correct.")
            return None
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        # ----------------------------------------------------------------------
        #  <<< SANITY CHECK CODE GOES HERE >>>
        # ----------------------------------------------------------------------
        
        # 1. Check if the output classifier layer weights are not zero (Corruption Check)
        classifier_weights = model.classifier[3].weight.data
        if torch.all(classifier_weights == 0):
            st.error("❌ WEIGHTS ERROR: Final classifier weights are all zeros. Check checkpoint saving process.")
            return None

        # 2. Check the model output shape for a dummy input (Architecture Check)
        temp_model = model.to(device)
        dummy_input = torch.randn(1, 1, config['img_dim_ai'], config['img_dim_ai']).to(device)
        
        with torch.no_grad():
            dummy_output = temp_model(dummy_input)
            
        expected_output_shape = (1, 2)
        if dummy_output.shape != expected_output_shape:
            st.error(f"❌ ARCHITECTURE ERROR: Model output shape is {dummy_output.shape}, expected {expected_output_shape}.")
            st.info("Check `outputs` parameter in LightTBNet_4blocks initialization.")
            return None
        
        # ----------------------------------------------------------------------

        # Final setup before returning the model
        model.to(device)
        model.eval()
        st.success(f"✅ Model loaded successfully from {model_path.parent.name}!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ==================== PREPROCESSING FUNCTION ====================
def preprocess_image(image_pil, img_dim=64):
    """
    Handles image preprocessing including channel checks for robustness.
    """
    try:
        # 1. Initial Load & Channel Check
        img_np = np.array(image_pil)
        
        # Convert to grayscale and ensure it is UINT8 (0-255 range)
        img_grayscale = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.uint8) 

        # 2. Apply standard CV2 CLAHE
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
        img_clahe = clahe_obj.apply(img_grayscale)
        
        # 3. Define the Albumentations pipeline for RESIZE and NORMALIZE
        transform = A.Compose([
            A.Resize(height=img_dim, width=img_dim),  
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0), 
            ToTensorV2(),
        ])

        img_float32 = img_clahe.astype(np.float32)

        # 4. Apply transforms
        transformed = transform(image=img_float32) 
        img_tensor = transformed['image'] 
        
        # 5. Add batch dimension (Shape [1, 1, 64, 64])
        img_tensor = img_tensor.unsqueeze(0)
             
        return img_tensor
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

# ==================== PREDICTION FUNCTION (MODIFIED) ====================
def predict_tb(image_pil, model, device, config):
    """ Make prediction on the image and return the predicted class """
    if model is None:
        st.error("Model not loaded")
        # Return 4 values to match the unpacking in main
        return None, None, None, None 
    try:
        img_tensor = preprocess_image(image_pil, img_dim=config['img_dim_ai'])
        if img_tensor is None:
            return None, None, None, None

        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Extract probabilities
        prob_no_tb = float(probabilities[0, 0].cpu().numpy())
        prob_tb = float(probabilities[0, 1].cpu().numpy())

        # Determine class (CRITICAL for Grad-CAM)
        pred_class = int(torch.argmax(probabilities, dim=1)[0].cpu().numpy())
        has_tb = pred_class == 1

        # Return the predicted class along with probabilities
        return has_tb, prob_tb, prob_no_tb, pred_class
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None
        
# ==================== GRAD-CAM GENERATION FUNCTION ====================
def generate_gradcam(image_pil, model, device, config, target_category: int):
    """
    Generate Grad-CAM heatmap for LightTBNet.
    """
    
    # 1. Preprocess image for model input
    img_tensor = preprocess_image(image_pil, img_dim=config['img_dim_ai'])
    if img_tensor is None:
        return None
    img_tensor = img_tensor.to(device)

    # 2. Target layer setup (Last convolutional block of the feature extractor)
    target_layers = [model.features.layer3]
    targets = [ClassifierOutputTarget(target_category)] 

    # 3. Initialize and Run GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    # 4. Prepare image for overlay (must be float32, RGB-like, 0-1 range)
    img_resized = image_pil.resize((config['img_dim_ai'], config['img_dim_ai']))
    img_np = np.array(img_resized)
    
    # Handle Grayscale/RGB conversion for overlay
    if img_np.ndim == 3 and img_np.shape[-1] == 3:
        img_float = img_np.astype(np.float32) / 255.0
    elif img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
        img_gray = img_np.squeeze()
        img_float = np.stack([img_gray]*3, axis=-1).astype(np.float32) / 255.0
    else:
        # Fallback to white image on error
        return Image.fromarray(np.ones((config['img_dim_ai'], config['img_dim_ai'], 3), dtype=np.uint8) * 255)
        
    # 5. Overlay CAM (cam_image is a NumPy array, RGB, 0-255)
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Convert NumPy array back to PIL Image for Streamlit
    return Image.fromarray(cam_image)
    
# ==================== MAIN UI ====================
def main():
    st.title("🫁 Tuberculosis Detection")
    st.write("Upload a chest X-ray image to detect tuberculosis using LightTBNet")

    # Load config and model (Using global variables and corrected load_model call)
    config = CONFIG_GLOBAL
    device = DEVICE
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    # When a file is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if st.button("🔍 Analyze X-ray", type="primary"):
            if model is None:
                st.error("Model is not loaded. Please check the model path.")
            else:
                with st.spinner("Analyzing..."):
                    # --- Make prediction (Receive pred_class) ---
                    has_tb, prob_tb, prob_no_tb, pred_class = predict_tb(image, model, device, config)
                    
                    if has_tb is not None:
                        st.markdown("---")
                        st.subheader("Analysis Results")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if has_tb:
                                st.error(f"⚠️ **TB DETECTED**")
                            else:
                                st.success(f"✓ **NO TB DETECTED**")
                        with col2:
                            st.metric("TB Probability", f"{prob_tb*100:.1f}%")
                        with col3:
                            st.metric("Normal Probability", f"{prob_no_tb*100:.1f}%")

                        st.progress(max(prob_tb, prob_no_tb))

                        st.markdown("### Confidence Breakdown")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"🦠 **TB Class**: {prob_tb*100:.2f}%")
                        with col2:
                            st.write(f"✓ **Normal Class**: {prob_no_tb*100:.2f}%")

                        st.warning(
                            "⚕️ **Important Disclaimer**: This is a screening tool only. "
                            "Please consult a qualified healthcare professional for proper diagnosis and treatment. "
                            "This model is for research purposes and should not be used for clinical decision-making without professional oversight."
                        )

                        # --- Generate and display Grad-CAM (NEW) ---
                        st.markdown("---")
                        st.subheader("Model Explanation (Grad-CAM)")

                        cam_image = generate_gradcam (
                            image_pil=image,
                            model=model,
                            device=device,
                            config=config,
                            target_category=pred_class 
                        )
                        st.image(
                            cam_image,
                            caption="Grad-CAM Heatmap",
                            use_container_width=True
                        )

# ==================== SETUP INSTRUCTIONS ====================
with st.expander("📋 Setup Instructions"):
    st.markdown(f"""
    ### Initial Setup Required:
    1. **Trained Model Path Verification** (CRITICAL):
       - This app uses the path: `.{Path('/models/model_best.pth')}`
    2. **Required Files** in same directory as app:
       - `config_train.yaml`
       - `model_best.pth` file (in the correct subfolder, i.e., `./models/`).
       - This script
    3. **Install Dependencies**: You will need `pytorch-grad-cam` in addition to others.
       ```bash
       pip install streamlit pillow torch torchvision opencv-python albumentations pyyaml pytorch-grad-cam
       ```
    4. **Run the App**:
       ```bash
       streamlit run app.py
       ```
    ### Model Details:
    - **Architecture**: LightTBNet_4blocks
    - **Input Size**: {CONFIG_GLOBAL.get('img_dim_ai', 64)}x{CONFIG_GLOBAL.get('img_dim_ai', 64)} grayscale
    - **Preprocessing**: **Dynamic Albumentations CLAHE** + Normalization 
    """)

if __name__ == "__main__":
    main() 

# ----------------------------------------------------------------------
# <<< FOOTER ATTRIBUTION >>>
# ----------------------------------------------------------------------

st.markdown("---") 
st.caption(
    f"""<div style="text-align: center;"> AI Semester Project  |  Developed by Raniya Yaqub and Tooba Mir</div>
    """,
    unsafe_allow_html=True
)
# ----------------------------------------------------------------------