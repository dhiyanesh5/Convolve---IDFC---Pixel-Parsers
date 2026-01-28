import streamlit as st
import cv2
import numpy as np
import json
import os
import time
from executable import InvoiceExtractionPipeline
from utils.visualize_and_crop import visualize_detections

# --- PAGE CONFIG ---
st.set_page_config(page_title="Invoice Extraction System - Pixel Parsers", layout="wide")

# --- CUSTOM STYLING (IDFC) ---
st.markdown("""
<style>
    /* IDFC Brand Colors: Maroon/Red & White */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #9d2235; /* IDFC Red */
    }
    .crop-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #9d2235;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    .field-label {
        font-weight: 600;
        color: #555;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .extracted-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-top: 8px;
        font-family: 'Consolas', monospace;
    }
    .confidence-tag {
        font-size: 0.75rem;
        background-color: #f0f2f6;
        padding: 2px 8px;
        border-radius: 10px;
        color: #666;
        display: inline-block;
        margin-top: 5px;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #9d2235;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
# --- SIDEBAR BRANDING ---
with st.sidebar:
    # Check if logo exists, otherwise skip to prevent error
    if os.path.exists("utils\idfc.avif"):
        st.image("utils\idfc.avif", use_container_width=True)
    else:
        st.markdown("## IDFC FIRST BANK")
    
    st.markdown("---")
    st.header("System Config")
    use_vlm = st.checkbox("Enable Qwen-2.5-VL", value=False, help="Use Vision-Language Model for complex handwriting")
    conf_thresh = st.slider("Detection Sensitivity", 0.0, 1.0, 0.17)
    
    st.markdown("---")
    st.info("**System Pipeline**: YOLOv11m + Tesseract + Qwen2.5")
st.title("Invoice Extraction Dashboard - Pixel Parsers")

# # --- SIDEBAR ---
# st.sidebar.header("System Configuration")
# use_vlm = st.sidebar.checkbox("Enable Qwen-2.5-VL (Vision Language Model)", value=False)
# conf_thresh = st.sidebar.slider("Detection Sensitivity", 0.05, 1.0, 0.17)

# --- LOAD MODELS ---
@st.cache_resource
def get_pipeline():
    return InvoiceExtractionPipeline(
        ml_model_path="models/yolo11_best.pt",
        use_vlm=True # loaded but toggled via UI
    )

try:
    with st.spinner("Initializing Neural Networks..."):
        pipeline = get_pipeline()
    st.sidebar.success("System Ready")
except Exception as e:
    st.error(f"Critical Error Loading Models: {e}")
    st.stop()

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload Invoice Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 1. READ IMAGE
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) # BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save temp for pipeline
    temp_path = "temp_dashboard_upload.jpg"
    cv2.imwrite(temp_path, image)

    # 2. RUN PIPELINE
    with st.spinner("Analyzing Document Structure..."):
        # Update config dynamically
        pipeline.ml_detector.confidence_threshold = conf_thresh
        pipeline.use_vlm = use_vlm
        
        # A. Run Extraction Pipeline
        start_time = time.time()
        final_json = pipeline.process_image(temp_path)
        proc_time = time.time() - start_time
        
        # B. Run Pure Detection (for visualization)
        raw_detections = pipeline.ml_detector.detect(image)
        
        # C. Create Visualized Image
        annotated_img = visualize_detections(temp_path, raw_detections, output_path=None)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # --- LAYOUT: ROW 1 (IMAGES) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Upload")
        # Fixed deprecation warning: use_container_width instead of use_column_width
        st.image(image_rgb, use_container_width=True)
        
    with col2:
        st.subheader("AI Vision Analysis")
        st.image(annotated_img_rgb, use_container_width=True, caption=f"Detected {sum(len(v) for v in raw_detections.values())} fields of interest")

    # --- LAYOUT: ROW 2 (CROPS & VALUES) ---
    st.markdown("---")
    st.subheader("Extraction Details (Visual Verification)")
    
    # Helper to display a "Card"
    def display_field_card(field_key, display_name, extracted_val, col):
        with col:
            st.markdown(f"<div class='crop-card'><div class='field-label'>{display_name}</div>", unsafe_allow_html=True)
            
            # 1. GET CROP
            dets = raw_detections.get(field_key, [])
            if dets:
                best_det = dets[0]
                x1, y1, x2, y2 = best_det['bbox']
                h, w, _ = image_rgb.shape
                pad = 10
                y1, y2 = max(0, y1-pad), min(h, y2+pad)
                x1, x2 = max(0, x1-pad), min(w, x2+pad)
                
                crop = image_rgb[y1:y2, x1:x2]
                st.image(crop, use_container_width=True)
                conf_display = f"Confidence: {best_det['confidence']:.2f}"
            else:
                st.warning("No Region Detected")
                conf_display = "Manual Review Required"

            # 2. SHOW VALUE
            val_str = str(extracted_val)
            if not val_str or val_str == "0" or val_str == "None":
                val_str = "—"
                val_color = "#999"
            else:
                val_color = "#2e7d32"
                
            st.markdown(f"<div class='extracted-value' style='color:{val_color}'>{val_str}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence'>{conf_display}</div></div>", unsafe_allow_html=True)

    # Create 4 columns
    c1, c2, c3, c4 = st.columns(4)
    
    display_field_card('dealer_name', "Dealer Name", final_json['dealer_name'], c1)
    display_field_card('model_name', "Model Name", final_json['model_name'], c2)
    display_field_card('horse_power', "Horse Power", final_json['horse_power'], c3)
    display_field_card('asset_cost', "Asset Cost", final_json['asset_cost'], c4)

    # --- LAYOUT: ROW 3 (AUTH) ---
    st.markdown("---")
    st.subheader("Authentication Checks")
    
    a1, a2, a3 = st.columns(3)
    
    with a1:
        st.markdown("**Dealer Stamp**")
        if final_json['stamp']['present']:
            st.success(f"DETECTED (Conf: {final_json['stamp']['confidence']:.2f})")
            dets = raw_detections.get('stamp', [])
            if dets:
                x1, y1, x2, y2 = dets[0]['bbox']
                st.image(image_rgb[y1:y2, x1:x2], width=200)
        else:
            st.error("NOT FOUND")

    with a2:
        st.markdown("**Signature**")
        if final_json['signature']['present']:
            st.success(f"DETECTED (Conf: {final_json['signature']['confidence']:.2f})")
            dets = raw_detections.get('signature', [])
            if dets:
                x1, y1, x2, y2 = dets[0]['bbox']
                st.image(image_rgb[y1:y2, x1:x2], width=200)
        else:
            st.error("NOT FOUND")
            
    with a3:
        st.markdown("**Performance Metrics**")
        st.metric("Processing Time", f"{proc_time:.2f}s")
        st.metric("Overall Confidence", f"{final_json['metadata']['overall_confidence']:.1%}")

    # --- ROW 4: RAW JSON ---
    with st.expander("View System Output (JSON)"):
        st.json(final_json)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)