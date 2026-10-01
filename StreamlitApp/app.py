import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import time
from collections import Counter
from Home import home
from About import about

st.set_page_config(page_title="Potato Leaf Disease Detection", page_icon="🥔", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "custom_model.h5")
MODEL_PATH2 = os.path.join(BASE_DIR, "model", "Inception.h5")
MODEL_PATH3 = os.path.join(BASE_DIR, "model", "ResNet.h5")

class_names = ["Early Blight", "Late Blight", "Healthy"]

DISEASE_INFO = {
    "Early Blight": {
        "badge": "⚠️ High Attention",
        "color": "#e67e22",
        "cause": "Fungal infection primarily caused by Alternaria solani.",
        "symptoms": "Concentric rings (target board pattern) forming brown-to-black necrotic lesions on older foliage, yellowing surrounding tissue.",
        "treatment": [
            "Apply targeted fungicides (e.g., Mancozeb, Chlorothalonil, or copper-based sprays).",
            "Prune and destroy infected lower leaves immediately to stop spore dispersal.",
            "Water at the base using drip irrigation; avoid overhead sprinklers to keep leaves dry.",
            "Implement a 2-3 year crop rotation with non-solanaceous crops."
        ]
    },
    "Late Blight": {
        "badge": "🚨 Critical Emergency",
        "color": "#e74c3c",
        "cause": "Aggressive water-mold pathogen Phytophthora infestans (the historical cause of the Irish potato famine).",
        "symptoms": "Irregular water-soaked pale-to-dark lesions rapidly expanding across leaves and stems, with white velvety mildew on leaf undersides in humid weather.",
        "treatment": [
            "Immediately apply systemic fungicides (e.g., Metalaxyl, Cymoxanil, or Dimethomorph).",
            "Uproot and destroy heavily infected plants; do NOT compost diseased foliage.",
            "Improve field aeration and drainage; avoid working in the field while plants are wet.",
            "Monitor nearby fields and forecast warnings regularly."
        ]
    },
    "Healthy": {
        "badge": "✅ Healthy Leaf",
        "color": "#27ae60",
        "cause": "No disease detected.",
        "symptoms": "Vibrant uniform green coloration, robust tissue structure with no signs of fungal or bacterial lesions.",
        "treatment": [
            "Maintain balanced N-P-K fertilization and consistent soil moisture.",
            "Continue periodic leaf inspection, particularly after rainy or humid periods.",
            "Ensure proper weed control and adequate field spacing for sunlight penetration."
        ]
    }
}

# Cache models to eliminate reload delay on every Streamlit rerun
@st.cache_resource(show_spinner="Loading trained CNN models...")
def load_all_models():
    custom = tf.keras.models.load_model(MODEL_PATH)
    inception = tf.keras.models.load_model(MODEL_PATH2)
    resnet = tf.keras.models.load_model(MODEL_PATH3)
    return custom, inception, resnet

try:
    model, model2, model3 = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    model, model2, model3 = None, None, None

def preprocess_image(img):
    """Ensure image is 3-channel RGB and properly normalized to [0, 1]."""
    img_rgb = img.convert('RGB')
    img_resized = img_rgb.resize((256, 256))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_frame(frame):
    """Preprocess OpenCV RGB frame for model prediction."""
    img = cv2.resize(frame, (256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def find_last_conv_layer(m):
    """Dynamically discover the last 4D Convolutional layer in the model."""
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            if len(layer.output.shape) == 4:
                return layer.name
    return None

def grad_cam(m, img_array, layer_name=None):
    """Compute Grad-CAM heatmap with automatic layer fallback and division-by-zero protection."""
    available_layer_names = [l.name for l in m.layers]
    if layer_name is None or layer_name not in available_layer_names:
        layer_name = find_last_conv_layer(m)

    if layer_name is None:
        return np.zeros((256, 256), dtype=np.float32)

    grad_model = tf.keras.models.Model(
        inputs=m.input,
        outputs=[m.get_layer(layer_name).output, m.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    gradients = tape.gradient(loss, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:-1], dtype=np.float32)

    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_gradients[i].numpy() * conv_outputs[:, :, i].numpy()

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap

def overlay_heatmap_Img(heatmap, img, opacity=0.4):
    """
    Overlay Grad-CAM heatmap on PIL image with correct RGB color channels.
    Fixes OpenCV BGR vs PIL/Streamlit RGB channel swap bug.
    """
    img_rgb = np.array(img.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_rgb, 1 - opacity, heatmap_rgb, opacity, 0)
    return superimposed_img

def render_prediction_summary(img):
    """Renders comprehensive predictions across Custom CNN, Inception, and ResNet models."""
    if model is None or model2 is None or model3 is None:
        st.error("Models could not be loaded.")
        return

    img_array = preprocess_image(img)

    with st.spinner("Analyzing image across deep learning models..."):
        p_custom = model.predict(img_array, verbose=0)[0]
        p_incept = model2.predict(img_array, verbose=0)[0]
        p_resnet = model3.predict(img_array, verbose=0)[0]

    idx_custom = int(np.argmax(p_custom))
    idx_incept = int(np.argmax(p_incept))
    idx_resnet = int(np.argmax(p_resnet))

    st.markdown("### 📊 Model Predictions & Confidence")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"#### 🧠 Custom CNN (95.7% Acc)")
        st.success(f"**{class_names[idx_custom]}**")
        st.progress(float(p_custom[idx_custom]))
        st.caption(f"Confidence: **{p_custom[idx_custom] * 100:.2f}%**")
        with st.expander("Probabilities breakdown"):
            for name, prob in zip(class_names, p_custom):
                st.write(f"- {name}: {prob*100:.2f}%")

    with col2:
        st.markdown(f"#### 🌐 InceptionV3 (98.0% Acc)")
        st.success(f"**{class_names[idx_incept]}**")
        st.progress(float(p_incept[idx_incept]))
        st.caption(f"Confidence: **{p_incept[idx_incept] * 100:.2f}%**")
        with st.expander("Probabilities breakdown"):
            for name, prob in zip(class_names, p_incept):
                st.write(f"- {name}: {prob*100:.2f}%")

    with col3:
        st.markdown(f"#### ⚡ ResNet50 (88.0% Acc)")
        st.success(f"**{class_names[idx_resnet]}**")
        st.progress(float(p_resnet[idx_resnet]))
        st.caption(f"Confidence: **{p_resnet[idx_resnet] * 100:.2f}%**")
        with st.expander("Probabilities breakdown"):
            for name, prob in zip(class_names, p_resnet):
                st.write(f"- {name}: {prob*100:.2f}%")

    # Primary consensus diagnosis (weighted by highest confidence among top models)
    primary_idx = idx_incept if p_incept[idx_incept] >= p_custom[idx_custom] else idx_custom
    primary_disease = class_names[primary_idx]
    info = DISEASE_INFO[primary_disease]

    st.markdown("---")
    st.markdown(f"### 🩺 Diagnosis: <span style='color:{info['color']}'>{info['badge']} - {primary_disease}</span>", unsafe_allow_html=True)
    st.write(f"**Cause:** {info['cause']}")
    st.write(f"**Symptoms:** {info['symptoms']}")
    st.markdown("**Actionable Recommendations & Treatment:**")
    for r in info["treatment"]:
        st.markdown(f"- {r}")

    # Grad-CAM Section
    st.markdown("---")
    st.markdown("### 🔥 Explainable AI: Grad-CAM Activation Heatmap")
    
    col_cam_opts, col_cam_img = st.columns([1, 2])
    with col_cam_opts:
        st.write("Grad-CAM highlights the exact regions of the leaf that led to the model's decision.")
        selected_model_name = st.selectbox(
            "Select model for Grad-CAM",
            ["InceptionV3 (Recommended)", "Custom CNN", "ResNet50"]
        )
        opacity = st.slider("Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

        if "Inception" in selected_model_name:
            active_model = model2
            active_layer = "conv2d_116" if "conv2d_116" in [l.name for l in model2.layers] else None
        elif "Custom" in selected_model_name:
            active_model = model
            active_layer = None
        else:
            active_model = model3
            active_layer = None

    with col_cam_img:
        heatmap = grad_cam(active_model, img_array, active_layer)
        result_img = overlay_heatmap_Img(heatmap, img, opacity=opacity)
        st.image(result_img, caption=f"Grad-CAM overlay using {selected_model_name}", use_container_width=True)

def upload():
    st.header("📤 Upload Potato Leaf Image")
    uploaded_file = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
            render_prediction_summary(img)
        except Exception as e:
            st.error(f"Error processing image: {e}")

def camera():
    st.header("📸 Capture Leaf Image from Camera")
    camera_image = st.camera_input("Take a photo of a potato leaf")
    if camera_image is not None:
        try:
            img = Image.open(camera_image)
            st.image(img, caption="Captured Image", use_container_width=True)
            render_prediction_summary(img)
        except Exception as e:
            st.error(f"Error processing captured image: {e}")

def realTime():
    st.header("📹 Real-Time Webcam Stream Analysis")
    st.write("Ensure your webcam is connected. The stream runs predictions and highlights infected regions.")

    if "video_state" not in st.session_state:
        st.session_state.video_state = "stopped"
    if "frame_predictions" not in st.session_state:
        st.session_state.frame_predictions = []

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Stream"):
            st.session_state.video_state = "playing"
    with col2:
        if st.button("⏹️ Stop Stream"):
            st.session_state.video_state = "stopped"

    FRAME_WINDOW = st.empty()

    if st.session_state.video_state == "playing":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access webcam. Please check your camera permissions or whether another app is using it.")
            st.session_state.video_state = "stopped"
            return

        frame_rate = 10
        frame_delay = 1 / frame_rate
        frame_count = 0

        while st.session_state.video_state == "playing":
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from webcam. Stream stopped.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_array = preprocess_frame(rgb_frame)

            predictions = model2.predict(img_array, verbose=0)
            class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_idx])

            st.session_state.frame_predictions.append((class_idx, confidence))
            frame_count += 1

            heatmap = grad_cam(model2, img_array, "conv2d_116")
            heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

            max_h = np.max(heatmap_resized)
            threshold = 0.5 * max_h if max_h > 0 else 0.67
            activated_region = heatmap_resized > threshold

            y_indices, x_indices = np.where(activated_region)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = max(0, x_min), min(frame.shape[1], x_max)
                y_min, y_max = max(0, y_min), min(frame.shape[0], y_max)
                cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 180, 255), 2)

            pred_label = f"{class_names[class_idx]} ({confidence * 100:.1f}%)"
            cv2.putText(rgb_frame, pred_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            FRAME_WINDOW.image(rgb_frame, channels="RGB")

            if frame_count >= 15:
                pred_counts = Counter([p[0] for p in st.session_state.frame_predictions])
                most_common_idx, _ = pred_counts.most_common(1)[0]
                avg_conf = np.mean([p[1] for p in st.session_state.frame_predictions if p[0] == most_common_idx])
                st.success(f"Confirmed Diagnosis: **{class_names[most_common_idx]}** (Avg Confidence: {avg_conf * 100:.2f}%)")
                st.session_state.frame_predictions = []
                frame_count = 0

            time.sleep(frame_delay)

        cap.release()
        FRAME_WINDOW.empty()
        st.write("Video stream stopped.")

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    st.sidebar.title("🌿 Navigation")
    option = st.sidebar.selectbox(
        "Choose Functionality",
        ["Home", "Upload Image", "Use Camera", "Real-Time Video", "About"],
        index=0
    )

    if option == "Home":
        home()
    elif option == "Upload Image":
        upload()
    elif option == "Use Camera":
        camera()
    elif option == "Real-Time Video":
        realTime()
    elif option == "About":
        about()

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Model Benchmarks:**\n- InceptionV3: ~98.0% Test Acc\n- Custom CNN: ~95.7% Test Acc\n- ResNet50: ~88.0% Test Acc")
    st.sidebar.caption("Potato Leaf Disease Detection System")
