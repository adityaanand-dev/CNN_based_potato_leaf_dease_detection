from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "custom_model.h5")
MODEL_PATH2 = os.path.join(BASE_DIR, "model", "Inception.h5")
MODEL_PATH3 = os.path.join(BASE_DIR, "model", "ResNet.h5")

try:
    custom_model = tf.keras.models.load_model(MODEL_PATH)
    inception_model = tf.keras.models.load_model(MODEL_PATH2)
    resnet_model = tf.keras.models.load_model(MODEL_PATH3)
except Exception as e:
    print(f"Warning: Could not load one or more models at startup: {e}")
    custom_model, inception_model, resnet_model = None, None, None

class_names = ["Early Blight", "Late Blight", "Healthy"]

DISEASE_INFO = {
    "Early Blight": {
        "cause": "Caused by the fungus Alternaria solani.",
        "symptoms": "Dark brown to black necrotic spots with concentric rings (target-like pattern) on older leaves.",
        "remedies": [
            "Apply copper-based or chlorothalonil fungicides promptly.",
            "Prune infected lower foliage to reduce spore spreading.",
            "Use drip irrigation to prevent leaves from staying wet.",
            "Practice 2-3 year crop rotation with non-solanaceous crops."
        ]
    },
    "Late Blight": {
        "cause": "Caused by the oomycete Phytophthora infestans.",
        "symptoms": "Rapidly expanding water-soaked dark lesions on leaves and stems with white fungal growth underneath.",
        "remedies": [
            "Apply systemic fungicides (e.g., metalaxyl, cymoxanil) immediately.",
            "Remove and destroy severely affected plants (do not compost).",
            "Ensure excellent field drainage and avoid working in wet foliage.",
            "Monitor regional disease forecasts and early warnings."
        ]
    },
    "Healthy": {
        "cause": "No disease detected.",
        "symptoms": "Leaf tissue is healthy and green with no observable lesions or abnormalities.",
        "remedies": [
            "Continue regular monitoring of soil moisture and plant nutrition.",
            "Ensure proper spacing and weed control for optimal aeration."
        ]
    }
}

# Camera state
is_paused = False

def preprocess_image(img):
    """Ensure image is 3-channel RGB and properly normalized to [0, 1]."""
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_frame(frame):
    """Preprocess frame for model prediction."""
    img = cv2.resize(frame, (256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def find_last_conv_layer(model):
    """Find the last Conv2D layer in the given model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            if len(layer.output.shape) == 4:
                return layer.name
    return None

def grad_cam(model, img_array, layer_name=None):
    """Generate Grad-CAM heatmap with automatic fallback layer detection."""
    available_layer_names = [l.name for l in model.layers]
    if layer_name is None or layer_name not in available_layer_names:
        layer_name = find_last_conv_layer(model)

    if layer_name is None:
        return np.zeros((256, 256), dtype=np.float32)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
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

def overlay_heatmap_img(heatmap, img, opacity=0.4):
    """Overlay heatmap on RGB image preserving true RGB channel ordering."""
    img_rgb = np.array(img.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_rgb, 1 - opacity, heatmap_rgb, opacity, 0)
    return superimposed_img

def generate_frames():
    global is_paused
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Generate placeholder frame if camera unavailable
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_frame, "Webcam unavailable or in use", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', black_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        while True:
            if not is_paused:
                success, frame = cap.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_frame = preprocess_frame(rgb_frame)

                if inception_model is not None:
                    predictions = inception_model.predict(input_frame, verbose=0)
                    class_idx = int(np.argmax(predictions[0]))
                    confidence = float(predictions[0][class_idx])
                    prediction_text = f"Prediction: {class_names[class_idx]} ({confidence * 100:.1f}%)"
                    color = (0, 255, 0) if class_idx == 2 else (0, 0, 255)
                    cv2.putText(frame, prediction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    finally:
        cap.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and prediction."""
    if 'image' not in request.files:
        return render_template('error.html', error="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('error.html', error="No file selected.")

    # Unique filename to prevent overwrite and browser caching
    unique_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    safe_name = f"upload_{unique_id}{ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    file.save(file_path)

    try:
        img = Image.open(file_path)
        img_array = preprocess_image(img)

        # Predictions from all three models
        predictions = custom_model.predict(img_array, verbose=0) if custom_model else np.zeros((1, 3))
        class_idx = int(np.argmax(predictions[0]))

        predictions2 = inception_model.predict(img_array, verbose=0) if inception_model else np.zeros((1, 3))
        class_idx2 = int(np.argmax(predictions2[0]))

        predictions3 = resnet_model.predict(img_array, verbose=0) if resnet_model else np.zeros((1, 3))
        class_idx3 = int(np.argmax(predictions3[0]))

        # Generate Grad-CAM heatmap using Inception model (or Custom if Inception unavailable)
        target_cam_model = inception_model if inception_model else custom_model
        layer_target = "conv2d_116" if inception_model and "conv2d_116" in [l.name for l in inception_model.layers] else None
        heatmap = grad_cam(target_cam_model, img_array, layer_target)
        result_img = overlay_heatmap_img(heatmap, img)

        grad_cam_filename = f"grad_cam_{unique_id}.jpg"
        result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], grad_cam_filename)
        Image.fromarray(result_img).save(result_img_path)

        # Primary disease diagnosis
        primary_disease = class_names[class_idx2]
        disease_info = DISEASE_INFO[primary_disease]

        return render_template(
            'results.html',
            custom_model_prediction=class_names[class_idx],
            custom_model_confidence=f"{predictions[0][class_idx] * 100:.2f}%",
            inception_model_prediction=class_names[class_idx2],
            inception_model_confidence=f"{predictions2[0][class_idx2] * 100:.2f}%",
            resnet_model_prediction=class_names[class_idx3],
            resnet_model_confidence=f"{predictions3[0][class_idx3] * 100:.2f}%",
            grad_cam_path=f"/uploads/{grad_cam_filename}",
            primary_disease=primary_disease,
            cause=disease_info["cause"],
            symptoms=disease_info["symptoms"],
            remedies=disease_info["remedies"]
        )
    except Exception as e:
        return render_template('error.html', error=f"Failed to process image: {str(e)}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"paused": is_paused, "status": "Paused" if is_paused else "Playing"})

@app.route("/realtime")
def realtime():
    return render_template('realtime.html')

if __name__ == '__main__':
    app.run(debug=True)
