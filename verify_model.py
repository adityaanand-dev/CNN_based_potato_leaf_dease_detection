"""
Validation and Benchmark Script for Potato Leaf Disease Detection Models
Evaluates Custom CNN, InceptionV3, and ResNet50 models on test dataset samples.
"""
import os
import sys
import time
import numpy as np
from PIL import Image

def main():
    print("=" * 60)
    print("🌿 Potato Leaf Disease Detection - Model Verification Suite")
    print("=" * 60)

    try:
        import tensorflow as tf
        import cv2
        print(f"TensorFlow version: {tf.__version__}")
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"Required package missing: {e}")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "StreamlitApp", "model")

    paths = {
        "Custom CNN": os.path.join(model_dir, "custom_model.h5"),
        "InceptionV3": os.path.join(model_dir, "Inception.h5"),
        "ResNet50": os.path.join(model_dir, "ResNet.h5")
    }

    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            t0 = time.time()
            try:
                models[name] = tf.keras.models.load_model(path)
                load_time = time.time() - t0
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✅ {name:12s} loaded in {load_time:.2f}s ({size_mb:.1f} MB, {len(models[name].layers)} layers)")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        else:
            print(f"⚠️ Model file not found: {path}")

    class_names = ["Early Blight", "Late Blight", "Healthy"]
    class_dirs = {
        "Early Blight": os.path.join(base_dir, "Potato", "Test", "Potato___Early_blight"),
        "Late Blight": os.path.join(base_dir, "Potato", "Test", "Potato___Late_blight"),
        "Healthy": os.path.join(base_dir, "Potato", "Test", "Potato___healthy")
    }

    # Verify test directories exist
    missing_dirs = [d for d in class_dirs.values() if not os.path.exists(d)]
    if missing_dirs:
        print(f"\n⚠️ Some test directories missing: {missing_dirs}")
        return

    print("\n" + "=" * 60)
    print("📊 Evaluating Test Predictions Across Classes (5 samples each)")
    print("=" * 60)

    results = {m_name: {"correct": 0, "total": 0, "latencies": []} for m_name in models}

    for true_class, folder in class_dirs.items():
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        print(f"\n📁 Class: {true_class} ({len(images)} samples)")

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            for m_name, model in models.items():
                t0 = time.time()
                preds = model.predict(img_array, verbose=0)[0]
                latency_ms = (time.time() - t0) * 1000
                pred_idx = int(np.argmax(preds))
                pred_class = class_names[pred_idx]
                conf = preds[pred_idx] * 100

                is_correct = (pred_class == true_class)
                results[m_name]["total"] += 1
                if is_correct:
                    results[m_name]["correct"] += 1
                results[m_name]["latencies"].append(latency_ms)

                status = "✓" if is_correct else "✗"
                print(f"  {status} [{m_name:11s}] -> {pred_class:12s} ({conf:5.1f}%) in {latency_ms:4.1f}ms")

    print("\n" + "=" * 60)
    print("📈 Benchmark Summary")
    print("=" * 60)
    for m_name, res in results.items():
        acc = (res["correct"] / res["total"]) * 100 if res["total"] > 0 else 0
        avg_lat = np.mean(res["latencies"]) if res["latencies"] else 0
        print(f"• {m_name:12s}: Accuracy = {acc:5.1f}% ({res['correct']}/{res['total']}), Avg Latency = {avg_lat:.1f}ms")

    # Grad-CAM verification
    print("\n" + "=" * 60)
    print("🔥 Grad-CAM Generation Test")
    print("=" * 60)

    sample_img_path = os.path.join(class_dirs["Early Blight"], os.listdir(class_dirs["Early Blight"])[0])
    img = Image.open(sample_img_path).convert('RGB')
    img_array = np.expand_dims(np.array(img.resize((256, 256)), dtype=np.float32) / 255.0, axis=0)

    for m_name, model in models.items():
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                if len(layer.output.shape) == 4:
                    last_conv = layer.name
                    break

        if last_conv:
            try:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[model.get_layer(last_conv).output, model.output]
                )
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    loss = predictions[:, np.argmax(predictions[0])]
                gradients = tape.gradient(loss, conv_outputs)
                pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = np.zeros(conv_outputs.shape[:-1], dtype=np.float32)
                for i in range(conv_outputs.shape[-1]):
                    heatmap += pooled_gradients[i].numpy() * conv_outputs[:, :, i].numpy()
                heatmap = np.maximum(heatmap, 0)
                max_v = np.max(heatmap)
                if max_v > 0:
                    heatmap /= max_v
                print(f"✅ Grad-CAM for {m_name} (layer: {last_conv}): Heatmap shape {heatmap.shape}, min={heatmap.min():.2f}, max={heatmap.max():.2f}")
            except Exception as e:
                print(f"❌ Grad-CAM for {m_name} failed: {e}")
        else:
            print(f"⚠️ No conv layer found for {m_name}")

    print("\n✨ Verification finished!")

if __name__ == "__main__":
    main()
