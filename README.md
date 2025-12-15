This project implements a real-time waste classification system on a Raspberry Pi using a quantized TensorFlow Lite (INT8) model.
The system captures images using the Raspberry Pi Camera Module, preprocesses them, runs TFLite inference, and prints the predicted waste category.

Features:
-Quantized INT8 TensorFlow Lite model for edge efficiency
-Real-time image capture using rpicam-still
-Fully implemented in Python 3 with tflite_runtime
-Works on Raspberry Pi OS (Trixie) — including manual TFLite build
-Achieves ~5 FPS inference loop
-Clean preprocessing + inference pipeline

Tech Stack:
-Raspberry Pi 4 / Zero 2W
-Raspberry Pi OS Trixie (64-bit)
-TensorFlow Lite Runtime (custom-built)
-OpenCV
-Python 3.11 / Virtual Environment
-rpicam-still (libcamera)

How It Works:
1. Capture an image
Uses the Raspberry Pi camera via:
rpicam-still --timeout 500 --nopreview --width 640 --height 480 --output /tmp/captured_image.jpg
A Python subprocess call executes this inside your script.

2. Preprocess
Resize to the model's input resolution
Convert to INT8
Shape → (1, height, width, 3)

3. Perform inference
Load TFLite model
Allocate tensors
Set input, invoke model
Read class probabilities

4. Print results
Real-time confidence + label.

Model Labels:
Update these depending on your trained model:
0 — Paper
1 — Cans 
2 — Plastic

❗ Notes & Troubleshooting
-TFLite on Raspberry Pi OS (Trixie)
TensorFlow Lite wheels do not match Python versions on Trixie.
Solution: build tflite_runtime manually inside a venv.

Camera Not Working?
libcamera tools may fail, but rpicam-still works reliably.
