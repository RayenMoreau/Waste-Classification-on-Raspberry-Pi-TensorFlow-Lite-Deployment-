# Define your class names. ADJUST THIS ORDER to match your model's output indices (Index 0, Index 1, Index 2, etc.)
CLASS_LABELS = [
    "Paper/Cardboard",      # Index 0
    "Cans (Aluminum/Steel)",# Index 1
    "Plastic (PET/HDPE)",   # Index 2
    # Add more labels here if your model has more output classes
]

MODEL = '/home/alimoro/tflite/waste_classifier_int8.tflite'
TEMP_FILE = '/tmp/captured_image.jpg' # Temporary file for capture

# --- Interpreter Setup ---
try:
    interpreter = tflite.Interpreter(model_path=MODEL)
except ValueError as e:
    print(f"ERROR: Could not load model. Check path: {MODEL}")
    print(e)
    exit()

interpreter.allocate_tensors()
inp_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']
_, h, w, _ = interpreter.get_input_details()[0]['shape']

# Command to capture the image
# --nopreview is essential to prevent the flashing window.
# --timeout 500ms is reliable for capture speed.
CAPTURE_CMD = [
    'rpicam-still',
    '--timeout', '500',
    '--nopreview',
    '--width', '640',
    '--height', '480',
    '--output', TEMP_FILE
]
# --- End Configuration ---

def preprocess(img):
    """Resizes the image and converts it to the INT8 format required by the quantized model."""
   
    # 1. Resize the image to the model's expected dimensions (w, h)
    img = cv2.resize(img, (w, h))
   
    # 2. Convert directly to INT8 (0-255 range), bypassing floating-point division
    # This is required for a quantized TFLite model.
    img = img.astype(np.int8)
   
    # 3. Reshape from (H, W, C) to (1, H, W, C) for the model input
    return np.expand_dims(img, axis=0)

def capture_image_from_cmd():
    """
    Executes rpicam-still to capture a frame to a temporary file,
    reads it with OpenCV, converts it, cleans up the file, and returns the array.
    """
   
    # 1. Execute the capture command
    try:
        subprocess.run(CAPTURE_CMD, check=True, capture_output=True, timeout=10)
    except subprocess.CalledProcessError as e:
        # Prints any error message from the rpicam-still utility
        print(f"ERROR: rpicam-still failed. Ensure camera is working: {e.stderr.decode().strip()}")
        return None
    except FileNotFoundError:
        print("ERROR: rpicam-still command not found. Is 'rpicam-apps' installed?")
        return None
    except subprocess.TimeoutExpired:
        print("ERROR: rpicam-still command timed out.")
        return None

    frame = None
    if os.path.exists(TEMP_FILE):
        try:
            # 2. Read the image from the temporary file using OpenCV (cv2)
            frame = cv2.imread(TEMP_FILE)
           
            # Convert BGR (OpenCV default) to RGB (TFLite standard)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
        except Exception as e:
            print(f"Error reading image file: {e}")
        finally:
            # 3. Clean up the temporary file immediately
            os.remove(TEMP_FILE)
           
    else:
        print(f"ERROR: Temporary file {TEMP_FILE} not found after capture.")
       
    return frame


print('Running... Ctrl-C to stop')
try:
    while True:
        frame = capture_image_from_cmd()
       
        # Check if frame is None (capture failed) or empty (corrupted file)
        if frame is None or frame.size == 0:
            print("Warning: Capture failed or image corrupted. Skipping inference.")
            time.sleep(1) # Wait longer if capture failed
            continue

        # --- Inference ---
        interpreter.set_tensor(inp_idx, preprocess(frame))
        interpreter.invoke()
        probs = interpreter.get_tensor(out_idx)[0]
       
        # Determine the class and confidence
        cls   = int(np.argmax(probs))
        confidence = probs[cls]

        # Look up the human-readable label
        if 0 <= cls < len(CLASS_LABELS):
            label_name = CLASS_LABELS[cls]
        else:
            label_name = f"UNKNOWN INDEX ({cls})"
           
        # --- Output ---
        print(f'Detected: {label_name:<25} | Confidence: {confidence:.2f}')
       
        time.sleep(0.2) # Wait between captures
       
except KeyboardInterrupt:
    print('\nStopped')