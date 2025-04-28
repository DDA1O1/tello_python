# tello_object_detection_web.py

import time
import cv2
import numpy as np
from djitellopy import Tello
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import asyncio # Needed for async functions in FastAPI

# --- Configuration ---
# Paths to the MobileNet SSD model files
prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
model_path = "MobileNetSSD_deploy.caffemodel"
# Minimum confidence threshold for detections
confidence_threshold = 0.4 # You can adjust this (0.0 to 1.0)

# COCO dataset class labels (MobileNet SSD was trained on this)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Global variables for Tello and model (to be initialized)
tello = None
net = None
frame_read = None
keep_running = True # Flag to control the main processing loop

# --- Initialization Functions ---
def initialize_model():
    """Loads the DNN model."""
    global net
    print("[INFO] Loading model...")
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("[INFO] Model loaded successfully.")
        return True
    except cv2.error as e:
        print(f"[ERROR] Failed to load model: {e}")
        print(f"[ERROR] Ensure '{prototxt_path}' and '{model_path}' exist.")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading the model: {e}")
        return False

def initialize_tello():
    """Connects to the Tello drone and starts the video stream."""
    global tello, frame_read
    print("[INFO] Initializing Tello drone...")
    tello = Tello()

    # --- Drone Connection ---
    keep_trying_connect = True
    max_connect_retries = 3
    connect_attempts = 0
    while keep_trying_connect and connect_attempts < max_connect_retries:
        connect_attempts += 1
        try:
            print(f"[INFO] Attempting Tello connection ({connect_attempts}/{max_connect_retries})...")
            tello.connect()
            print("[INFO] Tello connected successfully.")
            keep_trying_connect = False
        except Exception as e:
            print(f"[ERROR] Failed to connect to Tello: {e}.")
            if connect_attempts < max_connect_retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("[ERROR] Max connection retries reached. Aborting.")
                return False

    # Query battery
    try:
        battery = tello.get_battery()
        print(f"[INFO] Tello Battery: {battery}%")
        if battery < 20:
            print("[WARNING] Battery low! Consider charging before flight.")
    except Exception as e:
        print(f"[WARNING] Could not query battery: {e}")


    # --- Start video stream ---
    keep_trying_stream = True
    max_stream_retries = 3
    stream_attempts = 0
    while keep_trying_stream and stream_attempts < max_stream_retries:
        stream_attempts += 1
        try:
            print(f"[INFO] Attempting to start video stream ({stream_attempts}/{max_stream_retries})...")
            tello.streamon()
            print("[INFO] Video stream started command sent.")
            # It might take a moment for the stream to become active
            time.sleep(2) # Add a small delay
            frame_read = tello.get_frame_read()
            # Check if frame is valid initially
            if frame_read.frame is None:
                print("[ERROR] Failed to get initial frame. Retrying stream...")
                tello.streamoff() # Try turning off before retrying
                time.sleep(2)
                continue # Retry streamon
            print("[INFO] Receiving video frames...")
            keep_trying_stream = False
        except Exception as e:
            print(f"[ERROR] Failed to start stream: {e}.")
            # Attempt to gracefully handle potential connection issues
            if stream_attempts < max_stream_retries:
                print("Retrying in 5 seconds...")
                try:
                    tello.streamoff() # Ensure stream is off before retry
                    time.sleep(1)
                    tello.connect() # Re-establish connection just in case
                except Exception as inner_e:
                    print(f"[WARNING] Error during stream retry preparation: {inner_e}")
                time.sleep(5)
            else:
                print("[ERROR] Max stream retries reached. Aborting.")
                # Cleanup potential partial connection
                try: tello.streamoff()
                except: pass
                return False

    # --- Optional: Takeoff and Initial Positioning ---
    # Keep this commented out for safety when running in Docker first
    # You might want to add API endpoints to control takeoff/landing later
    # print("[ACTION] Taking off...")
    # try:
    #     tello.takeoff()
    #     time.sleep(5) # Give drone time to stabilize
    #     print("[ACTION] Moving up slightly...")
    #     tello.move_up(30) # Move up 30 cm
    #     time.sleep(2)
    # except Exception as e:
    #     print(f"[ERROR] Failed during takeoff/initial movement: {e}")
    #     # Consider landing if takeoff failed partially
    #     try: tello.land()
    #     except: pass
    #     return False

    return True


# --- Frame Generation for Streaming ---
async def generate_frames():
    """Generator function to yield processed frames for MJPEG stream."""
    global frame_read, net, keep_running
    last_error_time = 0
    error_throttle_period = 5 # Seconds

    while keep_running:
        try:
            if frame_read is None:
                print("[WARNING] frame_read object not available yet. Waiting...")
                await asyncio.sleep(0.5)
                continue

            # 1. Get Frame
            frame_original = frame_read.frame

            # Check if frame is valid
            if frame_original is None:
                # Avoid flooding logs if frames stop temporarily
                current_time = time.time()
                if current_time - last_error_time > error_throttle_period:
                    print("[WARNING] Received empty frame. Skipping...")
                    last_error_time = current_time
                await asyncio.sleep(0.05) # Wait a bit before trying again
                continue

            # Reset error timer on successful frame
            last_error_time = 0

            # Convert frame BGR for OpenCV processing
            frame = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)
            (h, w) = frame.shape[:2]

            # 2. Preprocess Frame for DNN
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # 3. Perform Inference
            net.setInput(blob)
            detections = net.forward()

            # 4. Process Detections and Draw Bounding Boxes
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    if 0 <= idx < len(CLASSES):
                        label = CLASSES[idx]
                        display_text = f"{label}: {confidence:.2f}"
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, display_text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print(f"[WARNING] Detected class index {idx} out of bounds.")

            # 5. Encode Frame as JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                print("[WARNING] Failed to encode frame to JPEG.")
                await asyncio.sleep(0.05)
                continue

            # 6. Yield the frame in MJPEG format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

            # Small delay to prevent overwhelming the CPU if frame rate is very high
            # and allow other async tasks to run. Adjust as needed.
            await asyncio.sleep(0.01)

        except Exception as e:
            # Log errors but try to continue
            current_time = time.time()
            if current_time - last_error_time > error_throttle_period:
                 print(f"[ERROR] An error occurred in the frame generation loop: {e}")
                 last_error_time = current_time
            await asyncio.sleep(1) # Longer sleep on error

# --- FastAPI Application ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize Model and Tello on server startup."""
    global keep_running
    if not initialize_model():
        print("[FATAL] Could not initialize model. Shutting down.")
        keep_running = False # Prevent stream from starting
        # In a real deployment, you might want to exit the process here
        # For simplicity, we let it start but the stream won't work.
        return

    if not initialize_tello():
        print("[FATAL] Could not initialize Tello. Shutting down.")
        keep_running = False
        # Attempt cleanup even if initialization failed partially
        cleanup()
        return
    print("[INFO] Startup complete. Tello and model ready.")


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup resources on server shutdown."""
    global keep_running
    print("[INFO] Shutdown event triggered.")
    keep_running = False # Signal the frame generator to stop
    cleanup()

def cleanup():
    """Handles landing the drone and stopping the stream."""
    global tello
    print("[INFO] Cleaning up resources...")
    if tello:
        
        # Turn off stream
        try:
            print("[INFO] Turning off video stream...")
            tello.streamoff()
            print("[INFO] Stream off command sent.")
        except Exception as e:
            print(f"[ERROR] Failed to turn off stream: {e}")

        # Optional: End connection explicitly (djitellopy often handles this, but good practice)
        # try:
        #     print("[INFO] Ending Tello connection...")
        #     tello.end()
        # except Exception as e:
        #     print(f"[ERROR] Failed to end Tello connection: {e}")
    else:
        print("[INFO] Tello object not initialized, skipping cleanup steps.")

    print("[INFO] Cleanup finished.")


@app.get("/video_feed")
async def video_feed():
    """Endpoint to stream the processed video feed."""
    if not keep_running or frame_read is None or net is None:
         return Response(content="Error: Tello stream or model not initialized.", status_code=503) # Service Unavailable
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def root():
    """Serves a simple HTML page to display the video stream."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tello Object Detection Stream</title>
    </head>
    <body>
        <h1>Tello Object Detection Stream</h1>
        <p>If you see this, the server is running. The video feed should appear below.</p>
        <img src="/video_feed" width="720" height="540" alt="Tello Video Feed"/>
        <p>Note: Ensure the Tello drone is connected and the stream has started successfully.</p>
        <p>Server logs provide connection status.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# --- Main Execution ---
if __name__ == "__main__":
    print("[INFO] Starting FastAPI server...")
    # Note: Running uvicorn directly like this is good for development.
    # For production within Docker, you might run `uvicorn tello_object_detection_web:app --host 0.0.0.0 --port 8000`
    # The host 0.0.0.0 makes it accessible outside the container.
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Code here will run *after* uvicorn stops (e.g., on Ctrl+C)
    # However, the FastAPI shutdown event is the more reliable place for cleanup.
    print("[INFO] FastAPI server stopped.")