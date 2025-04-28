# tello_object_detection.py

import time
import cv2
import numpy as np
from djitellopy import Tello

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
           "sofa", "train", "tvmonitor"] # Note: 'pen' and 'mouse' are not default classes

# --- Initialization ---
print("[INFO] Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[INFO] Model loaded successfully.")
except cv2.error as e:
    print(f"[ERROR] Failed to load model: {e}")
    print(f"[ERROR] Ensure '{prototxt_path}' and '{model_path}' are in the same directory as the script.")
    exit()

print("[INFO] Initializing Tello drone...")
tello = Tello()

# --- Drone Connection and Stream ---
keep_trying_connect = True
while keep_trying_connect:
    try:
        tello.connect()
        print("[INFO] Tello connected successfully.")
        keep_trying_connect = False
    except Exception as e:
        print(f"[ERROR] Failed to connect to Tello: {e}. Retrying in 5 seconds...")
        time.sleep(5)

# Query battery
print(f"[INFO] Tello Battery: {tello.get_battery()}%")
if tello.get_battery() < 20:
    print("[WARNING] Battery low! Consider charging before flight.")

# Start video stream
keep_trying_stream = True
while keep_trying_stream:
    try:
        tello.streamon()
        print("[INFO] Video stream started.")
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
        print(f"[ERROR] Failed to start stream: {e}. Retrying in 5 seconds...")
        # Attempt to gracefully handle potential connection issues
        try:
            tello.connect() # Re-establish connection just in case
        except:
            pass # Ignore connection errors here, focus on stream retry
        time.sleep(5)


# --- Optional: Takeoff and Initial Positioning ---
# Uncomment the following lines if you want the drone to take off automatically
# print("[ACTION] Taking off...")
# tello.takeoff()
# time.sleep(2) # Give drone time to stabilize
# print("[ACTION] Moving up slightly...")
# tello.move_up(30) # Move up 30 cm
# time.sleep(1)


# --- Main Loop: Video Processing and Detection ---
print("[INFO] Starting detection loop. Press 'q' to quit and land.")
should_stop = False

while not should_stop:
    try:
        # 1. Get Frame
        frame_original = frame_read.frame

        # Check if frame is valid
        if frame_original is None:
            print("[WARNING] Received empty frame. Skipping...")
            time.sleep(0.1) # Wait a bit before trying again
            # Attempt to reconnect stream if persistent issue (optional)
            # try:
            #    tello.streamon()
            #    frame_read = tello.get_frame_read()
            # except Exception as stream_err:
            #    print(f"[ERROR] Stream error during loop: {stream_err}")
            continue

        frame = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)

        # Get frame dimensions
        (h, w) = frame.shape[:2]

        # 2. Preprocess Frame for DNN
        # Create a blob from the image (resizing and mean subtraction)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # 3. Perform Inference
        net.setInput(blob)
        detections = net.forward()

        # 4. Process Detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > confidence_threshold:
                # Get class label index and coordinates
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Get class label string
                if idx < len(CLASSES):
                    label = CLASSES[idx]
                    display_text = f"{label}: {confidence:.2f}"
                    # Draw bounding box and label
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, display_text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                     print(f"[WARNING] Detected class index {idx} out of bounds for known CLASSES.")


        # 5. Display Output Frame
        cv2.imshow("Tello Object Detection", frame)

        # 6. Check for Quit Key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed. Landing and stopping...")
            should_stop = True
        # Add other key controls if needed (e.g., manual flight)
        # elif key == ord('t'): # Example: Takeoff on 't'
        #     print("[ACTION] Taking off...")
        #     tello.takeoff()
        # elif key == ord('l'): # Example: Land on 'l'
        #     print("[ACTION] Landing...")
        #     tello.land()


    except Exception as e:
        print(f"[ERROR] An error occurred in the main loop: {e}")
        # Optional: Decide if you want to land on any error
        # should_stop = True
        time.sleep(1) # Avoid flooding console if error repeats rapidly

# --- Cleanup ---
print("[INFO] Cleaning up...")
cv2.destroyAllWindows() # Close the display window

# Ensure landing command is sent, even if takeoff wasn't automatic
try:
    print("[ACTION] Sending land command...")
    tello.land()
    time.sleep(2) # Give it time to land
except Exception as e:
    print(f"[ERROR] Failed to send land command: {e}")

# Turn off stream
try:
    print("[INFO] Turning off video stream...")
    tello.streamoff()
except Exception as e:
    print(f"[ERROR] Failed to turn off stream: {e}")

# Optional: End connection explicitly (djitellopy often handles this on exit)
# try:
#     tello.end()
# except Exception as e:
#     print(f"[ERROR] Failed to end Tello connection: {e}")

print("[INFO] Script finished.")