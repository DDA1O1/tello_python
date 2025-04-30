# Tello Drone Real-Time Object Detection

A Python-based web application that performs real-time object detection using a DJI Tello drone camera feed. The application uses MobileNet SSD for object detection and provides a web interface to view the processed video stream.

## Features

- Real-time video streaming from Tello drone
- Object detection using MobileNet SSD model
- Web interface for viewing the processed video feed
- Automatic drone connection and video stream management
- Error handling and recovery mechanisms
- Support for detecting 20 different object classes

## Prerequisites

- Python 3.6 or higher
- DJI Tello drone
- Stable WiFi connection
- Required model files:
  - `MobileNetSSD_deploy.caffemodel`
  - `MobileNetSSD_deploy.prototxt.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd tello-object-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install fastapi uvicorn opencv-python djitellopy numpy
```

## Project Structure

```
.
├── tello_object_detection_web.py  # Main application file
├── MobileNetSSD_deploy.caffemodel # Pre-trained model weights
├── MobileNetSSD_deploy.prototxt.txt # Model architecture
└── README.md
```

## Usage

1. Ensure your computer is connected to the Tello drone's WiFi network.

2. Start the application:
```bash
python tello_object_detection_web.py
```
   Or using uvicorn directly:
```bash
uvicorn tello_object_detection_web:app --host 0.0.0.0 --port 8000
```

3. Open a web browser and navigate to:
```
http://localhost:8000
```

## Object Detection Classes

The system can detect the following objects:
- Person
- Bicycle
- Car
- Motorcycle
- Airplane
- Bus
- Train
- Truck
- Boat
- Traffic Light
- Fire Hydrant
- Stop Sign
- Parking Meter
- Bench
- Bird
- Cat
- Dog
- Horse
- Sheep
- Cow

## Configuration

Key configuration parameters in `tello_object_detection_web.py`:
- `confidence_threshold`: Minimum confidence for object detection (default: 0.4)
- `prototxt_path`: Path to the model architecture file
- `model_path`: Path to the model weights file

## Error Handling

The application includes robust error handling for:
- Drone connection failures
- Video stream interruptions
- Model loading issues
- Frame processing errors

## Safety Features

- Battery level monitoring
- Automatic stream cleanup on shutdown
- Connection retry mechanisms
- Error throttling to prevent log flooding

## Troubleshooting

1. If the drone doesn't connect:
   - Ensure you're connected to the Tello's WiFi network
   - Check if the drone's battery is sufficiently charged
   - Restart the drone and try again

2. If the video stream doesn't appear:
   - Check the server logs for error messages
   - Ensure the model files are present in the correct location
   - Verify that your browser supports MJPEG streams

3. If object detection is slow:
   - Consider reducing the frame resolution
   - Adjust the confidence threshold
   - Ensure your system meets the minimum requirements
