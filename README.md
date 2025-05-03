# Face Recognition Attendance System

A standalone face recognition system for tracking attendance using computer vision.

## Overview

This system uses your computer's webcam to identify people in real-time, automatically marking their attendance. It combines YOLOv8 for face detection with face_recognition for facial recognition.

## Complete Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/riskyle/face-attendance.git
cd face-attendance
```

### Step 2: Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:

- torch and torchvision
- opencv-python
- numpy
- face-recognition
- pandas
- ultralytics (YOLOv8)
- Other supporting libraries

### Step 4: Set Up the Face Database

1. Create a folder structure for people you want to recognize:

   ```
   known_faces/
   ├── Person_Name_1/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   ├── Person_Name_2/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   ```

2. For each person:
   - Create a folder with their name inside `known_faces/`
   - Add 1-3 clear, well-lit photos of their face
   - Photos should show the face clearly, ideally from different angles
   - Supported formats: JPG, JPEG, PNG

### Step 5: Run the Application

```bash
python face_attendance.py
```

The first run will:

1. Process and encode all face images (might take time)
2. Cache the encodings for faster startup next time
3. Start the webcam for real-time recognition

## Key Features

### Face Recognition & Attendance

- Real-time face detection and recognition
- Automatic attendance marking with timestamps
- CSV export of attendance records by date
- Visual feedback with bounding boxes and confidence scores

### Face Encoding Caching

The system uses an efficient caching mechanism for face encodings:

- Face encodings are saved to disk after the first processing
- Subsequent runs use the cached encodings, avoiding re-processing the same images
- Cache is automatically invalidated when new images are added or modified
- This significantly improves startup time after the initial run

## System Operation

### Understanding the Display

The system shows different colored bounding boxes:

- **Yellow**: Initial face detection
- **Yellow-Green**: Face recognized, confirming identity
- **Green**: Identity confirmed, attendance marked
- **Orange**: Unknown face

### Daily Attendance Records

Attendance is stored in CSV files:

- Files are created in the `attendance/` folder
- One file per day (format: `attendance_YYYY-MM-DD.csv`)
- Each file contains names, times, and dates of attendance

## System Requirements

- Python 3.6 or higher
- Webcam or USB camera
- CUDA-compatible GPU recommended for better performance (but not required)
- Required Python libraries (installed via requirements.txt)

## Troubleshooting

1. If the camera doesn't work:

   - Check if another application is using the camera
   - Try restarting the application
   - Ensure you have the correct camera drivers installed

2. If face recognition isn't working properly:

   - Ensure good lighting conditions
   - Make sure faces are clearly visible in training photos
   - Add more photos of each person from different angles

3. If the cache isn't working:

   - Ensure the `cache` directory exists and is writable
   - Delete the cache directory to force rebuilding if you encounter issues
   - Check console output for any cache-related error messages

4. If YOLOv8 model fails to load:
   - The system will fall back to the standard YOLOv8 model
   - For better face detection, download the `yolov8n-face.pt` model

## Usage Tips

- Press 'q' to exit the application
- Recognition typically requires 2+ confirmations of the same face
- Keep the face centered and well-lit for best results
- Attendance is only marked once per person per day
