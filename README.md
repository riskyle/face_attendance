# Face Recognition Attendance Mobile App

This is a mobile application for face recognition attendance system that works with the Python backend.

## Setup Instructions

### Backend Setup (Python Server)

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Run the Flask server:

```bash
python app.py
```

3. Note down your computer's IP address (use `ipconfig` on Windows)

### Mobile App Setup (Flutter)

1. Install Flutter SDK from [Flutter's official website](https://flutter.dev/docs/get-started/install)

2. Install Android Studio or Xcode (depending on your target platform)

3. Clone this repository and navigate to the project directory

4. Install Flutter dependencies:

```bash
flutter pub get
```

5. Connect your mobile device or start an emulator

6. Run the app:

```bash
flutter run
```

## Usage

1. When you first launch the app, it will ask for camera permissions - grant them

2. Configure the server URL:

   - Tap the settings icon in the top-right corner
   - Enter your computer's IP address with port 5000
   - Example: `http://192.168.1.100:5000`
   - Tap Save

3. Using the app:
   - The camera preview will show at the top
   - Tap the camera button to capture and process a frame
   - The attendance list will update automatically
   - The list shows who has been marked present and at what time

## Building for Distribution

### Android

```bash
flutter build apk --release
```

The APK will be in `build/app/outputs/flutter-apk/app-release.apk`

### iOS

```bash
flutter build ios --release
```

Then open the project in Xcode to create the IPA file

## Requirements

- Flutter SDK
- Android Studio / Xcode
- Python 3.11
- All Python dependencies listed in requirements.txt
- Mobile device with camera
- Both mobile device and computer must be on the same network

## Troubleshooting

1. If the app can't connect to the server:

   - Check if both devices are on the same network
   - Verify the server URL is correct
   - Make sure the Python server is running
   - Check if any firewall is blocking the connection

2. If the camera doesn't work:

   - Make sure camera permissions are granted
   - Try closing and reopening the app
   - Check if the device's camera is working properly

3. If face recognition isn't working:
   - Ensure good lighting conditions
   - Make sure faces are clearly visible
   - Check if known faces are properly added to the `known_faces` directory
