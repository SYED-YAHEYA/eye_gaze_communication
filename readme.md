Eye Gaze Communication for Physically Challenged People
This project enables individuals with severe speech and motor impairments to communicate using eye gestures and monitors their heart rate for safety alerts. It runs on a Raspberry Pi 4, using a webcam for eye-tracking, a MAX30102 sensor for heart rate monitoring, and Twilio for SMS alerts.
Features

Eye-Tracking: Detects eye gestures (left, right, up, blink) and maps them to commands (e.g., "Water," "Emergency").
Heart Rate Monitoring: Tracks pulse and sends SMS alerts for abnormal readings.
Audio Output: Vocalizes commands in English or Arabic via a Bluetooth speaker.
Remote Access: Supports RealVNC for remote GUI access.

Hardware Requirements

Raspberry Pi 4
USB Webcam
MAX30102 Heart Rate Sensor
Bluetooth Speaker
MicroSD Card (16GB or higher, high-speed recommended)

Software Requirements

Raspberry Pi OS
Python 3.9+
Dependencies listed in requirements.txt

Installation

Clone the repository:git clone https://github.com/SYED-YAHEYA/eye_gaze_communication.git
cd eye_gaze_communication


Install dependencies:pip install -r requirements.txt


Download the Dlib model (shape_predictor_68_face_landmarks.dat) and place it in the assets/ folder.
Update config.py with your Twilio credentials and desired settings.
Connect the hardware (webcam, heart rate sensor, Bluetooth speaker).
Run the project:python main.py



Usage

Look at the webcam to perform eye gestures (left, right, up, blink).
A sequence of three gestures maps to a command (e.g., "left left left" → "Water").
Commands are spoken via the Bluetooth speaker and displayed on the screen.
Abnormal heart rates trigger SMS alerts to the configured recipient.

Project Structure

main.py: Orchestrates eye-tracking, heart rate monitoring, and SMS alerts.
eye_tracking.py: Handles eye gesture detection and command mapping.
heart_rate.py: Monitors heart rate using a MAX30102 sensor.
sms_alerts.py: Sends SMS alerts via Twilio.
config.py: Configuration settings.
requirements.txt: Python dependencies.
assets/: Stores the Dlib model.

License
MIT License (see LICENSE file).
Authors

SYED YAHEYA

Acknowledgments
Developed as part of an academic project to assist physically challenged individuals.
