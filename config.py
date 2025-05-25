class Config:
    """Configuration settings for the eye gaze communication system."""

    # Eye-tracking settings
    ear_threshold = 0.20  # Eye Aspect Ratio threshold for blink detection
    dlib_model_path = "assets/shape_predictor_68_face_landmarks.dat"
    en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
    ar_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_arEG_Hoda"

    # Heart rate settings
    hr_min_threshold = 50  # Minimum normal heart rate (BPM)
    hr_max_threshold = 120  # Maximum normal heart rate (BPM)
    hr_interval = 2  # Seconds between heart rate readings

    # Twilio settings (replace with your credentials)
    twilio_account_sid = "YOUR_TWILIO_ACCOUNT_SID"
    twilio_auth_token = "YOUR_TWILIO_AUTH_TOKEN"
    twilio_from_number = "YOUR_TWILIO_PHONE_NUMBER"
    twilio_to_number = "YOUR_RECIPIENT_PHONE_NUMBER"