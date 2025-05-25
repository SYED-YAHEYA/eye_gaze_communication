import threading
import time
import logging
from eye_tracking import EyeTracker
from heart_rate import HeartRateMonitor
from sms_alerts import SMSAlert
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load configuration
    config = Config()

    # Initialize modules
    eye_tracker = EyeTracker(config)
    heart_rate_monitor = HeartRateMonitor(config)
    sms_alert = SMSAlert(config)

    # Start threads for concurrent execution
    eye_thread = threading.Thread(target=eye_tracker.run, daemon=True)
    heart_thread = threading.Thread(target=heart_rate_monitor.run, daemon=True)

    try:
        logging.info("Starting eye-tracking and heart rate monitoring...")
        eye_thread.start()
        heart_thread.start()

        # Main loop to check heart rate alerts
        while True:
            if heart_rate_monitor.is_abnormal():
                alert_message = f"Abnormal heart rate detected: {heart_rate_monitor.get_heart_rate()} BPM"
                sms_alert.send_alert(alert_message)
                logging.info("Sent SMS alert for abnormal heart rate")
            time.sleep(5)  # Check every 5 seconds
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        eye_tracker.stop()
        heart_rate_monitor.stop()
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        eye_tracker.stop()
        heart_rate_monitor.stop()

if __name__ == "__main__":
    main()