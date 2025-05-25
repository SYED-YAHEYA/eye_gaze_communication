import time
import logging
try:
    import board
    import adafruit_max30102
except ImportError:
    adafruit_max30102 = None  # Fallback for systems without hardware

class HeartRateMonitor:
    """Monitors heart rate and detects abnormal readings."""

    def __init__(self, config):
        """Initialize heart rate monitor with configuration."""
        self.config = config
        self.heart_rate = 0
        self.running = False
        self.sensor = None

        if adafruit_max30102:
            try:
                i2c = board.I2C()
                self.sensor = adafruit_max30102.MAX30102(i2c)
                logging.info("MAX30102 sensor initialized")
            except Exception as e:
                logging.error(f"Failed to initialize MAX30102: {e}")
                self.sensor = None
        else:
            logging.warning("No MAX30102 library; running in simulation mode")
            self.sensor = None

    def read_heart_rate(self):
        """Read heart rate from sensor or simulate data."""
        if self.sensor:
            try:
                self.heart_rate = self.sensor.heart_rate
                if self.heart_rate is None or self.heart_rate == 0:
                    logging.warning("No valid heart rate reading")
                    return 0
                return self.heart_rate
            except Exception as e:
                logging.error(f"Error reading heart rate: {e}")
                return 0
        else:
            # Simulate heart rate for testing
            import random
            self.heart_rate = random.randint(60, 100)
            return self.heart_rate

    def is_abnormal(self):
        """Check if heart rate is outside normal range."""
        return (self.heart_rate < self.config.hr_min_threshold or 
                self.heart_rate > self.config.hr_max_threshold)

    def get_heart_rate(self):
        """Return current heart rate."""
        return self.heart_rate

    def run(self):
        """Continuously monitor heart rate."""
        self.running = True
        while self.running:
            self.read_heart_rate()
            logging.info(f"Heart rate: {self.heart_rate} BPM")
            time.sleep(self.config.hr_interval)
        logging.info("HeartRateMonitor stopped")

    def stop(self):
        """Stop monitoring."""
        self.running = False