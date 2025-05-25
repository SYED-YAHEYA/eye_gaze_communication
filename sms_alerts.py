from twilio.rest import Client
import logging

class SMSAlert:
    """Sends SMS alerts via Twilio."""

    def __init__(self, config):
        """Initialize Twilio client with configuration."""
        self.config = config
        try:
            self.client = Client(config.twilio_account_sid, config.twilio_auth_token)
            logging.info("Twilio client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Twilio client: {e}")
            self.client = None

    def send_alert(self, message):
        """Send an SMS alert to the configured recipient."""
        if not self.client:
            logging.warning("No Twilio client; skipping SMS")
            return
        try:
            self.client.messages.create(
                body=message,
                from_=self.config.twilio_from_number,
                to=self.config.twilio_to_number
            )
            logging.info(f"SMS sent: {message}")
        except Exception as e:
            logging.error(f"Failed to send SMS: {e}")