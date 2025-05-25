import cv2
import dlib
import numpy as np
import pyttsx3
from translate import Translator
import os
import logging

class EyeTracker:
    """Handles eye-tracking and gesture-to-command mapping for communication."""

    def __init__(self, config):
        """Initialize eye tracker with configuration."""
        self.config = config
        self.detector = dlib.get_frontal_face_detector()
        model_path = os.path.join(os.getcwd(), config.dlib_model_path)
        self.predictor = dlib.shape_predictor(model_path)
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.lis = []
        self.words = []
        self.blink_counter = 0
        self.previous_ratio = 100
        self.kernel = np.ones((9, 9), np.uint8)
        self.left_eye = [36, 37, 38, 39, 40, 41]
        self.right_eye = [42, 43, 44, 45, 46, 47]

        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', config.en_voice_id)

        # Check webcam
        if not self.cap.isOpened():
            logging.error("Cannot open webcam")
            raise Exception("Webcam not accessible")

        cv2.namedWindow('image')
        cv2.createTrackbar('threshold', 'image', 75, 255, lambda x: None)
        logging.info("EyeTracker initialized")

    def translation(self, text, language="en"):
        """Translate text to the specified language."""
        if language == "en":
            return text
        translator = Translator(from_lang="en", to_lang=language)
        return translator.translate(text)

    def speech(self, text, language="en"):
        """Convert text to speech in the specified language."""
        try:
            if language == "Arabic":
                text = self.translation(text, "Arabic")
                self.engine.setProperty('voice', self.config.ar_voice_id)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Speech error: {e}")

    def eye_on_mask(self, mask, side, shape):
        """Create a mask for the eye region."""
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        l = points[0][0]
        t = (points[1][1] + points[2][1]) // 2
        r = points[3][0]
        b = (points[4][1] + points[5][1]) // 2
        return mask, [l, t, r, b]

    def midpoint(self, p1, p2):
        """Calculate midpoint between two points."""
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def euclidean_distance(self, leftx, lefty, rightx, righty):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)

    def get_EAR(self, eye_points, facial_landmarks):
        """Calculate Eye Aspect Ratio for blink detection."""
        left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
        right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line_length = self.euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
        ver_line_length = self.euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])
        return ver_line_length / hor_line_length if hor_line_length != 0 else 0

    def find_eyeball_position(self, end_points, cx, cy):
        """Determine eyeball position (left, right, up, or neutral)."""
        x_ratio = (end_points[0] - cx) / (cx - end_points[2]) if (cx - end_points[2]) != 0 else float('inf')
        y_ratio = (cy - end_points[1]) / (end_points[3] - cy) if (end_points[3] - cy) != 0 else float('inf')
        if x_ratio > 3:
            return 1  # Left
        elif x_ratio < 0.33:
            return 2  # Right
        elif y_ratio < 0.33:
            return 3  # Up
        return 0  # Neutral

    def process_thresh(self, thresh):
        """Process thresholded image for contour detection."""
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)
        return thresh

    def shape_to_np(self, shape, dtype="int"):
        """Convert Dlib shape to NumPy array."""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def search_word(self, lis):
        """Map gesture sequence to a command."""
        command_dict = {
            'left left left': 'Water',
            'left left right': 'Adjust my Specs',
            # ... (all 58 commands from cv_final.py, truncated for brevity)
            'Blink Blink Blink': 'I am Okay'
        }
        sequence = ' '.join(lis)
        return command_dict.get(sequence, 'Eye Movement States are not Recognized by Our System')

    def print_eye_pos(self, img, left, right):
        """Process and log eye positions."""
        if left == right and left != 0:
            text = ''
            if left == 1:
                text = 'left'
                logging.info("Looking left")
            elif left == 2:
                text = 'right'
                logging.info("Looking right")
            elif left == 3:
                text = 'up'
                logging.info("Looking up")
            self.lis.append(text)
        return self.lis

    def run(self):
        """Main loop for eye-tracking."""
        self.running = True
        last_gesture_time = time.time()

        while self.running:
            ret, img = self.cap.read()
            if not ret:
                logging.error("Failed to capture video frame")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)

            for rect in rects:
                shape = self.predictor(gray, rect)
                left_eye_ratio = self.get_EAR(self.left_eye, shape)
                right_eye_ratio = self.get_EAR(self.right_eye, shape)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

                if blinking_ratio < self.config.ear_threshold:
                    if self.previous_ratio > self.config.ear_threshold:
                        self.blink_counter += 1
                        self.lis.append('Blink')
                        logging.info("Blink detected")
                self.previous_ratio = blinking_ratio

                shape = self.shape_to_np(shape)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask, end_points_left = self.eye_on_mask(mask, self.left_eye, shape)
                mask, end_points_right = self.eye_on_mask(mask, self.right_eye, shape)
                mask = cv2.dilate(mask, self.kernel, 5)

                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = self.process_thresh(thresh)

                eyeball_pos_left = self.contouring(thresh[:, 0:mid], mid, img, end_points_left)
                eyeball_pos_right = self.contouring(thresh[:, mid:], mid, img, end_points_right, True)

                self.lis = self.print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)

                if len(self.lis) > 3:
                    self.lis = []
                if len(self.lis) == 3:
                    self.words.append(' '.join(self.lis))
                    text = self.search_word(self.words)
                    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                    logging.info(f"Command: {text}")
                    self.speech(text, 'en')
                    self.words.clear()
                    self.lis.clear()

                for (x, y) in shape[36:48]:
                    cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

            cv2.imshow('eyes', img)
            cv2.imshow('image', thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Reset gesture list if no new gesture in 5 seconds
            if len(self.lis) > 0 and time.time() - last_gesture_time > 5:
                self.lis.clear()
                logging.info("Gesture list reset due to timeout")
            if len(self.lis) > 0:
                last_gesture_time = time.time()

        self.stop()

    def contouring(self, thresh, mid, img, end_points, right=False):
        """Detect eyeball position via contouring."""
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
            return self.find_eyeball_position(end_points, cx, cy)
        except:
            return None

    def stop(self):
        """Clean up resources."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("EyeTracker stopped")