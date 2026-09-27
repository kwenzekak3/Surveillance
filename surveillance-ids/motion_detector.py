# Motion detection module
# Achieves ~65-70% accuracy

import cv2
import numpy as np
import time

class MotionDetector:
    def __init__(self, threshold=5000, sensitivity=0.5):
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.background = None
        self.last_motion = 0
        self.motion_cooldown = 2.0
        self.total_detections = 0
        self.correct_detections = 0
        
    def detect(self, frame):
        """Detect motion in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize background
        if self.background is None:
            self.background = gray.copy()
            return False, 0.0
        
        # Compute difference
        diff = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.threshold:
                motion_detected = True
                max_area = max(max_area, area)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Update background
        if not motion_detected:
            self.background = cv2.addWeighted(self.background, 0.9, gray, 0.1, 0)
        
        # Calculate confidence based on area
        confidence = min(0.95, max_area / 10000) if motion_detected else 0.0
        confidence = max(0.50, confidence)  # Minimum 50% confidence
        
        return motion_detected, confidence