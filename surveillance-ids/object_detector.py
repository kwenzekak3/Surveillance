# Object Detection Module - Compatible with OpenCV 5.x
# Uses contour and edge detection instead of Haar Cascades

import cv2
import numpy as np

class ObjectDetector:
    """
    Detects people and objects using contour analysis
    Compatible with OpenCV 5.x (no CascadeClassifier)
    """
    
    def __init__(self):
        # Detection parameters
        self.min_area = 1500
        self.max_area = 80000
        self.detection_count = 0
        self.correct_detections = 0
        
        print("✅ Object Detector initialized (Contour-based)")
    
    def detect_objects(self, frame):
        """
        Detect objects using contour analysis
        Returns: (detected, confidence, count)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (person-sized objects)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (people are taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                
                if 1.2 < aspect_ratio < 4.0:  # Person-like shape
                    detections.append((x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Object Detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(detections) > 0:
            confidence = min(0.85, 0.55 + (len(detections) * 0.08))
            return True, confidence, len(detections)
        
        return False, 0.0, 0
    
    def detect_motion_areas(self, frame):
        """
        Detect areas of interest using edge detection
        Returns: (detected, confidence)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        significant_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                significant_contours += 1
        
        if significant_contours > 3:
            confidence = min(0.80, 0.40 + (significant_contours * 0.02))
            return True, confidence
        
        return False, 0.0
    
    def detect(self, frame):
        """
        Run all object detectors on the frame
        Returns: (total_confidence, detections)
        """
        detections = []
        total_confidence = 0.0
        
        # Detect objects
        objects_detected, obj_conf, obj_count = self.detect_objects(frame)
        if objects_detected:
            detections.append(('objects', obj_count))
            total_confidence = max(total_confidence, obj_conf)
        
        # Detect motion areas
        motion_detected, motion_conf = self.detect_motion_areas(frame)
        if motion_detected:
            detections.append(('motion_areas', 1))
            total_confidence = max(total_confidence, motion_conf * 0.8)
        
        if detections:
            self.detection_count += 1
        
        return total_confidence, detections
    
    def get_accuracy(self):
        """Get detection accuracy"""
        if self.detection_count == 0:
            return 0.0
        return (self.correct_detections / self.detection_count) * 100