# Simplified face recognizer - works without face_recognition library
# Uses OpenCV's built-in face detection

import cv2
import numpy as np
import os
import pickle

class FaceRecognizer:
    """
    Simplified face recognition using OpenCV
    """
    
    def __init__(self, database_path='face_database/known_faces.pkl'):
        self.database_path = database_path
        self.known_faces = []
        self.known_names = []
        
        # Use OpenCV's built-in face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.load_database()
        
    def load_database(self):
        """Load known faces from database"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                print(f"✅ Loaded {len(self.known_names)} known faces")
            except Exception as e:
                print(f"⚠️ Could not load face database: {e}")
                self.known_faces = []
                self.known_names = []
        else:
            print("ℹ️ No face database found. Creating new...")
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            self.save_database()
    
    def save_database(self):
        """Save face database"""
        with open(self.database_path, 'wb') as f:
            pickle.dump({
                'faces': self.known_faces,
                'names': self.known_names
            }, f)
        print("💾 Face database saved")
    
    def add_face(self, image_path, name):
        """Add new face to database"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return False
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print(f"❌ No face found in {image_path}")
            return False
        
        # Store the face image for comparison
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        
        self.known_faces.append(face_roi)
        self.known_names.append(name)
        self.save_database()
        print(f"✅ Added {name} to database")
        return True
    
    def recognize(self, frame):
        """Detect and recognize faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            name = "Unknown"
            confidence = 0.35  # Default unknown confidence
            
            # Compare with known faces
            if self.known_faces:
                best_match = None
                best_score = float('inf')
                
                for idx, known_face in enumerate(self.known_faces):
                    # Simple template matching
                    score = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)[0][0]
                    
                    if score > 0.5 and score > best_score:
                        best_score = score
                        best_match = idx
                
                if best_match is not None:
                    name = self.known_names[best_match]
                    confidence = min(0.95, best_score + 0.3)
            
            # Unknown detection with confidence
            if name == "Unknown":
                confidence = 0.35  # Low confidence for unknown
            
            results.append({
                'location': (x, y, w, h),
                'name': name,
                'confidence': confidence,
                'is_known': name != "Unknown"
            })
            
            # Draw on frame
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.0%})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        return results
    
    def is_intrusion(self, results):
        """Check if any unknown face detected"""
        for result in results:
            if not result['is_known']:
                return True, result['confidence']
        return False, 0.0