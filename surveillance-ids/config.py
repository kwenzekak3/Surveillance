# Configuration settings
# Updated for 75% accuracy target

import os

class Config:
    SECRET_KEY = 'surveillance-key-2026'
    DEBUG = True
    
    # Detection thresholds (lowered for 75% target)
    CONFIDENCE_THRESHOLD = 0.45  # Lower = more detections
    MOTION_THRESHOLD = 1500      # More sensitive
    FACE_MATCH_THRESHOLD = 0.6
    
    # Enable/disable modules
    ENABLE_MOTION = True
    ENABLE_BEHAVIOR = True
    ENABLE_OBJECT = True
    
    # Voting weights (sum = 1.0)
    WEIGHT_MOTION = 0.35
    WEIGHT_BEHAVIOR = 0.30
    WEIGHT_OBJECT = 0.35
    
    # Logging
    LOG_FILE = 'logs/security.log'
    FACE_DATABASE = 'face_database/known_faces.pkl'