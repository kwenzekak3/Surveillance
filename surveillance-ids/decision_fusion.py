# Decision fusion - combines all detectors
# Updated with object detection for 75% accuracy

import time

class DecisionFusion:
    def __init__(self):
        # Weights for each detector (sum = 1.0)
        self.weights = {
            'motion': 0.35,
            'behavior': 0.30,
            'object': 0.35,
        }
        self.threshold = 0.45  # Lower threshold for more detections
        self.detection_count = 0
        self.correct_count = 0
        self.feedback_history = []
        self.consecutive_detections = 0
        self.alert_cooldown = 0
        self.last_alert_time = 0
        
    def decide(self, motion_result, face_result, behavior_result, object_result):
        """
        Combine all detection results using weighted voting
        Returns: (decision, confidence, details)
        """
        scores = []
        details = {
            'motion': 0.0,
            'face': 0.0,
            'behavior': 0.0,
            'object': 0.0
        }
        
        # Motion detection
        if motion_result:
            motion_conf = motion_result.get('confidence', 0.0)
            scores.append(motion_conf * self.weights['motion'])
            details['motion'] = motion_conf
        
        # Behavior analysis
        if behavior_result:
            behavior_conf = behavior_result.get('confidence', 0.0)
            scores.append(behavior_conf * self.weights['behavior'])
            details['behavior'] = behavior_conf
        
        # Object detection
        if object_result:
            object_conf = object_result.get('confidence', 0.0)
            scores.append(object_conf * self.weights['object'])
            details['object'] = object_conf
        
        # Calculate total confidence
        total_confidence = sum(scores)
        
        # Add confidence boost for multiple detections
        detection_count = sum(1 for s in scores if s > 0.05)
        if detection_count >= 2:
            # Boost confidence if multiple detectors agree
            total_confidence = min(1.0, total_confidence * 1.15)
            details['boost'] = 'multiple_detections'
        
        # Add confidence boost for consecutive frames
        if total_confidence >= self.threshold:
            self.consecutive_detections += 1
            if self.consecutive_detections >= 2:
                total_confidence = min(1.0, total_confidence * 1.10)
                details['boost'] = 'consecutive'
        else:
            self.consecutive_detections = 0
        
        # Decision
        current_time = time.time()
        decision = total_confidence >= self.threshold
        
        # Cooldown: Prevent too many alerts
        if decision and (current_time - self.last_alert_time) < 2.0:
            decision = False  # Wait 2 seconds between alerts
        
        # Track for accuracy
        if decision:
            self.detection_count += 1
            self.last_alert_time = current_time
        
        return decision, total_confidence, details
    
    def provide_feedback(self, is_correct):
        """
        Provide feedback to improve accuracy
        Called when user confirms or rejects an alert
        """
        self.feedback_history.append({
            'timestamp': time.time(),
            'correct': is_correct
        })
        
        if is_correct:
            self.correct_count += 1
        
        # Keep only last 100 feedback entries
        if len(self.feedback_history) > 100:
            self.feedback_history = self.feedback_history[-100:]
        
        # Adjust threshold based on feedback
        self._adjust_threshold()
    
    def _adjust_threshold(self):
        """
        Dynamically adjust threshold based on feedback
        """
        if len(self.feedback_history) < 10:
            return
        
        recent = self.feedback_history[-20:]
        correct_rate = sum(1 for f in recent if f['correct']) / len(recent)
        
        if correct_rate > 0.85:
            # Too many false positives? Increase threshold
            self.threshold = min(0.60, self.threshold + 0.01)
        elif correct_rate < 0.60:
            # Too few detections? Decrease threshold
            self.threshold = max(0.30, self.threshold - 0.01)
    
    def get_accuracy(self):
        """Get current accuracy with confidence"""
        if self.detection_count == 0:
            return 0.0, 'No detections yet'
        
        if self.correct_count == 0:
            return 0.0, 'No feedback received'
        
        accuracy = (self.correct_count / self.detection_count) * 100
        
        if accuracy >= 75:
            status = '✅ Good! Target reached!'
        elif accuracy >= 70:
            status = '📈 Approaching target'
        else:
            status = '📉 Need more detections'
        
        return accuracy, status