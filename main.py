"""
YOLO Adaptive Model Switcher
Research Implementation with Documented Constants

This implementation provides an adaptive switching mechanism between 
YOLOv8n (lightweight) and YOLOv8l (heavyweight) models based on real-time
image quality assessment.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque


class QualityMetrics:

    WEIGHT_BLUR = 0.4
    WEIGHT_BRIGHTNESS = 0.3
    WEIGHT_CONTRAST = 0.3
    

    BLUR_THRESHOLD = 500.0

    BRIGHTNESS_MIN = 100
    BRIGHTNESS_MAX = 140

    # Contrast threshold (standard deviation)
    CONTRAST_THRESHOLD = 80.0   
    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance method.
        
        The Laplacian operator detects edges by computing second derivatives.
        Sharp images have high Laplacian variance; blurry images have low variance.
        
        Returns: Score in [0, 1] where 1 = sharp, 0 = blurry
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to [0, 1] range
        normalized = np.clip(laplacian_var / self.BLUR_THRESHOLD, 0, 1)
        return normalized
    
    def calculate_brightness_score(self, image: np.ndarray) -> float:
        """
        Calculate brightness score using histogram analysis.
        
        Optimal brightness improves detection by ensuring objects are visible
        without overexposure or underexposure.
        
        Returns: Score in [0, 1] where 1 = optimal brightness, 0 = poor
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate mean brightness
        total_pixels = image.shape[0] * image.shape[1]
        mean_brightness = np.sum(hist.ravel() * np.arange(256)) / total_pixels
        
        # Score based on distance from optimal range
        if self.BRIGHTNESS_MIN <= mean_brightness <= self.BRIGHTNESS_MAX:
            return 1.0
        elif mean_brightness < self.BRIGHTNESS_MIN:
            # Too dark: linear penalty
            return mean_brightness / self.BRIGHTNESS_MIN
        else:
            # Too bright: linear penalty
            return 1.0 - ((mean_brightness - self.BRIGHTNESS_MAX) / 
                         (255 - self.BRIGHTNESS_MAX))
    
    def calculate_contrast_score(self, image: np.ndarray) -> float:
        """
        Calculate contrast score using standard deviation.
        
        Higher contrast improves edge detection and feature extraction
        for object detection models.    
        
        Returns: Score in [0, 1] where 1 = high contrast, 0 = low
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Normalize to [0, 1] range
        normalized = np.clip(contrast / self.CONTRAST_THRESHOLD, 0, 1)
        return normalized
    
    def calculate_overall_quality(self, image: np.ndarray) -> float:
        """
        Calculate weighted overall quality score.
        
        Formula: Q = 0.4*Q_blur + 0.3*Q_brightness + 0.3*Q_contrast
        
        Returns: Overall quality score in [0, 1]
        """
        blur = self.calculate_blur_score(image)
        brightness = self.calculate_brightness_score(image)
        contrast = self.calculate_contrast_score(image)
        
        overall = (self.WEIGHT_BLUR * blur + 
                  self.WEIGHT_BRIGHTNESS * brightness + 
                  self.WEIGHT_CONTRAST * contrast)
        
        return overall


class AdaptiveYOLOSwitcher:
    SWITCH_COOLDOWN = 10

    
    def __init__(self, model_n_path: str, model_l_path: str, 
                 low_threshold: float = 0.3, high_threshold: float = 0.6):
        self.model_n = YOLO(model_n_path)
        self.model_l = YOLO(model_l_path)
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.current_model = "n"  
        
        self.quality_metrics = QualityMetrics()
        
        # Performance tracking
        self.frame_count = 0
        self.switch_count = 0
        self.last_switch_frame = 0
        self.processing_times = deque(maxlen=30)
    
    def predict(self, image: np.ndarray):
        """
        Run object detection with adaptive model selection.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            tuple: (results, quality_score, model_used, processing_time)
        """
        start_time = time.time()
        
        # Calculate image quality
        quality_score = self.quality_metrics.calculate_overall_quality(image)
        
        # Apply hysteresis switching with cooldown
        frames_since_switch = self.frame_count - self.last_switch_frame
        
        if frames_since_switch >= self.SWITCH_COOLDOWN:
            previous_model = self.current_model
            
            # Switching logic
            if self.current_model == "n" and quality_score < self.low_threshold:
                # Poor quality: switch to heavy model for better accuracy
                self.current_model = "l"
                self.switch_count += 1
                self.last_switch_frame = self.frame_count
                print(f"Frame {self.frame_count}: Switch n→l (Q={quality_score:.3f})")
                
            elif self.current_model == "l" and quality_score > self.high_threshold:
                # Good quality: switch to light model for speed
                self.current_model = "n"
                self.switch_count += 1
                self.last_switch_frame = self.frame_count
                print(f"Frame {self.frame_count}: Switch l→n (Q={quality_score:.3f})")
        
        # Run inference with selected model
        model = self.model_l if self.current_model == "l" else self.model_n
        results = model(image, verbose=False)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        return results, quality_score, self.current_model, processing_time
    
    def get_statistics(self):
        """Get performance statistics."""
        avg_fps = 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'total_frames': self.frame_count,
            'total_switches': self.switch_count,
            'average_fps': avg_fps,
            'current_model': self.current_model,
            'switch_rate': self.switch_count / max(self.frame_count, 1)
        }


def process_video(video_path: str, model_n_path: str, model_l_path: str,
                 low_threshold: float = 0.3, high_threshold: float = 0.6,
                 display: bool = True):

    print("=" * 80)
    print("ADAPTIVE YOLO MODEL SWITCHER")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Thresholds: Low={low_threshold:.3f}, High={high_threshold:.3f}")
    print(f"Hysteresis Gap: {high_threshold - low_threshold:.3f}")
    print("=" * 80)
    
    # Initialize switcher
    switcher = AdaptiveYOLOSwitcher(model_n_path, model_l_path, 
                                    low_threshold, high_threshold)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...\n")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results, quality, model_used, proc_time = switcher.predict(frame)
        
        # Display (optional)
        if display:
            annotated = results[0].plot()
            
            # Add info overlay
            cv2.putText(annotated, f"Frame: {switcher.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated, f"Model: YOLOv8{model_used}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f"Quality: {quality:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Switches: {switcher.switch_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("Adaptive YOLO", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress update
        if switcher.frame_count % 100 == 0:
            stats = switcher.get_statistics()
            print(f"Progress: {switcher.frame_count}/{total_frames} frames "
                  f"({100*switcher.frame_count/total_frames:.1f}%) | "
                  f"FPS: {stats['average_fps']:.1f} | "
                  f"Switches: {switcher.switch_count}")
    
    # Cleanup
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # Final statistics
    stats = switcher.get_statistics()
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total Frames:     {stats['total_frames']}")
    print(f"Total Switches:   {stats['total_switches']}")
    print(f"Switch Rate:      {stats['switch_rate']:.4f} switches/frame")
    print(f"Average FPS:      {stats['average_fps']:.2f}")
    print(f"Final Model:      YOLOv8{stats['current_model']}")
    print("=" * 80)
    
    return switcher


if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "D:/Downloads/manythings.mp4"
    MODEL_N_PATH = "D:/Projects/Capstone Project/models/8n_final.pt"
    MODEL_L_PATH = "D:/Projects/Capstone Project/models/8m_final.pt"

    LOW_THRESHOLD = 0.486   # Switch to YOLOv8l below this quality
    HIGH_THRESHOLD = 0.536  # Switch to YOLOv8n above this quality
    
    # Process video
    switcher = process_video(
        VIDEO_PATH, 
        MODEL_N_PATH, 
        MODEL_L_PATH,
        LOW_THRESHOLD, 
        HIGH_THRESHOLD,
        display=True
    )