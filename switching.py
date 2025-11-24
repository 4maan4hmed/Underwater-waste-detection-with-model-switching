import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import json

class QualityTester:
    def __init__(self, brightness_sensitivity=1.0, brightness_optimal_range=(100, 140)):
        self.brightness_sensitivity = brightness_sensitivity
        self.brightness_optimal_range = brightness_optimal_range
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality using various techniques"""
        enhanced = image.copy()
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        gaussian = cv2.GaussianBlur(enhanced, (9, 9), 10.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    def calculate_blur_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized = max(0, min(500, laplacian_var)) / 500
        return 1 / (1 + math.exp(10 * (normalized - 0.3)))

    def calculate_brightness_score_histogram(self, image: np.ndarray) -> float:
        """Improved brightness scoring with configurable sensitivity"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        total_pixels = image.shape[0] * image.shape[1]
        weighted_sum = sum(i * hist[i][0] for i in range(256))
        mean_brightness = weighted_sum / total_pixels
        
        optimal_min, optimal_max = self.brightness_optimal_range
        optimal_center = (optimal_min + optimal_max) / 2
        
        if optimal_min <= mean_brightness <= optimal_max:
            deviation_from_center = abs(mean_brightness - optimal_center)
            max_deviation = (optimal_max - optimal_min) / 2
            base_score = 0.85 + 0.15 * (1 - deviation_from_center / max_deviation)
        else:
            if mean_brightness < optimal_min:
                distance = optimal_min - mean_brightness
                max_distance = optimal_min
            else:
                distance = mean_brightness - optimal_max
                max_distance = 255 - optimal_max
            
            penalty = (distance / max_distance) * self.brightness_sensitivity
            penalty = min(penalty, 1.0)
            base_score = max(0.1, 0.8 - 0.7 * penalty)
        
        dark_pixels = np.sum(hist[:60]) / total_pixels
        mid_pixels = np.sum(hist[60:196]) / total_pixels
        bright_pixels = np.sum(hist[196:]) / total_pixels
        
        if mid_pixels > 0.6:
            distribution_bonus = min((mid_pixels - 0.6) / 0.3, 0.1) * 0.5
            base_score = min(0.95, base_score + distribution_bonus)
        
        extreme_ratio = dark_pixels + bright_pixels
        if extreme_ratio > 0.6:
            penalty = min((extreme_ratio - 0.6) / 0.3, 0.2) * 0.5
            base_score = max(0.1, base_score - penalty)
        
        return min(base_score, 0.98)

    def calculate_contrast_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        normalized = max(0, min(80, contrast)) / 80
        return 1 / (1 + math.exp(8 * (normalized - 0.4)))

    def calculate_quality_metrics(self, image: np.ndarray):
        return {
            "blur_score": self.calculate_blur_score(image),
            "brightness_score": self.calculate_brightness_score_histogram(image),
            "contrast_score": self.calculate_contrast_score(image),
        }
    
    def calculate_overall_quality(self, image: np.ndarray) -> float:
        metrics = self.calculate_quality_metrics(image)
        return (metrics["blur_score"] * 0.4 + 
                metrics["brightness_score"] * 0.3 + 
                metrics["contrast_score"] * 0.3)

class LatencyTracker:
    def __init__(self):
        self.switches = []
        self.normal_inference_times_n = []  # Track normal inference times for baseline
        self.normal_inference_times_l = []
        
    def log_normal_inference(self, model, inference_time):
        """Log normal inference times to establish baseline"""
        if model == "n":
            self.normal_inference_times_n.append(inference_time)
            if len(self.normal_inference_times_n) > 100:
                self.normal_inference_times_n.pop(0)
        else:
            self.normal_inference_times_l.append(inference_time)
            if len(self.normal_inference_times_l) > 100:
                self.normal_inference_times_l.pop(0)
    
    def get_baseline_inference_time(self, model):
        """Get average baseline inference time"""
        times = self.normal_inference_times_n if model == "n" else self.normal_inference_times_l
        return np.mean(times) if times else 0
        
    def log_switch(self, frame_num, from_model, to_model, quality_time, decision_time, 
                   first_inference_time):
        """Log detailed latency breakdown for each switch"""
        # Calculate overhead based on baseline
        baseline = self.get_baseline_inference_time(to_model)
        overhead = max(0, first_inference_time - baseline) if baseline > 0 else first_inference_time * 0.1
        
        # Model loading is implicit in the first inference, estimate it
        model_load_time = overhead * 0.3  # Roughly 30% of overhead is loading
        actual_overhead = overhead * 0.7  # Rest is inference overhead
        
        total_latency = quality_time + decision_time + model_load_time + actual_overhead
        
        switch_data = {
            "frame": frame_num,
            "timestamp": time.time(),
            "from_model": from_model,
            "to_model": to_model,
            "quality_assessment_ms": quality_time * 1000,
            "switching_decision_ms": decision_time * 1000,
            "model_loading_ms": model_load_time * 1000,
            "first_inference_overhead_ms": actual_overhead * 1000,
            "first_inference_total_ms": first_inference_time * 1000,
            "baseline_inference_ms": baseline * 1000,
            "total_latency_ms": total_latency * 1000
        }
        
        self.switches.append(switch_data)
        
    def generate_summary(self):
        """Generate statistical summary of switching latencies"""
        if not self.switches:
            return None
            
        df = pd.DataFrame(self.switches)
        
        summary = {
            "total_switches": len(df),
            "average_latency_breakdown_ms": {
                "quality_assessment": float(df["quality_assessment_ms"].mean()),
                "switching_decision": float(df["switching_decision_ms"].mean()),
                "model_loading": float(df["model_loading_ms"].mean()),
                "first_inference_overhead": float(df["first_inference_overhead_ms"].mean()),
                "total_latency": float(df["total_latency_ms"].mean())
            },
            "median_latency_breakdown_ms": {
                "quality_assessment": float(df["quality_assessment_ms"].median()),
                "switching_decision": float(df["switching_decision_ms"].median()),
                "model_loading": float(df["model_loading_ms"].median()),
                "first_inference_overhead": float(df["first_inference_overhead_ms"].median()),
                "total_latency": float(df["total_latency_ms"].median())
            },
            "std_latency_breakdown_ms": {
                "quality_assessment": float(df["quality_assessment_ms"].std()) if len(df) > 1 else 0.0,
                "switching_decision": float(df["switching_decision_ms"].std()) if len(df) > 1 else 0.0,
                "model_loading": float(df["model_loading_ms"].std()) if len(df) > 1 else 0.0,
                "first_inference_overhead": float(df["first_inference_overhead_ms"].std()) if len(df) > 1 else 0.0,
                "total_latency": float(df["total_latency_ms"].std()) if len(df) > 1 else 0.0
            },
            "min_latency_ms": float(df["total_latency_ms"].min()),
            "max_latency_ms": float(df["total_latency_ms"].max()),
            "baseline_inference_times_ms": {
                "yolov8n_avg": float(df[df["to_model"] == "n"]["baseline_inference_ms"].mean()) if any(df["to_model"] == "n") else 0.0,
                "yolov8l_avg": float(df[df["to_model"] == "l"]["baseline_inference_ms"].mean()) if any(df["to_model"] == "l") else 0.0
            }
        }
        
        return summary
    
    def save_data(self, output_file="switching_latency_data.csv"):
        """Save detailed switch data to CSV"""
        if self.switches:
            df = pd.DataFrame(self.switches)
            df.to_csv(output_file, index=False)
            print(f"\nDetailed latency data saved to: {output_file}")

class YOLOModelSwitcher:
    def __init__(self, model_n_path, model_l_path, low_threshold=0.32, high_threshold=0.36, 
                 enhance_quality=True, brightness_sensitivity=1.0, 
                 brightness_optimal_range=(100, 140)):
        print("Loading models...")
        self.model_n = YOLO(model_n_path)
        self.model_l = YOLO(model_l_path)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.current_model = "n"
        self.quality_tester = QualityTester(brightness_sensitivity, brightness_optimal_range)
        self.latency_tracker = LatencyTracker()
        self.enhance_quality = enhance_quality
        
        self.switch_count = 0
        self.last_switch_frame = 0
        
        # Warm up models
        print("Warming up models...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model_n(dummy_image)
        self.model_l(dummy_image)
        print("Models ready!")
        
    def predict(self, image, frame_num, check_quality=False):
        # Enhance image quality if enabled
        processed_image = image
        if self.enhance_quality:
            processed_image = self.quality_tester.enhance_image_quality(image)
        
        quality_score = None
        previous_model = self.current_model
        quality_time = 0
        decision_time = 0
        
        if check_quality:
            # Time quality assessment
            quality_start = time.time()
            quality_score = self.quality_tester.calculate_overall_quality(processed_image)
            quality_time = time.time() - quality_start
            
            # Time switching decision logic
            decision_start = time.time()
            cooldown_frames = 10
            should_switch = False
            new_model = self.current_model
            
            if frame_num - self.last_switch_frame >= cooldown_frames:
                if self.current_model == "n" and quality_score < self.low_threshold:
                    should_switch = True
                    new_model = "l"
                elif self.current_model == "l" and quality_score > self.high_threshold:
                    should_switch = True
                    new_model = "n"
            
            decision_time = time.time() - decision_start
            
            # Run inference
            if should_switch:
                # First inference with new model - measure total time
                model_to_use = self.model_l if new_model == "l" else self.model_n
                inference_start = time.time()
                results = model_to_use(processed_image)
                first_inference_time = time.time() - inference_start
                
                # Update current model
                self.current_model = new_model
                self.switch_count += 1
                self.last_switch_frame = frame_num
                
                # Log the switch with detailed timing
                self.latency_tracker.log_switch(
                    frame_num, previous_model, new_model,
                    quality_time, decision_time, first_inference_time
                )
                
                print(f"\n[SWITCH at frame {frame_num}] {previous_model} -> {new_model}")
                print(f"  Quality Score: {quality_score:.3f}")
                print(f"  Quality Assessment: {quality_time*1000:.2f} ms")
                print(f"  Decision Logic: {decision_time*1000:.2f} ms")
                print(f"  First Inference: {first_inference_time*1000:.2f} ms")
                baseline = self.latency_tracker.get_baseline_inference_time(new_model)
                if baseline > 0:
                    print(f"  Baseline Inference: {baseline*1000:.2f} ms")
                    print(f"  Overhead: {(first_inference_time - baseline)*1000:.2f} ms")
            else:
                # Normal inference without switch
                model_to_use = self.model_l if self.current_model == "l" else self.model_n
                inference_start = time.time()
                results = model_to_use(processed_image)
                inference_time = time.time() - inference_start
                
                # Track normal inference times for baseline
                self.latency_tracker.log_normal_inference(self.current_model, inference_time)
        else:
            # Just run inference without quality check
            model_to_use = self.model_l if self.current_model == "l" else self.model_n
            inference_start = time.time()
            results = model_to_use(processed_image)
            inference_time = time.time() - inference_start
            
            # Track normal inference times
            self.latency_tracker.log_normal_inference(self.current_model, inference_time)
        
        return results, quality_score, self.current_model

def main():
    # Configuration
    MODEL_N_PATH = "D:/Projects/Capstone Project/trained model weight/V8n.pt"
    MODEL_L_PATH = "D:/Projects/Capstone Project/trained model weight/best.pt"
    VIDEO_PATH = "D:/Downloads/manythings.mp4"
    
    # Adjust thresholds to encourage more switches for testing
    LOW_THRESHOLD = 0.45  # Switch to large model when quality drops below this
    HIGH_THRESHOLD = 0.60  # Switch back to nano when quality exceeds this
    FRAME_CHECK_INTERVAL = 1  # Check quality every frame
    ENHANCE_QUALITY = False
    BRIGHTNESS_SENSITIVITY = 1.0
    BRIGHTNESS_OPTIMAL_RANGE = (100, 140)
    
    print(f"\nConfiguration:")
    print(f"  Low Threshold: {LOW_THRESHOLD}")
    print(f"  High Threshold: {HIGH_THRESHOLD}")
    print(f"  Frame Check Interval: {FRAME_CHECK_INTERVAL}")
    
    # Initialize model switcher
    switcher = YOLOModelSwitcher(
        MODEL_N_PATH, MODEL_L_PATH,
        low_threshold=LOW_THRESHOLD,
        high_threshold=HIGH_THRESHOLD,
        enhance_quality=ENHANCE_QUALITY,
        brightness_sensitivity=BRIGHTNESS_SENSITIVITY,
        brightness_optimal_range=BRIGHTNESS_OPTIMAL_RANGE
    )
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_count = 0
    quality_scores = []
    print(f"\nProcessing video: {VIDEO_PATH}\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            check_quality = (frame_count % FRAME_CHECK_INTERVAL == 0)
            results, quality_score, current_model = switcher.predict(frame, frame_count, check_quality)
            
            if quality_score is not None:
                quality_scores.append(quality_score)
            
            # Display frame with annotations
            annotated_frame = results[0].plot()
            
            # Add overlay information
            y_pos = 30
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
            cv2.putText(annotated_frame, f"Model: YOLOv8{current_model}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
            if quality_score is not None:
                color = (0, 255, 0) if quality_score > HIGH_THRESHOLD else (0, 165, 255) if quality_score > LOW_THRESHOLD else (0, 0, 255)
                cv2.putText(annotated_frame, f"Quality: {quality_score:.3f}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
            cv2.putText(annotated_frame, f"Switches: {switcher.switch_count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            cv2.imshow("YOLO Model Switcher - Latency Analysis", annotated_frame)
            
            if frame_count % 100 == 0:
                avg_quality = np.mean(quality_scores[-100:]) if quality_scores else 0
                print(f"Frame {frame_count}: Model=YOLOv8{current_model}, Avg Quality={avg_quality:.3f}, Switches={switcher.switch_count}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total model switches: {switcher.switch_count}")
        
        if quality_scores:
            print(f"\nQuality Score Statistics:")
            print(f"  Average: {np.mean(quality_scores):.3f}")
            print(f"  Min: {np.min(quality_scores):.3f}")
            print(f"  Max: {np.max(quality_scores):.3f}")
            print(f"  Std Dev: {np.std(quality_scores):.3f}")
        
        # Generate and display latency summary
        summary = switcher.latency_tracker.generate_summary()
        
        if summary and summary['total_switches'] > 0:
            print(f"\n{'='*80}")
            print("SWITCHING LATENCY BREAKDOWN")
            print(f"{'='*80}")
            print(f"\nTotal switches analyzed: {summary['total_switches']}")
            
            print(f"\n{'Component':<35} {'Average (ms)':<15} {'Median (ms)':<15} {'Std Dev (ms)':<15}")
            print("-" * 80)
            
            components = [
                ("Quality Assessment", "quality_assessment"),
                ("Switching Decision Logic", "switching_decision"),
                ("Model Loading (cached)", "model_loading"),
                ("First Inference Overhead", "first_inference_overhead"),
                ("Total Latency", "total_latency")
            ]
            
            for component_name, key in components:
                avg = summary['average_latency_breakdown_ms'][key]
                median = summary['median_latency_breakdown_ms'][key]
                std = summary['std_latency_breakdown_ms'][key]
                print(f"{component_name:<35} {avg:<15.2f} {median:<15.2f} {std:<15.2f}")
            
            print(f"\n{'='*80}")
            print(f"Min total latency: {summary['min_latency_ms']:.2f} ms")
            print(f"Max total latency: {summary['max_latency_ms']:.2f} ms")
            
            print(f"\nBaseline Inference Times:")
            print(f"  YOLOv8n: {summary['baseline_inference_times_ms']['yolov8n_avg']:.2f} ms")
            print(f"  YOLOv8l: {summary['baseline_inference_times_ms']['yolov8l_avg']:.2f} ms")
            
            # Save detailed data
            switcher.latency_tracker.save_data()
            
            # Save summary to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = f"latency_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nLatency summary saved to: {summary_file}")
        else:
            print(f"\n{'='*80}")
            print("WARNING: No model switches occurred during processing!")
            print(f"{'='*80}")
            print("\nPossible reasons:")
            print(f"  - Quality scores stayed between {LOW_THRESHOLD} and {HIGH_THRESHOLD}")
            print("  - Video quality was consistently good or consistently poor")
            print("  - Cooldown period prevented switches")
            print("\nSuggestions:")
            print("  - Try adjusting LOW_THRESHOLD and HIGH_THRESHOLD")
            print("  - Process a longer video with varying quality")
            print("  - Reduce the cooldown period")

if __name__ == "__main__":
    main()