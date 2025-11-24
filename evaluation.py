

import cv2
import numpy as np
import time
import os
import json
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import shutil

class QualityMetrics:
    """Calculate image quality score."""
    
    def calculate_quality(self, image: np.ndarray) -> float:
        """Calculate overall quality score (0-1)."""
        if image is None:
            return 0.0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur score (Laplacian variance)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = np.clip(blur / 500.0, 0, 1)
        
        # Brightness score
        mean_brightness = gray.mean()
        if 100 <= mean_brightness <= 140:
            brightness_score = 1.0
        elif mean_brightness < 100:
            brightness_score = mean_brightness / 100
        else:
            brightness_score = 1.0 - ((mean_brightness - 140) / 115)
        
        # Contrast score
        contrast = gray.std()
        contrast_score = np.clip(contrast / 80.0, 0, 1)
        
        # Weighted combination
        quality = 0.4 * blur_score + 0.3 * brightness_score + 0.3 * contrast_score
        return quality


class AdaptiveSwitcher:
    """Adaptive switcher with hysteresis (two thresholds)."""
    
    def __init__(self, model_n_path: str, model_l_path: str, 
                 low_threshold: float = 0.763, high_threshold: float = 0.813):
        """
        Args:
            low_threshold: Switch to large model when quality drops below this
            high_threshold: Switch back to nano model when quality rises above this
        """
        self.model_n = YOLO(model_n_path)
        self.model_l = YOLO(model_l_path)
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.current_model = "n"  # Start with nano
        
        self.quality_metrics = QualityMetrics()
        self.switch_count = 0
        self.model_usage = {"n": 0, "l": 0}
        self.quality_scores = []
    
    def predict(self, image: np.ndarray):
        """Run detection with adaptive model selection."""
        # Calculate quality
        quality = self.quality_metrics.calculate_quality(image)
        self.quality_scores.append(quality)
        
        # Hysteresis switching logic
        previous_model = self.current_model
        
        if self.current_model == "n" and quality < self.low_threshold:
            self.current_model = "l"
        elif self.current_model == "l" and quality > self.high_threshold:
            self.current_model = "n"
        
        # Count actual switches
        if previous_model != self.current_model:
            self.switch_count += 1
        
        # Track usage
        self.model_usage[self.current_model] += 1
        
        # Run inference
        start_time = time.time()
        model = self.model_l if self.current_model == "l" else self.model_n
        results = model(image, verbose=False)[0]
        proc_time = time.time() - start_time
        
        return results, quality, self.current_model, proc_time


def evaluate_adaptive_map(dataset_path, model_n_path, model_l_path, 
                          low_threshold=0.4, high_threshold=0.6):
    """
    Evaluate adaptive approach with FIXED paths
    """
    print(f"\nEvaluating Adaptive (Low={low_threshold}, High={high_threshold})...")
    
    # FIXED: Use your exact paths
    val_images_path = os.path.join(dataset_path, "val", "images")
    val_labels_path = os.path.join(dataset_path, "val", "labels")
    
    print(f"Images path: {val_images_path}")
    print(f"Labels path: {val_labels_path}")
    
    # Verify paths exist
    if not os.path.exists(val_images_path):
        print(f"ERROR: Images path not found: {val_images_path}")
        return None
    if not os.path.exists(val_labels_path):
        print(f"ERROR: Labels path not found: {val_labels_path}")
        return None
    
    temp_pred_path = "temp_adaptive_predictions"
    
    # Clean up old predictions
    if os.path.exists(temp_pred_path):
        shutil.rmtree(temp_pred_path)
    os.makedirs(temp_pred_path, exist_ok=True)
    
    # Initialize switcher
    switcher = AdaptiveSwitcher(model_n_path, model_l_path, low_threshold, high_threshold)
    
    # Get all validation images
    image_files = sorted([f for f in os.listdir(val_images_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Found {len(image_files)} validation images")
    
    if len(image_files) == 0:
        print("ERROR: No images found in validation folder!")
        return None
    
    # Run adaptive predictions
    predictions = {}
    processing_times = []
    
    print("Running adaptive detection...")
    for img_file in tqdm(image_files):
        img_path = os.path.join(val_images_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        h, w = image.shape[:2]
        result, quality, model_used, proc_time = switcher.predict(image)
        
        predictions[img_file] = result
        processing_times.append(proc_time)
    
    print(f"Successfully processed {len(predictions)} images")
    
    # Save predictions in label format
    print("Saving predictions...")
    for img_file, result in tqdm(predictions.items()):
        img_path = os.path.join(val_images_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        h, w = image.shape[:2]
        
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(temp_pred_path, label_file)
        
        with open(label_path, 'w') as f:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    
                    # Filter low confidence detections
                    if conf < 0.25:  # Confidence threshold
                        continue
                    
                    # Get xyxy coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to YOLO format (normalized xywh)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Ensure values are within [0,1] range
                    x_center = np.clip(x_center, 0.0, 1.0)
                    y_center = np.clip(y_center, 0.0, 1.0)
                    width = np.clip(width, 0.0, 1.0)
                    height = np.clip(height, 0.0, 1.0)
                    
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Calculate mAP using improved method
    print("Calculating mAP...")
    
    # Improved mAP calculation
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    
    for iou_thresh in iou_thresholds:
        ap = calculate_ap(temp_pred_path, val_labels_path, iou_thresh)
        aps.append(ap)
        print(f"IoU {iou_thresh}: AP = {ap:.4f}")
    
    map50 = aps[0]  # AP at IoU 0.5
    map50_95 = np.mean(aps)  # mAP across IoU thresholds
    
    # Calculate precision and recall at IoU 0.5
    precision, recall = calculate_precision_recall(temp_pred_path, val_labels_path, iou_threshold=0.5)
    
    # Clean up
    shutil.rmtree(temp_pred_path)
    
    # Calculate usage stats
    total_images = len(image_files)
    avg_fps = 1.0 / np.mean(processing_times) if processing_times else 0
    
    return {
        'model': 'Adaptive',
        'mAP50': map50,
        'mAP50-95': map50_95,
        'precision': precision,
        'recall': recall,
        'avg_fps': avg_fps,
        'switch_count': switcher.switch_count,
        'n_usage_percent': (switcher.model_usage['n'] / total_images) * 100,
        'l_usage_percent': (switcher.model_usage['l'] / total_images) * 100,
        'avg_quality': np.mean(switcher.quality_scores) if switcher.quality_scores else 0,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold
    }


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in YOLO format (x_center, y_center, w, h)."""
    def xywh_to_xyxy(box):
        x_c, y_c, w, h = box
        return [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
    
    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)
    
    # Calculate intersection
    x1_i = max(box1_xyxy[0], box2_xyxy[0])
    y1_i = max(box1_xyxy[1], box2_xyxy[1])
    x2_i = min(box1_xyxy[2], box2_xyxy[2])
    y2_i = min(box1_xyxy[3], box2_xyxy[3])
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(pred_path, gt_path, iou_threshold=0.5):
    """Calculate Average Precision for given IoU threshold."""
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]
    
    all_matches = []
    all_gt_count = 0
    
    for pred_file in pred_files:
        gt_file = os.path.join(gt_path, pred_file)
        pred_file_path = os.path.join(pred_path, pred_file)
        
        # Load predictions
        preds = []
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        preds.append([int(parts[0])] + [float(x) for x in parts[1:5]])
        
        # Load ground truth
        gts = []
        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gts.append([int(parts[0])] + [float(x) for x in parts[1:5]])
            all_gt_count += len(gts)
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred in preds:
            pred_cls, pred_box = pred[0], pred[1:]
            best_iou = 0
            best_idx = -1
            
            for idx, gt in enumerate(gts):
                gt_cls, gt_box = gt[0], gt[1:]
                
                if pred_cls == gt_cls and idx not in matched_gt:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
            
            if best_iou >= iou_threshold:
                all_matches.append(1)  # True Positive
                matched_gt.add(best_idx)
            else:
                all_matches.append(0)  # False Positive
    
    # Calculate precision and recall
    tp = sum(all_matches)
    fp = len(all_matches) - tp
    fn = all_gt_count - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # AP is precision for this simple case
    return precision * recall


def calculate_precision_recall(pred_path, gt_path, iou_threshold=0.5):
    """Calculate precision and recall."""
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]
    
    tp = 0
    fp = 0
    fn = 0
    
    for pred_file in pred_files:
        gt_file = os.path.join(gt_path, pred_file)
        pred_file_path = os.path.join(pred_path, pred_file)
        
        # Load predictions
        preds = []
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        preds.append([int(parts[0])] + [float(x) for x in parts[1:5]])
        
        # Load ground truth
        gts = []
        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gts.append([int(parts[0])] + [float(x) for x in parts[1:5]])
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred in preds:
            pred_cls, pred_box = pred[0], pred[1:]
            best_iou = 0
            best_idx = -1
            
            for idx, gt in enumerate(gts):
                gt_cls, gt_box = gt[0], gt[1:]
                
                if pred_cls == gt_cls and idx not in matched_gt:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        
        fn += len(gts) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall


def main():
    """Run adaptive evaluation only."""
    
    # FIXED Configuration with your exact paths
    DATASET_PATH = "D:/Projects/Capstone Project/model switching/trash_inst_material"
    MODEL_N_PATH = "D:/Projects/Capstone Project/models/8n_final.pt"
    MODEL_L_PATH = "D:/Projects/Capstone Project/models/8m_final.pt"  # or 8l_final.pt if you have it
    
    # Hysteresis thresholds
    LOW_THRESHOLD = 0.763   # Switch to large model below this
    HIGH_THRESHOLD = 0.813  # Switch back to nano above this

    print("\n" + "="*80)
    print("RUNNING ADAPTIVE EVALUATION")
    print("="*80)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Model N path: {MODEL_N_PATH}")
    print(f"Model L path: {MODEL_L_PATH}")

    adaptive_results = evaluate_adaptive_map(
        DATASET_PATH, MODEL_N_PATH, MODEL_L_PATH,
        LOW_THRESHOLD, HIGH_THRESHOLD
    )

    if adaptive_results is None:
        print("ERROR: Evaluation failed! Check the paths above.")
        return

    print("\n" + "="*80)
    print("ADAPTIVE EVALUATION RESULTS")
    print("="*80)
    
    print(f"{'Model':<15} {'mAP@0.5':<10} {'mAP@0.5:0.95':<15} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    print(f"{adaptive_results['model']:<15} {adaptive_results['mAP50']:<10.4f} "
          f"{adaptive_results['mAP50-95']:<15.4f} {adaptive_results['precision']:<10.4f} "
          f"{adaptive_results['recall']:<10.4f}")

    if 'avg_fps' in adaptive_results:
        print(f"\nPerformance:")
        print(f"  Average FPS: {adaptive_results['avg_fps']:.2f}")
        print(f"  Switch Count: {adaptive_results['switch_count']}")
        print(f"  YOLOv8n Usage: {adaptive_results['n_usage_percent']:.1f}%")
        print(f"  YOLOv8l Usage: {adaptive_results['l_usage_percent']:.1f}%")
        print(f"  Avg Quality: {adaptive_results['avg_quality']:.3f}")
        print(f"  Thresholds: Low={LOW_THRESHOLD}, High={HIGH_THRESHOLD}")
    
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/adaptive_results.json', 'w') as f:
        json.dump(adaptive_results, f, indent=2)

    print("\nResults saved to: evaluation_results/adaptive_results.json")

if __name__ == "__main__":
    main()