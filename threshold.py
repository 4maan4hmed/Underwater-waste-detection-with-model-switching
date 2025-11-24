"""
Dataset Threshold Optimizer for Trash Detection
Calculates optimal switching thresholds for your specific dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
from datetime import datetime
import os
from tqdm import tqdm

class DatasetThresholdOptimizer:
    """
    Optimizes thresholds for trash detection dataset using your quality metrics.
    """
    
    def __init__(self, dataset_path, sample_size=500):
        """
        Initialize the dataset optimizer.
        
        Args:
            dataset_path: Path to your dataset (d:/Projects/Capstone Project/model switching/trash_inst_material)
            sample_size: Number of images to sample from validation set
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.quality_scores = []
        self.brightness_scores = []
        self.blur_scores = []
        self.contrast_scores = []
        
    def calculate_quality_score(self, image):
        """
        Calculate overall quality score for an image.
        Using the exact same method as your model switcher.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. BLUR SCORE (Laplacian Variance Method)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_normalized = np.clip(laplacian_var / 500.0, 0, 1)
        blur_score = blur_normalized
        
        # 2. BRIGHTNESS SCORE (Histogram Analysis)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = image.shape[0] * image.shape[1]
        mean_brightness = np.sum(hist.ravel() * np.arange(256)) / total_pixels
        
        optimal_min, optimal_max = 100, 140
        
        if optimal_min <= mean_brightness <= optimal_max:
            brightness_score = 1.0
        elif mean_brightness < optimal_min:
            brightness_score = mean_brightness / optimal_min
        else:
            brightness_score = 1.0 - ((mean_brightness - optimal_max) / (255 - optimal_max))
        
        brightness_score = np.clip(brightness_score, 0, 1)
        
        # 3. CONTRAST SCORE (Standard Deviation)
        contrast = np.std(gray)
        contrast_normalized = np.clip(contrast / 80.0, 0, 1)
        contrast_score = contrast_normalized
        
        # OVERALL QUALITY (weighted average)
        overall_quality = (
            0.4 * blur_score +
            0.3 * brightness_score +
            0.3 * contrast_score
        )
        
        return {
            'overall': overall_quality,
            'blur': blur_score,
            'brightness': brightness_score,
            'contrast': contrast_score,
            'raw_brightness': mean_brightness,
            'raw_blur': laplacian_var,
            'raw_contrast': contrast
        }
    
    def load_dataset_images(self, split='val'):
        """Load images from the dataset split"""
        images_path = os.path.join(self.dataset_path, split, 'images')
        if not os.path.exists(images_path):
            raise ValueError(f"Dataset path not found: {images_path}")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend([os.path.join(images_path, f) for f in os.listdir(images_path) if f.lower().endswith(ext[1:])])
        
        return sorted(image_files)
    
    def analyze_dataset(self):
        """
        Analyze the entire dataset to calculate quality distribution.
        """
        print("Loading dataset images...")
        image_files = self.load_dataset_images('val')
        
        # Sample images if dataset is large
        if len(image_files) > self.sample_size:
            step = len(image_files) / self.sample_size
            sampled_indices = [int(i * step) for i in range(self.sample_size)]
            image_files = [image_files[i] for i in sampled_indices]
        
        print(f"Analyzing {len(image_files)} images from dataset...")
        
        for image_file in tqdm(image_files, desc="Analyzing images"):
            try:
                image = cv2.imread(image_file)
                if image is None:
                    continue
                
                scores = self.calculate_quality_score(image)
                self.quality_scores.append(scores['overall'])
                self.brightness_scores.append(scores['brightness'])
                self.blur_scores.append(scores['blur'])
                self.contrast_scores.append(scores['contrast'])
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        print("Dataset analysis complete!")
        return self.get_statistics()
    
    def get_statistics(self):
        """Calculate statistical metrics for quality scores."""
        quality_array = np.array(self.quality_scores)
        
        stats = {
            'mean': float(np.mean(quality_array)),
            'median': float(np.median(quality_array)),
            'std': float(np.std(quality_array)),
            'min': float(np.min(quality_array)),
            'max': float(np.max(quality_array)),
            'percentiles': {
                '10': float(np.percentile(quality_array, 10)),
                '25': float(np.percentile(quality_array, 25)),
                '50': float(np.percentile(quality_array, 50)),
                '75': float(np.percentile(quality_array, 75)),
                '90': float(np.percentile(quality_array, 90))
            },
            'total_images': len(quality_array)
        }
        
        return stats
    
    def recommend_thresholds(self, strategy='balanced'):
        """
        Recommend optimal thresholds based on dataset quality distribution.
        
        Strategies:
        - 'balanced': Equal focus on speed and accuracy
        - 'accuracy': Prioritize detection accuracy
        - 'speed': Prioritize processing speed
        """
        stats = self.get_statistics()
        quality_array = np.array(self.quality_scores)
        
        if strategy == 'balanced':
            # Use heavy model for bottom 30%, light model for top 60%
            low_threshold = np.percentile(quality_array, 30)
            high_threshold = np.percentile(quality_array, 60)
        elif strategy == 'accuracy':
            # Use heavy model more frequently (bottom 50%)
            low_threshold = np.percentile(quality_array, 50)
            high_threshold = np.percentile(quality_array, 70)
        elif strategy == 'speed':
            # Use light model more frequently (top 70%)
            low_threshold = np.percentile(quality_array, 20)
            high_threshold = np.percentile(quality_array, 50)
        else:
            raise ValueError("Strategy must be 'balanced', 'accuracy', or 'speed'")
        
        # Ensure minimum gap of 0.05 for hysteresis
        gap = high_threshold - low_threshold
        if gap < 0.05:
            midpoint = (low_threshold + high_threshold) / 2
            low_threshold = midpoint - 0.025
            high_threshold = midpoint + 0.025
        
        recommendation = {
            'strategy': strategy,
            'low_threshold': round(low_threshold, 3),
            'high_threshold': round(high_threshold, 3),
            'hysteresis_gap': round(high_threshold - low_threshold, 3),
            'expected_l_model_usage': self._estimate_model_usage(low_threshold, quality_array),
            'expected_n_model_usage': self._estimate_model_usage(high_threshold, quality_array, above=True),
            'statistics': stats,
            'reasoning': self._generate_reasoning(low_threshold, high_threshold, stats, strategy)
        }
        
        return recommendation
    
    def _estimate_model_usage(self, threshold, quality_array, above=False):
        """Estimate percentage of images that will use a specific model."""
        if above:
            percentage = np.sum(quality_array >= threshold) / len(quality_array) * 100
        else:
            percentage = np.sum(quality_array <= threshold) / len(quality_array) * 100
        return round(percentage, 1)
    
    def _generate_reasoning(self, low_thresh, high_thresh, stats, strategy):
        """Generate reasoning for threshold selection."""
        reasoning = []
        
        reasoning.append(f"Dataset quality analysis:")
        reasoning.append(f"  - Images analyzed: {stats['total_images']}")
        reasoning.append(f"  - Mean quality: {stats['mean']:.3f}")
        reasoning.append(f"  - Quality range: {stats['min']:.3f} to {stats['max']:.3f}")
        reasoning.append(f"  - Standard deviation: {stats['std']:.3f}")
        
        reasoning.append(f"\nStrategy: {strategy.upper()}")
        if strategy == 'balanced':
            reasoning.append("  - Balanced approach: Optimizes both speed and accuracy")
            reasoning.append("  - YOLOv8l used for bottom 30% quality images")
            reasoning.append("  - YOLOv8n used for top 60% quality images")
        elif strategy == 'accuracy':
            reasoning.append("  - Accuracy-focused: Maximizes detection performance")
            reasoning.append("  - YOLOv8l used for bottom 50% quality images")
            reasoning.append("  - Better for challenging conditions")
        else:  # speed
            reasoning.append("  - Speed-focused: Maximizes processing throughput")
            reasoning.append("  - YOLOv8l used for bottom 20% quality images")
            reasoning.append("  - Better for real-time applications")
        
        reasoning.append(f"\nThreshold configuration:")
        reasoning.append(f"  - Low threshold: {low_thresh:.3f} (switch to YOLOv8l below this)")
        reasoning.append(f"  - High threshold: {high_thresh:.3f} (switch to YOLOv8n above this)")
        reasoning.append(f"  - Hysteresis gap: {high_thresh - low_thresh:.3f}")
        
        return "\n".join(reasoning)
    
    def plot_dataset_quality(self, output_path=None):
        """
        Visualize dataset quality distribution and recommended thresholds.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trash Detection Dataset - Quality Analysis', fontsize=16)
        
        # Get recommendations for all strategies
        rec_balanced = self.recommend_thresholds('balanced')
        rec_accuracy = self.recommend_thresholds('accuracy')
        rec_speed = self.recommend_thresholds('speed')
        
        # Overall quality distribution
        axes[0, 0].hist(self.quality_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(rec_balanced['low_threshold'], color='red', linestyle='--', 
                          label=f"Balanced Low: {rec_balanced['low_threshold']:.3f}")
        axes[0, 0].axvline(rec_balanced['high_threshold'], color='green', linestyle='--', 
                          label=f"Balanced High: {rec_balanced['high_threshold']:.3f}")
        axes[0, 0].axvline(rec_accuracy['low_threshold'], color='orange', linestyle=':', 
                          label=f"Accuracy Low: {rec_accuracy['low_threshold']:.3f}")
        axes[0, 0].axvline(rec_speed['high_threshold'], color='purple', linestyle=':', 
                          label=f"Speed High: {rec_speed['high_threshold']:.3f}")
        axes[0, 0].set_xlabel('Overall Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Dataset Quality Distribution')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component scores
        axes[0, 1].hist(self.blur_scores, bins=30, alpha=0.5, label='Blur', color='red')
        axes[0, 1].hist(self.brightness_scores, bins=30, alpha=0.5, label='Brightness', color='green')
        axes[0, 1].hist(self.contrast_scores, bins=30, alpha=0.5, label='Contrast', color='blue')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Quality Component Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative distribution with all strategies
        sorted_quality = np.sort(self.quality_scores)
        cumulative = np.arange(1, len(sorted_quality) + 1) / len(sorted_quality) * 100
        axes[1, 0].plot(sorted_quality, cumulative, 'b-', linewidth=2)
        
        # Add threshold lines for all strategies
        strategies = [
            ('Balanced', rec_balanced, 'red'),
            ('Accuracy', rec_accuracy, 'orange'),
            ('Speed', rec_speed, 'purple')
        ]
        
        for name, rec, color in strategies:
            axes[1, 0].axvline(rec['low_threshold'], color=color, linestyle='--', alpha=0.7,
                              label=f"{name}: {rec['low_threshold']:.3f}")
            axes[1, 0].axvline(rec['high_threshold'], color=color, linestyle=':', alpha=0.7,
                              label=f"{name}: {rec['high_threshold']:.3f}")
        
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Cumulative Percentage (%)')
        axes[1, 0].set_title('Cumulative Distribution - All Strategies')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model usage comparison
        strategies_names = ['Balanced', 'Accuracy', 'Speed']
        l_usage = [rec_balanced['expected_l_model_usage'], 
                  rec_accuracy['expected_l_model_usage'], 
                  rec_speed['expected_l_model_usage']]
        n_usage = [rec_balanced['expected_n_model_usage'], 
                  rec_accuracy['expected_n_model_usage'], 
                  rec_speed['expected_n_model_usage']]
        
        x = np.arange(len(strategies_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, l_usage, width, label='YOLOv8l Usage (%)', color='red', alpha=0.7)
        axes[1, 1].bar(x + width/2, n_usage, width, label='YOLOv8n Usage (%)', color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Strategy')
        axes[1, 1].set_ylabel('Percentage of Images (%)')
        axes[1, 1].set_title('Expected Model Usage by Strategy')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(strategies_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        plt.show()
        return fig
    
    def save_dataset_report(self, output_folder="dataset_threshold_analysis"):
        """Save complete dataset analysis report."""
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get recommendations for all strategies
        balanced = self.recommend_thresholds('balanced')
        accuracy = self.recommend_thresholds('accuracy')
        speed = self.recommend_thresholds('speed')
        
        report = {
            'analysis_timestamp': timestamp,
            'dataset_path': self.dataset_path,
            'images_analyzed': len(self.quality_scores),
            'balanced_recommendation': balanced,
            'accuracy_recommendation': accuracy,
            'speed_recommendation': speed,
            'quality_statistics': self.get_statistics()
        }
        
        # Save JSON report
        json_path = os.path.join(output_folder, f"dataset_threshold_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save plot
        plot_path = os.path.join(output_folder, f"dataset_quality_analysis_{timestamp}.png")
        self.plot_dataset_quality(plot_path)
        
        # Save text report
        text_path = os.path.join(output_folder, f"dataset_threshold_report_{timestamp}.txt")
        with open(text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRASH DETECTION DATASET - THRESHOLD ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {timestamp}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Images Analyzed: {len(self.quality_scores)}\n\n")
            
            stats = self.get_statistics()
            f.write("DATASET QUALITY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Quality:      {stats['mean']:.3f}\n")
            f.write(f"Median Quality:    {stats['median']:.3f}\n")
            f.write(f"Std Deviation:     {stats['std']:.3f}\n")
            f.write(f"Quality Range:     {stats['min']:.3f} - {stats['max']:.3f}\n")
            f.write(f"10th Percentile:   {stats['percentiles']['10']:.3f}\n")
            f.write(f"90th Percentile:   {stats['percentiles']['90']:.3f}\n\n")
            
            f.write("RECOMMENDED THRESHOLDS:\n")
            f.write("=" * 80 + "\n\n")
            
            strategies = [
                ('BALANCED (Recommended)', balanced),
                ('ACCURACY FOCUSED', accuracy),
                ('SPEED FOCUSED', speed)
            ]
            
            for strategy_name, recommendation in strategies:
                f.write(strategy_name + "\n")
                f.write("-" * 40 + "\n")
                f.write(f"Low Threshold:  {recommendation['low_threshold']:.3f}\n")
                f.write(f"High Threshold: {recommendation['high_threshold']:.3f}\n")
                f.write(f"Hysteresis Gap: {recommendation['hysteresis_gap']:.3f}\n")
                f.write(f"YOLOv8l Usage:  {recommendation['expected_l_model_usage']}%\n")
                f.write(f"YOLOv8n Usage:  {recommendation['expected_n_model_usage']}%\n\n")
                
                f.write("Reasoning:\n")
                f.write(recommendation['reasoning'] + "\n\n")
            
            f.write("IMPLEMENTATION CODE:\n")
            f.write("=" * 80 + "\n\n")
            f.write("Use these thresholds in your AdaptiveYOLOSwitcher:\n\n")
            f.write("```python\n")
            f.write("# For balanced performance (recommended):\n")
            f.write(f"LOW_THRESHOLD = {balanced['low_threshold']}\n")
            f.write(f"HIGH_THRESHOLD = {balanced['high_threshold']}\n")
            f.write("\n# Initialize switcher:\n")
            f.write("switcher = AdaptiveYOLOSwitcher(\n")
            f.write('    "path/to/yolov8n.pt",\n')
            f.write('    "path/to/yolov8l.pt",\n')
            f.write(f"    low_threshold={balanced['low_threshold']},\n")
            f.write(f"    high_threshold={balanced['high_threshold']}\n")
            f.write(")\n")
            f.write("```\n")
        
        print(f"\nComplete dataset analysis saved to: {output_folder}/")
        print(f"  - JSON report: {json_path}")
        print(f"  - Text report: {text_path}")
        print(f"  - Visualization: {plot_path}")
        
        return report


def optimize_dataset_thresholds():
    """
    Main function to optimize thresholds for your trash detection dataset.
    """
    print("=" * 80)
    print("TRASH DETECTION DATASET - THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Configuration - UPDATE THESE PATHS
    DATASET_PATH = "d:/Projects/Capstone Project/model switching/trash_inst_material"
    
    # Initialize optimizer
    optimizer = DatasetThresholdOptimizer(DATASET_PATH, sample_size=500)
    
    # Analyze dataset
    print("Step 1: Analyzing dataset quality...")
    stats = optimizer.analyze_dataset()
    
    print("\nDataset Quality Statistics:")
    print(f"  Images Analyzed: {stats['total_images']}")
    print(f"  Mean Quality:    {stats['mean']:.3f}")
    print(f"  Median Quality:  {stats['median']:.3f}")
    print(f"  Std Deviation:   {stats['std']:.3f}")
    print(f"  Quality Range:   {stats['min']:.3f} - {stats['max']:.3f}")
    
    # Get recommendations
    print("\nStep 2: Computing optimal thresholds...")
    balanced = optimizer.recommend_thresholds('balanced')
    accuracy = optimizer.recommend_thresholds('accuracy')
    speed = optimizer.recommend_thresholds('speed')
    
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLDS FOR YOUR DATASET")
    print("=" * 80)
    
    print("\nBALANCED (Recommended):")
    print(f"  Low Threshold:  {balanced['low_threshold']:.3f}")
    print(f"  High Threshold: {balanced['high_threshold']:.3f}")
    print(f"  YOLOv8l Usage:  {balanced['expected_l_model_usage']}%")
    print(f"  YOLOv8n Usage:  {balanced['expected_n_model_usage']}%")
    
    print("\nACCURACY FOCUSED:")
    print(f"  Low Threshold:  {accuracy['low_threshold']:.3f}")
    print(f"  High Threshold: {accuracy['high_threshold']:.3f}")
    print(f"  YOLOv8l Usage:  {accuracy['expected_l_model_usage']}%")
    print(f"  YOLOv8n Usage:  {accuracy['expected_n_model_usage']}%")
    
    print("\nSPEED FOCUSED:")
    print(f"  Low Threshold:  {speed['low_threshold']:.3f}")
    print(f"  High Threshold: {speed['high_threshold']:.3f}")
    print(f"  YOLOv8l Usage:  {speed['expected_l_model_usage']}%")
    print(f"  YOLOv8n Usage:  {speed['expected_n_model_usage']}%")
    
    # Save complete report
    print("\nStep 3: Generating comprehensive report...")
    optimizer.save_dataset_report("dataset_threshold_analysis")
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION INSTRUCTIONS")
    print("=" * 80)
    print("1. Use the BALANCED thresholds for general use:")
    print(f"   LOW_THRESHOLD = {balanced['low_threshold']}")
    print(f"   HIGH_THRESHOLD = {balanced['high_threshold']}")
    print()
    print("2. Update your model switcher initialization:")
    print("```python")
    print("switcher = AdaptiveYOLOSwitcher(")
    print('    "path/to/yolov8n.pt",')
    print('    "path/to/yolov8l.pt",')
    print(f"    low_threshold={balanced['low_threshold']},")
    print(f"    high_threshold={balanced['high_threshold']}")
    print(")")
    print("```")
    print()
    print("3. For accuracy-critical applications, use ACCURACY thresholds")
    print("4. For real-time applications, use SPEED thresholds")
    print("=" * 80)
    
    return optimizer, balanced


if __name__ == "__main__":
    # Run the optimization
    optimizer, balanced_thresholds = optimize_dataset_thresholds()
    
    # Print final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION FOR YOUR DATASET:")
    print(f"   Use LOW_THRESHOLD = {balanced_thresholds['low_threshold']}")
    print(f"   Use HIGH_THRESHOLD = {balanced_thresholds['high_threshold']}")