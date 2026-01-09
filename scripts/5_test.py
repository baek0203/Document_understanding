#!/usr/bin/env python3
"""
YOLO Model Testing and Comparison Script

This script performs inference using trained YOLO models and generates
comparison visualizations. Supports multiple models and batch processing.

Usage:
    python scripts/5_test.py [--num_images N] [--output_dir DIR]
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.consts import CATEGORY_ID_TO_NAME, DEFAULT_CONFIDENCE, DEFAULT_IOU
from utils.image_utils import preprocess_for_detection
from utils.visualization import (
    draw_yolo_results,
    create_side_by_side,
    draw_statistics,
    save_visualization
)


class YOLOTester:
    """YOLO model testing and comparison"""

    def __init__(
        self,
        model_paths: dict,
        conf_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU
    ):
        """
        Initialize tester with models.

        Args:
            model_paths: Dict of {model_name: model_path}
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.models = {}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        print("=" * 60)
        print("Loading YOLO Models")
        print("=" * 60)

        for name, path in model_paths.items():
            if not os.path.exists(path):
                print(f"⚠️  Model not found: {path}")
                continue

            print(f"Loading {name}: {path}")
            self.models[name] = YOLO(path)

        print(f"\n✓ Loaded {len(self.models)} models")
        print()

    def run_inference(
        self,
        image_path: str,
        model_name: str,
        preprocess: bool = True
    ) -> tuple:
        """
        Run inference on single image.

        Args:
            image_path: Path to input image
            model_name: Name of model to use
            preprocess: Apply preprocessing

        Returns:
            (results, inference_time)
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        model = self.models[model_name]

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Preprocess if requested
        if preprocess:
            from PIL import Image
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_pil = preprocess_for_detection(image_pil, target_size=1000)
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Run inference
        start_time = time.time()
        results = model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time

        return results[0], inference_time

    def compare_models(
        self,
        image_path: str,
        output_dir: str,
        save_individual: bool = True
    ) -> dict:
        """
        Compare all loaded models on single image.

        Args:
            image_path: Path to input image
            output_dir: Output directory
            save_individual: Save individual model results

        Returns:
            Dict of results per model
        """
        os.makedirs(output_dir, exist_ok=True)

        base_name = Path(image_path).stem
        image = cv2.imread(image_path)

        print(f"\nProcessing: {base_name}")
        print("-" * 60)

        results_dict = {}

        # Run inference for each model
        for model_name in self.models.keys():
            results, inf_time = self.run_inference(image_path, model_name)

            # Count detections by category
            stats = {}
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                for cls in results.boxes.cls:
                    cat_name = CATEGORY_ID_TO_NAME.get(int(cls), f"Class-{int(cls)}")
                    stats[cat_name] = stats.get(cat_name, 0) + 1

            # Draw results
            vis_image = draw_yolo_results(image, results)
            vis_image = draw_statistics(vis_image, stats)

            # Save individual result
            if save_individual:
                save_path = os.path.join(output_dir, f"{base_name}_{model_name}.jpg")
                save_visualization(vis_image, save_path)

            results_dict[model_name] = {
                'results': results,
                'image': vis_image,
                'stats': stats,
                'inference_time': inf_time,
                'num_detections': len(results.boxes) if hasattr(results, 'boxes') else 0
            }

            print(f"  {model_name:15s} | Time: {inf_time:.3f}s | "
                  f"Detections: {results_dict[model_name]['num_detections']}")

        # Create comparison if multiple models
        if len(self.models) == 2:
            model_names = list(results_dict.keys())
            img1 = results_dict[model_names[0]]['image']
            img2 = results_dict[model_names[1]]['image']

            comparison = create_side_by_side(
                img1, img2,
                title1=model_names[0],
                title2=model_names[1]
            )

            save_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
            save_visualization(comparison, save_path)
            print(f"  Saved comparison: {save_path}")

        return results_dict

    def batch_test(
        self,
        image_paths: list,
        output_dir: str
    ) -> dict:
        """
        Batch test on multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Output directory

        Returns:
            Dict of aggregate statistics
        """
        print("\n" + "=" * 60)
        print(f"Batch Testing: {len(image_paths)} images")
        print("=" * 60)

        aggregate_stats = {name: {
            'total_detections': 0,
            'total_time': 0.0,
            'num_images': 0
        } for name in self.models.keys()}

        for img_path in image_paths:
            results = self.compare_models(img_path, output_dir)

            # Update aggregate stats
            for model_name, result in results.items():
                aggregate_stats[model_name]['total_detections'] += result['num_detections']
                aggregate_stats[model_name]['total_time'] += result['inference_time']
                aggregate_stats[model_name]['num_images'] += 1

        # Print summary
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)

        for model_name, stats in aggregate_stats.items():
            avg_time = stats['total_time'] / stats['num_images'] if stats['num_images'] > 0 else 0
            avg_det = stats['total_detections'] / stats['num_images'] if stats['num_images'] > 0 else 0

            print(f"\n{model_name}:")
            print(f"  Images processed:     {stats['num_images']}")
            print(f"  Total detections:     {stats['total_detections']}")
            print(f"  Avg detections/image: {avg_det:.1f}")
            print(f"  Total inference time: {stats['total_time']:.2f}s")
            print(f"  Avg inference time:   {avg_time:.3f}s")

        return aggregate_stats


def main():
    parser = argparse.ArgumentParser(description="Test YOLO models on DocLayNet")
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of test images to process')
    parser.add_argument('--output_dir', type=str, default='out_compare',
                        help='Output directory for results')
    parser.add_argument('--test_dir', type=str,
                        default='data/doclaynet_yolo_split/images/test',
                        help='Test images directory')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU,
                        help='IoU threshold for NMS')
    parser.add_argument('--preprocess', action='store_true',
                        help='Apply preprocessing to images')

    args = parser.parse_args()

    # Model paths
    model_paths = {
        'yolo11n': 'runs/doclaynet/yolo11n/weights/best.pt',
        'yolov8n': 'runs/doclaynet/yolov8n_doclaynet4/weights/best.pt',
    }

    # Find test images
    test_dir = args.test_dir
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return

    img_list = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    if len(img_list) == 0:
        print(f"❌ No JPG files found in: {test_dir}")
        return

    # Limit number of images
    img_list = img_list[:args.num_images]

    print(f"Found {len(img_list)} test images")

    # Initialize tester
    tester = YOLOTester(
        model_paths=model_paths,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Run batch test
    start_time = time.time()
    stats = tester.batch_test(img_list, args.output_dir)
    total_time = time.time() - start_time

    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
    print(f"✓  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
