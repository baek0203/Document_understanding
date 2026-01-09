#!/usr/bin/env python3
"""
Constants for Document Layout Analysis

This module defines configuration constants used throughout the project.
Based on DocLayNet dataset and YOLO object detection framework.
"""

# Image processing constants
IMAGE_MIN_SIZE = 640  # Minimum image dimension for YOLO
IMAGE_MAX_SIZE = 1280  # Maximum image dimension
IMAGE_FACTOR = 32  # Factor for dimension rounding (YOLO stride)

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf']
SUPPORTED_ALL_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_DOCUMENT_EXTENSIONS

# DocLayNet categories (11 classes, 0-indexed for YOLO)
DOCLAYNET_CATEGORIES = [
    'Caption',       # 0
    'Footnote',      # 1
    'Formula',       # 2
    'List-item',     # 3
    'Page-footer',   # 4
    'Page-header',   # 5
    'Picture',       # 6
    'Section-header',# 7
    'Table',         # 8
    'Text',          # 9
    'Title'          # 10
]

# Category ID to name mapping
CATEGORY_ID_TO_NAME = {i: name for i, name in enumerate(DOCLAYNET_CATEGORIES)}

# Category name to ID mapping
CATEGORY_NAME_TO_ID = {name: i for i, name in enumerate(DOCLAYNET_CATEGORIES)}

# Color palette for visualization (BGR format for OpenCV)
# Each category gets a distinct color
CATEGORY_COLORS = {
    'Caption': (255, 200, 100),      # Light blue
    'Footnote': (180, 180, 180),     # Gray
    'Formula': (255, 100, 255),      # Magenta
    'List-item': (100, 255, 100),    # Light green
    'Page-footer': (150, 150, 150),  # Dark gray
    'Page-header': (200, 200, 200),  # Light gray
    'Picture': (255, 150, 0),        # Orange
    'Section-header': (0, 255, 255), # Yellow
    'Table': (255, 0, 0),            # Blue
    'Text': (0, 255, 0),             # Green
    'Title': (0, 0, 255)             # Red
}

# YOLO model configurations
YOLO_MODELS = {
    'yolov8n': {
        'name': 'YOLOv8n',
        'description': 'Nano model - fastest, lowest accuracy'
    },
    'yolov8s': {
        'name': 'YOLOv8s',
        'description': 'Small model - balanced speed and accuracy'
    },
    'yolo11n': {
        'name': 'YOLO11n',
        'description': 'Latest nano model - improved performance'
    },
    'yolo11s': {
        'name': 'YOLO11s',
        'description': 'Latest small model - best overall'
    }
}

# Default inference parameters
DEFAULT_CONFIDENCE = 0.25  # Minimum confidence threshold
DEFAULT_IOU = 0.45  # IoU threshold for NMS
DEFAULT_IMG_SIZE = 640  # Default image size for inference

# Training parameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE_TRAIN = 640
DEFAULT_WORKERS = 4

# Paths (relative to project root)
DEFAULT_DATA_DIR = './data'
DEFAULT_OUTPUT_DIR = './out'
DEFAULT_RUNS_DIR = './runs'
DEFAULT_CONFIG_DIR = './config'
