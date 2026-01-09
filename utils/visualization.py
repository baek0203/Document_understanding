#!/usr/bin/env python3
"""
Visualization Utilities

Functions for drawing detection results and creating comparison images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Union, Optional
from .consts import CATEGORY_COLORS, CATEGORY_ID_TO_NAME


def draw_detections(
    image: Union[np.ndarray, Image.Image],
    detections: List[Dict],
    show_labels: bool = True,
    show_conf: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.

    Args:
        image: Input image (numpy array or PIL Image)
        detections: List of detection dicts with keys:
            - bbox: [x1, y1, x2, y2]
            - category: category name or id
            - confidence: confidence score (optional)
        show_labels: Show category labels
        show_conf: Show confidence scores
        thickness: Box line thickness
        font_scale: Font size scale

    Returns:
        Image with drawn detections (numpy array, BGR)
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = image.copy()

    for det in detections:
        bbox = det.get('bbox', [])
        category = det.get('category', '')
        confidence = det.get('confidence', 0.0)

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Get category name
        if isinstance(category, int):
            cat_name = CATEGORY_ID_TO_NAME.get(category, f'Class-{category}')
        else:
            cat_name = str(category)

        # Get color
        color = CATEGORY_COLORS.get(cat_name, (255, 255, 255))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if show_labels:
            label = cat_name
            if show_conf and confidence > 0:
                label += f' {confidence:.2f}'

            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw background
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness=1
            )

    return image


def draw_yolo_results(
    image: Union[np.ndarray, Image.Image],
    results,
    show_labels: bool = True,
    show_conf: bool = True
) -> np.ndarray:
    """
    Draw YOLO model results on image.

    Args:
        image: Input image
        results: YOLO results object
        show_labels: Show category labels
        show_conf: Show confidence scores

    Returns:
        Image with drawn detections (numpy array, BGR)
    """
    # Extract detections from YOLO results
    detections = []

    if hasattr(results, 'boxes'):
        boxes = results.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())

            detections.append({
                'bbox': bbox.tolist(),
                'category': cls,
                'confidence': conf
            })

    return draw_detections(image, detections, show_labels, show_conf)


def create_comparison_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    grid_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Create a grid of images for comparison.

    Args:
        images: List of images (numpy arrays, BGR)
        titles: Optional titles for each image
        grid_shape: (rows, cols) or None for auto

    Returns:
        Grid image (numpy array, BGR)
    """
    if not images:
        raise ValueError("No images provided")

    n_images = len(images)

    # Auto grid shape
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_shape

    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Resize all images to same size
    resized = []
    for img in images:
        if img.shape[0] != max_h or img.shape[1] != max_w:
            img_resized = cv2.resize(img, (max_w, max_h))
        else:
            img_resized = img.copy()

        # Add title if provided
        if titles and len(resized) < len(titles):
            title = titles[len(resized)]
            cv2.putText(
                img_resized,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        resized.append(img_resized)

    # Pad with blank images if needed
    while len(resized) < rows * cols:
        blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        resized.append(blank)

    # Create grid
    grid_rows = []
    for r in range(rows):
        row_images = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_images))

    grid = np.vstack(grid_rows)
    return grid


def create_side_by_side(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Model 1",
    title2: str = "Model 2"
) -> np.ndarray:
    """
    Create side-by-side comparison of two images.

    Args:
        image1: First image (numpy array, BGR)
        image2: Second image (numpy array, BGR)
        title1: Title for first image
        title2: Title for second image

    Returns:
        Combined image (numpy array, BGR)
    """
    return create_comparison_grid([image1, image2], [title1, title2], (1, 2))


def draw_statistics(
    image: np.ndarray,
    stats: Dict[str, int],
    position: str = 'top-right'
) -> np.ndarray:
    """
    Draw detection statistics on image.

    Args:
        image: Input image (numpy array, BGR)
        stats: Dictionary of category -> count
        position: Position ('top-right', 'top-left', 'bottom-right', 'bottom-left')

    Returns:
        Image with statistics (numpy array, BGR)
    """
    image = image.copy()
    h, w = image.shape[:2]

    # Prepare text
    lines = ["Detection Statistics:"]
    for cat, count in sorted(stats.items()):
        lines.append(f"  {cat}: {count}")

    # Calculate text box size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 10

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = max(size[0] for size in text_sizes) + 2 * padding
    box_h = sum(size[1] for size in text_sizes) + len(lines) * 5 + 2 * padding

    # Calculate position
    if position == 'top-right':
        x, y = w - box_w - 10, 10
    elif position == 'top-left':
        x, y = 10, 10
    elif position == 'bottom-right':
        x, y = w - box_w - 10, h - box_h - 10
    else:  # bottom-left
        x, y = 10, h - box_h - 10

    # Draw background
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (255, 255, 255), 2)

    # Draw text
    y_offset = y + padding + text_sizes[0][1]
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x + padding, y_offset),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        y_offset += text_sizes[i][1] + 5

    return image


def save_visualization(
    image: np.ndarray,
    save_path: str,
    quality: int = 95
) -> None:
    """
    Save visualization image with high quality.

    Args:
        image: Image to save (numpy array, BGR)
        save_path: Output file path
        quality: JPEG quality (0-100)
    """
    if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(save_path, image)
