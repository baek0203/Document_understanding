#!/usr/bin/env python3
"""
Image Preprocessing Utilities

Functions for image enhancement and preprocessing before YOLO inference.
Inspired by Samsung AI Challenge preprocessing techniques.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Union, Tuple, Optional


def preprocess_for_detection(
    image: Union[np.ndarray, Image.Image],
    target_size: Optional[int] = None,
    enhance_contrast: float = 1.2,
    enhance_sharpness: float = 1.2,
    denoise: bool = True
) -> Image.Image:
    """
    Preprocess image for better detection results.

    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target minimum dimension (None = no resize)
        enhance_contrast: Contrast enhancement factor (1.0 = no change)
        enhance_sharpness: Sharpness enhancement factor (1.0 = no change)
        denoise: Apply median filter for noise reduction

    Returns:
        Preprocessed PIL Image
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            image = Image.fromarray(image).convert('RGB')
        else:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize if needed
    if target_size is not None:
        w, h = image.size
        if max(w, h) < target_size:
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

    # Enhance contrast
    if enhance_contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(enhance_contrast)

    # Enhance sharpness
    if enhance_sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(enhance_sharpness)

    # Denoise
    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    return image


def add_padding(
    image: Union[np.ndarray, Image.Image],
    padding: int = 10,
    color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Add padding around image.

    Args:
        image: Input image
        padding: Padding size in pixels
        color: Padding color (RGB)

    Returns:
        Padded PIL Image
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Create padded canvas
    new_w = image.width + 2 * padding
    new_h = image.height + 2 * padding
    padded = Image.new('RGB', (new_w, new_h), color)
    padded.paste(image, (padding, padding))

    return padded


def normalize_image_size(
    image: Image.Image,
    target_size: int = 640,
    stride: int = 32
) -> Image.Image:
    """
    Normalize image size for YOLO (divisible by stride).

    Args:
        image: Input PIL Image
        target_size: Target dimension
        stride: Model stride (must be divisible)

    Returns:
        Resized PIL Image
    """
    w, h = image.size

    # Calculate scale to fit target_size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Round to nearest stride multiple
    new_w = (new_w // stride) * stride
    new_h = (new_h // stride) * stride

    # Ensure minimum size
    new_w = max(stride, new_w)
    new_h = max(stride, new_h)

    return image.resize((new_w, new_h), Image.LANCZOS)


def convert_to_grayscale(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert image to grayscale numpy array.

    Args:
        image: Input image

    Returns:
        Grayscale numpy array
    """
    if isinstance(image, Image.Image):
        return np.array(image.convert('L'))
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def auto_rotate(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Auto-rotate image based on detected orientation.
    Uses simple heuristic: if width > height significantly, might be landscape.

    Args:
        image: Input image

    Returns:
        Rotated PIL Image if needed
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    w, h = image.size

    # If extremely wide, might need rotation
    if w / h > 2.0:
        # Could implement more sophisticated detection here
        pass

    return image


def batch_preprocess(
    images: list,
    **kwargs
) -> list:
    """
    Batch preprocess multiple images.

    Args:
        images: List of images
        **kwargs: Arguments for preprocess_for_detection

    Returns:
        List of preprocessed images
    """
    return [preprocess_for_detection(img, **kwargs) for img in images]


def crop_bbox(
    image: Union[np.ndarray, Image.Image],
    bbox: Tuple[int, int, int, int],
    margin: int = 5
) -> Image.Image:
    """
    Crop image region by bounding box with margin.

    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        margin: Extra margin around bbox

    Returns:
        Cropped PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    x1, y1, x2, y2 = bbox
    w, h = image.size

    # Add margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image.crop((x1, y1, x2, y2))


def validate_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int]
) -> bool:
    """
    Validate if bbox is within image boundaries and valid.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_size: Image (width, height)

    Returns:
        True if valid, False otherwise
    """
    x1, y1, x2, y2 = bbox
    w, h = image_size

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    if x2 <= x1 or y2 <= y1:
        return False

    return True
