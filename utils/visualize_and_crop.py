"""
Visualization Helper
Draws bounding boxes on images to see what ML detected
"""

import cv2
import numpy as np
from pathlib import Path


def visualize_detections(image_path: str, detections: dict, output_path: str = None):
    """
    Draw bounding boxes on image.
    
    Args:
        image_path: Path to input image
        detections: Detection results from MLDetector
        output_path: Path to save visualization (optional)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    
    # Make a copy
    vis_image = image.copy()
    
    # Colors for each field
    colors = {
        'dealer_name': (255, 0, 0),      # Blue
        'model_name': (0, 255, 0),       # Green
        'horse_power': (0, 0, 255),      # Red
        'asset_cost': (255, 255, 0),     # Cyan
        'stamp': (255, 0, 255),          # Magenta
        'signature': (0, 255, 255)       # Yellow
    }
    
    # Draw each detection
    for field_name, dets in detections.items():
        if not dets:
            continue
        
        color = colors.get(field_name, (128, 128, 128))
        
        for i, det in enumerate(dets):
            bbox = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                3
            )
            
            # Draw label
            label = f"{field_name}: {conf:.2f}"
            
            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1] - text_h - 10),
                (bbox[0] + text_w, bbox[1]),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"✓ Visualization saved to: {output_path}")
    else:
        # Auto-generate output path
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_detections.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"✓ Visualization saved to: {output_path}")
    
    return vis_image


def visualize_crops(image_path: str, detections: dict, output_dir: str = "crops"):
    """
    Save individual crops for each detection.
    
    Args:
        image_path: Path to input image
        detections: Detection results
        output_dir: Directory to save crops
    """
    import os
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save crops
    for field_name, dets in detections.items():
        if not dets:
            continue
        
        for i, det in enumerate(dets):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Crop
            crop = image[y1:y2, x1:x2]
            
            # Save
            crop_filename = output_path / f"{field_name}_{i}_conf{det['confidence']:.2f}.jpg"
            cv2.imwrite(str(crop_filename), crop)
            print(f"✓ Saved crop: {crop_filename}")
    
    print(f"\n✓ All crops saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python visualize.py <image_path> [detections.json]")
        print("\nOr use in your code:")
        print("  from visualize import visualize_detections")
        print("  visualize_detections(image_path, ml_detections)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # If detections JSON provided, use it
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            data = json.load(f)
            # Extract detections from debug
            detections = data.get('debug', {}).get('all_predictions', {})
            
            # Convert format
            ml_dets = {}
            for field, preds in detections.items():
                if 'ml' in preds:
                    ml_dets[field] = [preds['ml']]
            
            visualize_detections(image_path, ml_dets)
    else:
        # Run ML detection
        sys.path.insert(0, '.')
        from utils.ml_detector import MLDetector
        
        detector = MLDetector("models/yolo11_best.pt")
        import cv2
        image = cv2.imread(image_path)
        detections = detector.detect(image)
        
        visualize_detections(image_path, detections)
        visualize_crops(image_path, detections)
