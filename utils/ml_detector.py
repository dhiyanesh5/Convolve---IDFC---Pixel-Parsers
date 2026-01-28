"""
ML Detector - YOLO11 Wrapper
Detects objects (Dealer name, Model name, HP, Asset cost, Stamp, Signature)
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Install: pip install ultralytics")


class MLDetector:
    """YOLO11-based object detector for invoice fields."""
    
    # Class mapping (must match your training)
    CLASS_NAMES = {
        0: "dealer_name",
        1: "model_name", 
        2: "horse_power",
        3: "asset_cost",
        4: "signature",
        5: "stamp"
    }
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.15):
        """
        Initialize ML detector.
        
        Args:
            model_path: Path to YOLO11 best.pt file
            confidence_threshold: Minimum confidence for detections
        """
        logger.info(f"Initializing ML Detector...")
        
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install: pip install ultralytics")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load YOLO model
        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"✓ ML Detector loaded from: {self.model_path}")
        logger.info(f"✓ Confidence threshold: {confidence_threshold}")
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run detection on image.
        
        Args:
            image: Input image as numpy array (BGR or RGB)
            
        Returns:
            Dictionary with detections for each class
        """
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                # YOLO expects RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assuming BGR from OpenCV, convert to RGB
                    image_rgb = image[:, :, ::-1]
                else:
                    image_rgb = image
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Run inference
            results = self.model(pil_image, conf=self.confidence_threshold, verbose=False)
            
            # Parse results
            detections = self._parse_results(results[0])
            
            logger.info(f"✓ ML Detection complete: {len(detections)} objects found")
            
            return detections
            
        except Exception as e:
            logger.error(f"ML detection error: {e}", exc_info=True)
            return self._empty_detections()
    
    def _parse_results(self, result) -> Dict[str, Any]:
        """Parse YOLO results into structured format."""
        detections = {
            'dealer_name': [],
            'model_name': [],
            'horse_power': [],
            'asset_cost': [],
            'signature': [],
            'stamp': []
            
        }
        
        # Extract boxes
        boxes = result.boxes
        
        for box in boxes:
            # Get box data
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Get class name
            class_name = self.CLASS_NAMES.get(cls, f"unknown_{cls}")
            
            # Create detection dict
            detection = {
                'bbox': [
                    int(xyxy[0]),  # x1
                    int(xyxy[1]),  # y1
                    int(xyxy[2]),  # x2
                    int(xyxy[3])   # y2
                ],
                'confidence': conf,
                'class': class_name,
                'class_id': cls
            }
            
            # Add to appropriate list
            if class_name in detections:
                detections[class_name].append(detection)
        
        # Sort by confidence (highest first)
        for key in detections:
            detections[key].sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _empty_detections(self) -> Dict[str, Any]:
        """Return empty detections structure."""
        return {
            'dealer_name': [],
            'model_name': [],
            'horse_power': [],
            'asset_cost': [],
            'signature': [],
            'stamp': []
        }
    
    def visualize(self, image: np.ndarray, detections: Dict[str, Any], 
                  output_path: str = None) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: Detection results from detect()
            output_path: Optional path to save visualization
            
        Returns:
            Image with bounding boxes drawn
        """
        import cv2
        
        # Copy image
        vis_image = image.copy()
        
        # Color map for each class
        colors = {
            'dealer_name': (255, 0, 0),      # Blue
            'model_name': (0, 255, 0),       # Green
            'horse_power': (0, 0, 255),      # Red
            'asset_cost': (255, 255, 0),     # Cyan
            'signature': (255, 0, 255),          # Magenta
            'stamp': (0, 255, 255)       # Yellow
        }
        
        # Draw boxes
        for class_name, dets in detections.items():
            color = colors.get(class_name, (128, 128, 128))
            
            for det in dets:
                bbox = det['bbox']
                conf = det['confidence']
                
                # Draw rectangle
                cv2.rectangle(
                    vis_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color,
                    2
                )
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(
                    vis_image,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"✓ Visualization saved to: {output_path}")
        
        return vis_image


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ml_detector.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Load detector
    detector = MLDetector(model_path)
    
    # Load image
    import cv2
    image = cv2.imread(image_path)
    
    # Detect
    detections = detector.detect(image)
    
    # Print results
    print("\n" + "="*60)
    print("ML DETECTION RESULTS")
    print("="*60)
    
    for class_name, dets in detections.items():
        print(f"\n{class_name.upper()}: {len(dets)} detections")
        for i, det in enumerate(dets, 1):
            print(f"  {i}. Confidence: {det['confidence']:.3f}, BBox: {det['bbox']}")
    
    # Visualize
    vis = detector.visualize(image, detections, "ml_detection_output.jpg")
    print(f"\n✓ Visualization saved to: ml_detection_output.jpg")