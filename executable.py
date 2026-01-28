"""
Complete Ensemble Invoice Extraction Pipeline
Combines ML (YOLO11) + Tesseract OCR + (Optional VLM)
"""

import sys
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any
import time
 
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
from utils.ml_detector import MLDetector
from utils.ocr_extractor import ImprovedTesseractOCR as TesseractOCR
from utils.ensemble import ImprovedEnsembleExtractor as EnsembleExtractor

class InvoiceExtractionPipeline:
    """Complete invoice extraction pipeline with ensemble approach."""
    
    def __init__(self, 
                 ml_model_path: str,
                 tesseract_path: str = None,
                 use_vlm: bool = False):
        """
        Initialize pipeline.
        
        Args:
            ml_model_path: Path to YOLO11 best.pt model
            tesseract_path: Path to tesseract.exe (optional)
            use_vlm: Whether to use VLM (requires more setup)
        """
        logger.info("="*80)
        logger.info("INITIALIZING INVOICE EXTRACTION PIPELINE")
        logger.info("="*80)
        
        # Initialize ML detector
        logger.info("\n1. Loading ML Model (YOLO11)...")
        self.ml_detector = MLDetector(ml_model_path)
        
        # Initialize Tesseract OCR
        logger.info("\n2. Loading Tesseract OCR...")
        self.ocr = TesseractOCR(tesseract_cmd=tesseract_path, lang='eng+hin')
        
        # Initialize Ensemble
        logger.info("\n3. Initializing Ensemble Logic...")
        self.ensemble = EnsembleExtractor()
        
        # Optional VLM
        self.use_vlm = use_vlm
        self.vlm = None
        
        if use_vlm:
            logger.info("\n4. Loading VLM (Qwen2-VL)...")
            try:
                from utils.vlm_extractor import VLMExtractor
                import os
                # Correct path pointing to root/models
                local_model_path = os.path.join("models", "qwen2.5-vl-3b")
                
                # SAFETY CHECK: If model folder doesn't exist, skip immediately
                if not os.path.exists(local_model_path):
                    logger.warning(f"Model not found at {local_model_path}. Skipping VLM.")
                    self.use_vlm = False
                else:
                    self.vlm = VLMExtractor(model_path=local_model_path)
                    
                    # If initialization failed internally (e.g. OOM), disable it
                    if not self.vlm.available:
                         logger.warning("VLM failed to initialize (likely Low RAM). Disabling VLM.")
                         self.use_vlm = False

            except Exception as e:
                logger.warning(f"VLM crashed during load: {e}")
                logger.warning("System switching to 'Safe Mode' (YOLO + OCR only).")
                self.use_vlm = False
    
    def process_image(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Process a single invoice image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save JSON output
            
        Returns:
            Extraction results as dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING: {Path(image_path).name}")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        # Load image
        logger.info("Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        logger.info(f"✓ Image loaded: {image.shape}")
        
        # Step 1: ML Detection
        logger.info("\n[1/3] Running ML Detection (YOLO11)...")
        ml_detections = self.ml_detector.detect(image)
        self._log_detections(ml_detections, "ML")
        
        # Step 2: Tesseract OCR
        logger.info("\n[2/3] Running Tesseract OCR...")
        ocr_results = self.ocr.extract_text(image)
        logger.info(f"✓ OCR complete: {ocr_results['num_blocks']} text blocks")
        
        # Extract fields from OCR with ML guidance
        tesseract_fields = self.ocr.extract_fields_with_crops(image, ml_detections)
        self._log_fields(tesseract_fields, "Tesseract")
        
        # Step 3: Optional VLM
        vlm_results = None
        if self.use_vlm and self.vlm:
            logger.info("\n[3/3] Running VLM Extraction...")
            vlm_results = self.vlm.extract(image)
            self._log_fields(vlm_results, "VLM")
        
        # Step 4: Ensemble
        logger.info("\n[ENSEMBLE] Combining all predictions...")
        combined = self.ensemble.combine_predictions(
            ml_detections,
            tesseract_fields,
            vlm_results
        )
        
        # Step 5: Validate and refine
        logger.info("[ENSEMBLE] Validating and refining...")
        refined = self.ensemble.validate_and_refine(combined)
        
        # Step 6: Format output
        output = self.ensemble.format_output(refined)
        
        # Add metadata
        output['doc_id'] = Path(image_path).stem
        output['processing_time_sec'] = round(time.time() - start_time, 2)
        
        # Log final results
        self._log_final_results(output)
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"\n✓ Results saved to: {output_path}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ PROCESSING COMPLETE ({output['processing_time_sec']}s)")
        logger.info(f"{'='*80}\n")
        
        return output
    
    def _log_detections(self, detections: Dict, source: str):
        """Log detection results."""
        total = sum(len(dets) for dets in detections.values())
        logger.info(f"✓ {source} found {total} detections:")
        for field, dets in detections.items():
            if dets:
                logger.info(f"  - {field}: {len(dets)} (conf: {dets[0]['confidence']:.2f})")
    
    def _log_fields(self, fields: Dict, source: str):
        """Log field extraction results."""
        logger.info(f"✓ {source} extracted {len(fields)} fields:")
        for field, data in fields.items():
            if data:
                val = data.get('value', 'N/A')
                conf = data.get('confidence', 0)
                logger.info(f"  - {field}: {val} (conf: {conf:.2f})")
    
    def _log_final_results(self, output: Dict):
        """Log final extraction results."""
        logger.info("\n" + "="*80)
        logger.info("FINAL EXTRACTION RESULTS")
        logger.info("="*80)
        logger.info(f"Dealer:    {output['dealer_name']}")
        logger.info(f"Model:     {output['model_name']}")
        logger.info(f"HP:        {output['horse_power']}")
        logger.info(f"Cost:      {output['asset_cost']}")
        logger.info(f"Signature: {output['signature']['present']} (conf: {output['signature']['confidence']:.2f})")
        logger.info(f"Stamp:     {output['stamp']['present']} (conf: {output['stamp']['confidence']:.2f})")
        logger.info(f"\nOverall Confidence: {output['metadata']['overall_confidence']:.1%}")
        logger.info(f"Sources Used: {output['metadata']['sources_used']}")
        logger.info("="*80)


def main():
    """Main execution function."""
    if len(sys.argv) < 3:
        print("\nUsage: python executable_ensemble.py <image_path> <output_json> [options]")
        print("\nRequired:")
        print("  image_path    Path to invoice image")
        print("  output_json   Path to save results JSON")
        print("\nOptional:")
        print("  --ml-model <path>        Path to YOLO11 best.pt (default: models/yolo11_best.pt)")
        print("  --tesseract <path>       Path to tesseract.exe (default: auto-detect)")
        print("  --use-vlm                Enable VLM extraction (slower)")
        print("\nExample:")
        print("  python executable_ensemble.py data/train_images/invoice.png results/output.json")
        print("  python executable_ensemble.py invoice.png output.json --ml-model models/best.pt --tesseract C:/Tesseract/tesseract.exe")
        sys.exit(1)
    
    # Parse arguments
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Optional arguments
    ml_model_path = "models/yolo11_best.pt"
    tesseract_path = None
    use_vlm = False
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--ml-model' and i + 1 < len(sys.argv):
            ml_model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--tesseract' and i + 1 < len(sys.argv):
            tesseract_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--use-vlm':
            use_vlm = True
            i += 1
        else:
            i += 1
    
    try:
        # Initialize pipeline
        pipeline = InvoiceExtractionPipeline(
            ml_model_path=ml_model_path,
            tesseract_path=tesseract_path,
            use_vlm=use_vlm
        )
        
        # Process image
        result = pipeline.process_image(image_path, output_path)
        
        print("\n✅ SUCCESS!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()