"""
IMPROVED OCR Extractor using Tesseract
- Better preprocessing for cropped regions
- Enhanced digit/text extraction
- HP range validation (20-65 typical)
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image, ImageEnhance
import re
import cv2

logger = logging.getLogger(__name__)

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available. Install: pip install pytesseract")

# IMPORTANT: Set default Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ImprovedTesseractOCR:
    """Enhanced Tesseract OCR with preprocessing for better accuracy."""
    
    def __init__(self, tesseract_cmd: str = None, lang: str = 'eng+hin'):
        """
        Initialize Tesseract OCR.
        
        Args:
            tesseract_cmd: Path to tesseract.exe (if not in PATH)
            lang: Language codes (e.g., 'eng+hin' for English + Hindi)
        """
        logger.info("Initializing Improved Tesseract OCR...")
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not available. Install: pip install pytesseract")
        
        # Set tesseract command path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info(f"✓ Tesseract path set to: {tesseract_cmd}")
        
        self.lang = lang
        
        # Test Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"✓ Tesseract version: {version}")
            logger.info(f"✓ Languages: {lang}")
        except Exception as e:
            logger.error(f"Tesseract test failed: {e}")
            raise
        
        # Devanagari digit mapping
        self.devanagari_digits = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        
        # HP validation range (based on your observation)
        self.HP_MIN = 15
        self.HP_MAX = 100  # Extended slightly for safety
        self.HP_TYPICAL_MAX = 65  # Typical upper bound
        
        logger.info(f"✓ HP range set to: {self.HP_MIN}-{self.HP_MAX} (typical: {self.HP_TYPICAL_MAX})")
    
    def preprocess_crop(self, crop: np.ndarray, field_type: str = 'text') -> Image.Image:
        """
        Preprocess cropped region for better OCR.
        
        Args:
            crop: Cropped image region (numpy array)
            field_type: 'text', 'digit', or 'mixed'
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Resize if too small (helps OCR)
        h, w = gray.shape
        if h < 50 or w < 100:
            scale = max(50/h, 100/w, 1.5)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # For digits, use more aggressive preprocessing
        if field_type == 'digit':
            # Denoise
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Adaptive threshold (better for varying lighting)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 10
            )
        else:
            # For text, use simple threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert to PIL
        pil_image = Image.fromarray(binary)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        return pil_image
    
    def extract_text_from_crop(self, crop: np.ndarray, field_name: str) -> Tuple[str, float]:
        """
        Extract text from a cropped region with field-specific preprocessing.
        
        Args:
            crop: Cropped image region
            field_name: Field name for context
            
        Returns:
            (extracted_text, confidence)
        """
        # Determine field type
        if field_name in ['horse_power', 'asset_cost']:
            field_type = 'digit'
            config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
        elif field_name == 'model_name':
            field_type = 'mixed'
            config = '--psm 7 --oem 3'
        else:
            field_type = 'text'
            config = '--psm 6 --oem 3'
        
        # Preprocess
        preprocessed = self.preprocess_crop(crop, field_type)
        
        # Try multiple OCR attempts with different configs
        attempts = [
            (config, self.lang),
            ('--psm 6 --oem 3', self.lang),
            ('--psm 7 --oem 3', 'eng'),  # English only fallback
        ]
        
        best_text = ""
        best_conf = 0.0
        
        for cfg, lang in attempts:
            try:
                # Get detailed output
                data = pytesseract.image_to_data(
                    preprocessed,
                    lang=lang,
                    config=cfg,
                    output_type=pytesseract.Output.DICT
                )
                
                # Combine all text with confidence
                texts = []
                confs = []
                for i, text in enumerate(data['text']):
                    if text.strip() and int(data['conf'][i]) > 0:
                        texts.append(text.strip())
                        confs.append(float(data['conf'][i]))
                
                if texts:
                    combined_text = ' '.join(texts)
                    avg_conf = sum(confs) / len(confs) / 100.0
                    
                    if avg_conf > best_conf:
                        best_text = combined_text
                        best_conf = avg_conf
                        
            except Exception as e:
                logger.debug(f"OCR attempt failed with {cfg}: {e}")
                continue
        
        # Convert Devanagari
        best_text = self._convert_devanagari(best_text)
        
        return best_text, best_conf
    
    def _convert_devanagari(self, text: str) -> str:
        """Convert Devanagari/Hindi digits to Latin."""
        for dev, lat in self.devanagari_digits.items():
            text = text.replace(dev, lat)
        return text
    
    def clean_and_validate(self, text: str, field_name: str) -> Any:
        """
        Clean and validate extracted text for specific field.
        
        Args:
            text: Raw extracted text
            field_name: Field name for validation
            
        Returns:
            Cleaned and validated value
        """
        if not text:
            return None
        
        text = text.upper().strip()
        
        if field_name == 'horse_power':
            # Extract digits only
            digits = re.findall(r'\d+', text)
            if not digits:
                return None
            
            # Try each number found
            for digit_str in digits:
                hp = int(digit_str)
                
                # Validate range
                if self.HP_MIN <= hp <= self.HP_MAX:
                    # Add confidence boost if in typical range
                    if hp <= self.HP_TYPICAL_MAX:
                        logger.info(f"✓ HP {hp} in typical range ({self.HP_MIN}-{self.HP_TYPICAL_MAX})")
                    return hp
            
            # If found but out of range, log warning
            if digits:
                logger.warning(f"HP value {digits[0]} out of valid range ({self.HP_MIN}-{self.HP_MAX})")
            return None
            
        elif field_name == 'asset_cost':
            # Remove all non-digits
            text_clean = re.sub(r'[^\d]', '', text)
            if not text_clean:
                return None
            
            cost = int(text_clean)
            
            # Validate range (1L to 20Cr typical for tractors)
            if 100000 <= cost <= 20000000:
                return cost
            else:
                logger.warning(f"Asset cost {cost} out of typical range (1L-2Cr)")
                # Still return it, but with warning
                return cost if cost > 0 else None
                
        elif field_name == 'model_name':
            # Clean model name
            # Remove extra spaces
            text = ' '.join(text.split())
            
            # Common patterns: "MAHINDRA 575 DI", "SWARAJ 744 FE"
            # Keep alphanumeric and common separators
            text = re.sub(r'[^A-Z0-9\s\-\+]', '', text)
            
            return text if len(text) >= 3 else None
        
        elif field_name == 'horse_power':
            # Fix common OCR error: Letter 'O' -> Digit '0'
            text = text.replace('O', '0').replace('o', '0')
            
            # Extract digits only
            digits = re.findall(r'\d+', text)
            
        elif field_name == 'dealer_name':
            # Keep full dealer name
            text = ' '.join(text.split())
            return text if len(text) >= 5 else None
        
        return text
    
    def extract_fields_with_crops(self, image: np.ndarray, ml_detections: Dict) -> Dict[str, Any]:
        """
        Extract fields using ML-detected crop regions.
        This is the main improvement - using crops for better OCR.
        
        Args:
            image: Full input image
            ml_detections: ML detection results with bounding boxes
            
        Returns:
            Extracted fields with confidence
        """
        fields = {}
        
        # Convert image if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Process each field type
        for field_name in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            detections = ml_detections.get(field_name, [])
            
            if not detections:
                logger.debug(f"No ML detection for {field_name}")
                continue
            
            # Use highest confidence detection
            best_det = detections[0]
            bbox = best_det['bbox']
            ml_conf = best_det['confidence']
            
            # Crop region with small padding
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # Add 5% padding
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                logger.warning(f"Empty crop for {field_name}")
                continue
            
            # Extract text from crop
            logger.info(f"Extracting {field_name} from crop {bbox}...")
            raw_text, ocr_conf = self.extract_text_from_crop(crop, field_name)
            
            logger.info(f"  Raw OCR: '{raw_text}' (conf: {ocr_conf:.2f})")
            
            # Clean and validate
            clean_value = self.clean_and_validate(raw_text, field_name)
            
            if clean_value is not None:
                # Combine ML detection confidence with OCR confidence
                combined_conf = (ml_conf * 0.6 + ocr_conf * 0.4)
                
                fields[field_name] = {
                    'value': clean_value,
                    'bbox': bbox,
                    'confidence': combined_conf,
                    'source': 'tesseract_ml_guided_improved',
                    'raw_ocr': raw_text,
                    'ml_confidence': ml_conf,
                    'ocr_confidence': ocr_conf
                }
                
                logger.info(f"✓ Extracted {field_name}: {clean_value} (conf: {combined_conf:.2f})")
            else:
                logger.warning(f"Failed to validate {field_name}: '{raw_text}'")
        
        return fields
    
    def extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract full text from image (for compatibility with old pipeline).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Convert to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Extract text with details
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract full text
            full_text = pytesseract.image_to_string(
                pil_image,
                lang=self.lang
            )
            
            # Convert Devanagari digits
            full_text_converted = self._convert_devanagari(full_text)
            
            # Parse into structured format
            text_blocks = self._parse_to_blocks(data)
            
            logger.info(f"✓ Tesseract OCR complete: {len(text_blocks)} text blocks")
            
            return {
                'full_text': full_text_converted,
                'full_text_original': full_text,
                'text_blocks': text_blocks,
                'num_blocks': len(text_blocks)
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}", exc_info=True)
            return {
                'full_text': '',
                'full_text_original': '',
                'text_blocks': [],
                'num_blocks': 0
            }
    
    def _parse_to_blocks(self, data: Dict) -> List[Dict]:
        """Parse Tesseract output data into text blocks."""
        blocks = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            # Skip empty text
            if int(data['conf'][i]) < 0:  # -1 means no text detected
                continue
            
            text = data['text'][i].strip()
            if not text:
                continue
            
            # Get bounding box
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Convert Devanagari
            text_converted = self._convert_devanagari(text)
            
            blocks.append({
                'text': text_converted,
                'text_original': text,
                'confidence': float(data['conf'][i]) / 100.0,
                'bbox': [x, y, x + w, y + h]
            })
        
        return blocks
    
    # Keep the old method for backward compatibility
    def extract_fields(self, image: np.ndarray, ml_detections: Dict = None) -> Dict[str, Any]:
        """
        Extract specific fields using OCR + ML bounding boxes.
        
        Args:
            image: Input image
            ml_detections: Optional ML detection results to guide extraction
            
        Returns:
            Extracted fields
        """
        if ml_detections:
            return self.extract_fields_with_crops(image, ml_detections)
        else:
            # Fallback - no ML detections provided
            return {}