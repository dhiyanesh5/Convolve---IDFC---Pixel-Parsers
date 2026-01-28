"""
IMPROVED Ensemble Module
- Better prediction fusion logic
- Field-specific confidence scoring
- Cross-field validation
- HP range awareness (20-65 typical)
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ImprovedEnsembleExtractor:
    """
    Enhanced ensemble extractor with smart prediction fusion.
    """
    
    def __init__(self):
        """Initialize improved ensemble extractor."""
        logger.info("Initializing Improved Ensemble Extractor...")
        
        # Field priorities - which source to trust most
        self.field_priorities = {
            'stamp': ['ml'],  # Only ML for visual elements
            'signature': ['ml'],
            'dealer_name': ['tesseract', 'vlm', 'ml'],
            'model_name': ['tesseract', 'vlm', 'ml'],
            'horse_power': ['tesseract', 'vlm', 'ml'],
            'asset_cost': ['tesseract', 'vlm', 'ml']
        }
        
        # Confidence thresholds for accepting predictions
        self.confidence_thresholds = {
            'dealer_name': 0.5,
            'model_name': 0.4,
            'horse_power': 0.4,
            'asset_cost': 0.3,
            'signature': 0.3,
            'stamp': 0.3
        }
        
        # HP validation (based on real data)
        self.HP_MIN = 15
        self.HP_MAX = 100
        self.HP_TYPICAL_MAX = 65
        
        # Known model-HP mappings (from training data)
        self.known_model_hp = {
            '744': 48,
            '855': 55,
            '735': 35,
            '575': 40,
            '475': 42,
            '265': 30,
            '385': 38,
            '555': 50,
            '605': 50,
        }
        
        logger.info("✓ Improved Ensemble initialized")
    
    def combine_predictions(self,
                           ml_detections: Dict,
                           tesseract_results: Dict,
                           vlm_results: Dict = None) -> Dict[str, Any]:
        """
        Intelligently combine predictions from all sources.
        
        Args:
            ml_detections: ML detection results
            tesseract_results: Tesseract OCR results
            vlm_results: Optional VLM results
            
        Returns:
            Combined predictions with metadata
        """
        logger.info("Combining predictions with improved logic...")
        
        combined = {}
        
        # Process each field
        for field_name in ['dealer_name', 'model_name', 'horse_power', 'asset_cost',
                          'stamp', 'signature']:
            
            combined[field_name] = self._combine_field_improved(
                field_name,
                ml_detections,
                tesseract_results,
                vlm_results
            )
        
        # Cross-validate fields
        combined = self._cross_validate_fields(combined)
        
        # Calculate overall confidence
        field_confs = []
        for field_name, field_data in combined.items():
            if field_name not in ['_metadata']:
                conf = field_data.get('confidence', 0)
                # Weight important fields more
                weight = 2.0 if field_name in ['horse_power', 'asset_cost'] else 1.0
                field_confs.extend([conf] * int(weight))
        
        overall_confidence = sum(field_confs) / len(field_confs) if field_confs else 0
        
        combined['_metadata'] = {
            'overall_confidence': overall_confidence,
            'sources_used': self._get_sources_used(combined),
            'validation_flags': self._get_validation_flags(combined)
        }
        
        logger.info(f"✓ Ensemble complete. Overall confidence: {overall_confidence:.2%}")
        
        return combined
    
    def _combine_field_improved(self,
                               field_name: str,
                               ml_detections: Dict,
                               tesseract_results: Dict,
                               vlm_results: Dict = None) -> Dict[str, Any]:
        """
        Improved field combination with smarter logic.
        """
        # Collect all predictions
        predictions = {}
        
        # ML predictions
        ml_dets = ml_detections.get(field_name, [])
        if ml_dets:
            best_ml = ml_dets[0]
            predictions['ml'] = {
                'value': None,
                'bbox': best_ml['bbox'],
                'confidence': best_ml['confidence'],
                'source': 'ml_yolo11'
            }
        
        # Tesseract predictions
        tess_field = tesseract_results.get(field_name, {})
        if tess_field and tess_field.get('value'):
            predictions['tesseract'] = tess_field
        
        # VLM predictions
        if vlm_results:
            vlm_field = vlm_results.get(field_name, {})
            if vlm_field and vlm_field.get('value'):
                predictions['vlm'] = vlm_field
        
        # Smart selection logic
        if field_name in ['stamp', 'signature']:
            # Visual elements - only use ML
            if 'ml' in predictions:
                return {
                    'present': True,
                    'bbox': predictions['ml']['bbox'],
                    'confidence': predictions['ml']['confidence'],
                    'source': 'ml'
                }
            else:
                return {
                    'present': False,
                    'bbox': [],
                    'confidence': 0.0,
                    'source': 'none'
                }
        
        # Text fields - use confidence-weighted selection
        best_pred = None
        best_score = 0.0
        best_source = 'none'
        
        for source in self.field_priorities.get(field_name, ['tesseract', 'vlm', 'ml']):
            if source not in predictions:
                continue
            
            pred = predictions[source]
            value = pred.get('value')
            
            if value is None or value == '' or value == 0:
                continue
            
            # Calculate weighted score
            base_conf = pred.get('confidence', 0)
            
            # Apply source preference weights
            source_weights = {
                'tesseract': 1.2,  # Prefer Tesseract for text
                'vlm': 1.0,
                'ml': 0.8
            }
            
            score = base_conf * source_weights.get(source, 1.0)
            
            # Field-specific bonuses
            if field_name == 'horse_power':
                # Bonus for values in typical range
                hp_val = value
                if isinstance(hp_val, int) and self.HP_MIN <= hp_val <= self.HP_TYPICAL_MAX:
                    score *= 1.3
                    logger.debug(f"HP bonus for {hp_val}: {score:.3f}")
            
            if field_name == 'dealer_name':
                # Bonus for longer names (more likely correct)
                if isinstance(value, str) and len(value) > 15:
                    score *= 1.1
            
            if score > best_score:
                best_score = score
                best_pred = pred
                best_source = source
        
        # Return best prediction or empty
        if best_pred:
            result = {
                'value': best_pred['value'],
                'confidence': best_pred.get('confidence', 0),
                'source': best_source,
                'bbox': best_pred.get('bbox'),
                'all_predictions': predictions
            }
            
            # Add metadata if available
            if 'raw_ocr' in best_pred:
                result['raw_ocr'] = best_pred['raw_ocr']
            
            return result
        else:
            # No valid prediction
            empty_val = '' if field_name in ['dealer_name', 'model_name'] else 0
            return {
                'value': empty_val,
                'confidence': 0.0,
                'source': 'none',
                'bbox': None,
                'all_predictions': predictions
            }
    
    def _cross_validate_fields(self, combined: Dict) -> Dict:
        """
        Cross-validate fields using domain knowledge.
        """
        logger.info("Cross-validating fields...")
        
        # Extract model and HP
        model_data = combined.get('model_name', {})
        hp_data = combined.get('horse_power', {})
        
        model_value = model_data.get('value', '')
        hp_value = hp_data.get('value', 0)
        
        # Check if model matches known HP
        if model_value and hp_value:
            for model_num, expected_hp in self.known_model_hp.items():
                if model_num in str(model_value):
                    if hp_value == expected_hp:
                        # Perfect match - boost confidence
                        logger.info(f"✓ Model {model_value} matches expected HP {expected_hp}")
                        combined['horse_power']['confidence'] = min(0.95, hp_data.get('confidence', 0) * 1.3)
                        combined['horse_power']['validation'] = 'cross_validated_with_model'
                    else:
                        # Mismatch - flag it
                        logger.warning(f"⚠ Model {model_value} typically has {expected_hp} HP, got {hp_value}")
                        combined['horse_power']['validation'] = 'model_hp_mismatch'
                    break
        
        # Validate HP range
        if hp_value > 0:
            if hp_value < self.HP_MIN or hp_value > self.HP_MAX:
                logger.warning(f"⚠ HP {hp_value} out of valid range ({self.HP_MIN}-{self.HP_MAX})")
                combined['horse_power']['confidence'] *= 0.5
                combined['horse_power']['validation'] = 'out_of_range'
            elif hp_value > self.HP_TYPICAL_MAX:
                logger.info(f"ℹ HP {hp_value} above typical max ({self.HP_TYPICAL_MAX}) but valid")
                combined['horse_power']['validation'] = 'above_typical_range'
        
        # Validate asset cost
        cost_data = combined.get('asset_cost', {})
        cost_value = cost_data.get('value', 0)
        
        if cost_value > 0:
            if cost_value < 100000:
                logger.warning(f"⚠ Asset cost {cost_value} seems low (<1L)")
                combined['asset_cost']['validation'] = 'below_typical_range'
            elif cost_value > 20000000:
                logger.warning(f"⚠ Asset cost {cost_value} seems high (>2Cr)")
                combined['asset_cost']['validation'] = 'above_typical_range'
        
        return combined
    
    def _get_sources_used(self, combined: Dict) -> Dict[str, int]:
        """Count which sources were used."""
        sources = {'ml': 0, 'tesseract': 0, 'vlm': 0, 'none': 0}
        
        for field_name, field_data in combined.items():
            if field_name == '_metadata':
                continue
            
            source = field_data.get('source', 'none')
            if source in sources:
                sources[source] += 1
        
        return sources
    
    def _get_validation_flags(self, combined: Dict) -> Dict[str, str]:
        """Get validation flags for each field."""
        flags = {}
        
        for field_name, field_data in combined.items():
            if field_name == '_metadata':
                continue
            
            validation = field_data.get('validation', 'ok')
            if validation != 'ok':
                flags[field_name] = validation
        
        return flags
    
    def validate_and_refine(self, combined: Dict) -> Dict[str, Any]:
        """
        Validate and refine combined predictions.
        (For backward compatibility - validation already done in combine_predictions)
        
        Args:
            combined: Combined predictions from combine_predictions
            
        Returns:
            Refined predictions (same as input, validation already done)
        """
        logger.info("Validation complete (already done in combine_predictions)")
        return combined
    
    def format_output(self, refined: Dict) -> Dict[str, Any]:
        """Format final output for JSON export."""
        
        output = {
            'dealer_name': refined.get('dealer_name', {}).get('value', ''),
            'model_name': refined.get('model_name', {}).get('value', ''),
            'horse_power': refined.get('horse_power', {}).get('value', 0),
            'asset_cost': refined.get('asset_cost', {}).get('value', 0),
            'signature': {
                'present': refined.get('signature', {}).get('present', False),
                'bbox': refined.get('signature', {}).get('bbox', []),
                'confidence': refined.get('signature', {}).get('confidence', 0.0)
            },
            'stamp': {
                'present': refined.get('stamp', {}).get('present', False),
                'bbox': refined.get('stamp', {}).get('bbox', []),
                'confidence': refined.get('stamp', {}).get('confidence', 0.0)
            },
            'metadata': {
                'overall_confidence': refined.get('_metadata', {}).get('overall_confidence', 0),
                'sources_used': refined.get('_metadata', {}).get('sources_used', {}),
                'validation_flags': refined.get('_metadata', {}).get('validation_flags', {}),
                'field_sources': {
                    'dealer_name': refined.get('dealer_name', {}).get('source', 'none'),
                    'model_name': refined.get('model_name', {}).get('source', 'none'),
                    'horse_power': refined.get('horse_power', {}).get('source', 'none'),
                    'asset_cost': refined.get('asset_cost', {}).get('source', 'none'),
                    'signature': refined.get('signature', {}).get('source', 'none'),
                    'stamp': refined.get('stamp', {}).get('source', 'none')
                }
            }
        }
        
        # Add debug info with all predictions
        if any(field.get('all_predictions') for field in refined.values() if isinstance(field, dict)):
            output['debug'] = {
                'all_predictions': {
                    k: v.get('all_predictions', {})
                    for k, v in refined.items()
                    if isinstance(v, dict) and 'all_predictions' in v
                },
                'raw_ocr_values': {
                    k: v.get('raw_ocr', '')
                    for k, v in refined.items()
                    if isinstance(v, dict) and 'raw_ocr' in v
                }
            }
        
        return output