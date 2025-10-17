import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass


@dataclass
class POICandidate:
    """Represents a candidate POI detection."""
    index: int
    confidence: float
    time: float
    poi_num: int


class PartialFillDetector:
    """
    Handles partial fill detection for viscometer channel positions (POI 4, 5, 6).
    Ensures logical constraints and applies backtracking when needed.
    """

    def __init__(self,
                 min_confidence_threshold: float = 0.3,
                 confidence_decay_factor: float = 0.8,
                 min_relative_distance: float = 0.1,
                 confidence_boost_threshold: float = 0.45):
        """
        Initialize the partial fill detector.

        Args:
            min_confidence_threshold: Minimum confidence for initial candidate selection
            confidence_decay_factor: Factor to reduce threshold when backtracking
            min_relative_distance: Minimum relative distance between POIs (as fraction of total range)
            confidence_boost_threshold: Threshold below which to attempt confidence boosting
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.confidence_decay_factor = confidence_decay_factor
        self.min_relative_distance = min_relative_distance
        self.confidence_boost_threshold = confidence_boost_threshold

    def process_predictions(self,
                            df: pd.DataFrame,
                            clf_results: Dict[int, Tuple[int, float]],
                            poi2_idx: int = -1) -> Dict[int, Tuple[int, float]]:
        """
        Process classification results to handle partial fills with logical constraints.

        Args:
            df: DataFrame with time series data (must have 'Relative_time' column)
            clf_results: Dict mapping POI numbers (4,5,6) to (index, confidence) tuples
            poi2_idx: Index of POI2 (end-of-fill) for constraint checking

        Returns:
            Processed predictions with backtracking and constraints applied
        """
        # First, check if we have valid detections for all POIs
        all_pois_detected = all(
            poi_num in clf_results and clf_results[poi_num][0] != -1
            for poi_num in [4, 5, 6]
        )

        # Extract initial positions
        initial_positions = {}
        for poi_num, (idx, conf) in clf_results.items():
            if idx != -1:
                initial_positions[poi_num] = (idx, conf)

        # If all POIs are detected, validate the configuration
        if all_pois_detected:
            if self._validate_full_configuration(df, initial_positions, poi2_idx):
                # Configuration is valid, potentially just boost low confidence scores
                return self._boost_valid_configuration(df, initial_positions)
            else:
                # Configuration violates constraints, need to fix it
                return self._fix_invalid_configuration(df, initial_positions, poi2_idx)

        # Partial fill case - apply original logic
        return self._handle_partial_fill(df, clf_results, poi2_idx)

    def _validate_full_configuration(self,
                                     df: pd.DataFrame,
                                     positions: Dict[int, Tuple[int, float]],
                                     poi2_idx: int) -> bool:
        """
        Validate that all detected POIs form a valid configuration.
        """
        # Extract indices
        poi4_idx = positions[4][0]
        poi5_idx = positions[5][0]
        poi6_idx = positions[6][0]

        # Check basic ordering
        if not (poi4_idx < poi5_idx < poi6_idx):
            return False

        # Check minimum distance requirements
        total_range = poi6_idx - poi4_idx
        if total_range == 0:
            return False

        # POI4 to POI5 should be a reasonable portion of the total
        dist_4_5 = poi5_idx - poi4_idx
        dist_5_6 = poi6_idx - poi5_idx

        # Basic sanity check - each segment should be at least 5% of total
        min_segment = total_range * 0.05
        if dist_4_5 < min_segment or dist_5_6 < min_segment:
            return False

        # Check relative distances - POI5-POI6 should generally be >= POI4-POI5
        # Allow some flexibility for measurement variance
        if dist_5_6 < dist_4_5 * 0.8:  # Allow 20% variance
            return False

        # If POI2 exists, check that constraint
        if poi2_idx != -1 and poi2_idx < poi4_idx:
            dist_2_4 = poi4_idx - poi2_idx
            # POI4-POI5 distance should be >= POI2-POI4 distance (with some flexibility)
            if dist_4_5 < dist_2_4 * 0.8:  # Allow 20% variance
                return False

        return True

    def _boost_valid_configuration(self,
                                   df: pd.DataFrame,
                                   positions: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        """
        Boost confidence scores for valid configurations with low confidence detections.
        """
        boosted_positions = {}

        for poi_num, (idx, conf) in positions.items():
            if conf < self.confidence_boost_threshold:
                # Boost low confidence scores that are part of valid configuration
                # Boost by 25% but cap at 0.95
                boosted_conf = min(conf * 1.25, 0.95)
                boosted_positions[poi_num] = (idx, boosted_conf)
            else:
                boosted_positions[poi_num] = (idx, conf)

        return boosted_positions

    def _fix_invalid_configuration(self,
                                   df: pd.DataFrame,
                                   positions: Dict[int, Tuple[int, float]],
                                   poi2_idx: int) -> Dict[int, Tuple[int, float]]:
        """
        Fix an invalid configuration while preserving high-confidence detections.
        """
        # Identify which POIs have high confidence and should be trusted
        high_conf_pois = {
            poi_num for poi_num, (_, conf) in positions.items()
            if conf > 0.7  # High confidence threshold
        }

        # If all are low confidence, keep the most confident ones
        if not high_conf_pois:
            confidences = [(conf, poi_num)
                           for poi_num, (_, conf) in positions.items()]
            confidences.sort(reverse=True)
            # Keep top 2 most confident
            high_conf_pois = {confidences[0][1], confidences[1][1]}

        # Fix positions based on what we trust
        fixed_positions = {}

        # Always preserve high confidence detections
        for poi_num in high_conf_pois:
            fixed_positions[poi_num] = positions[poi_num]

        # Try to fix low confidence POIs
        for poi_num in [4, 5, 6]:
            if poi_num not in high_conf_pois:
                new_pos = self._find_valid_position(
                    df, poi_num, fixed_positions, positions, poi2_idx
                )
                if new_pos:
                    fixed_positions[poi_num] = new_pos
                else:
                    # Keep original if no better position found
                    fixed_positions[poi_num] = positions[poi_num]

        return fixed_positions

    def _find_valid_position(self,
                             df: pd.DataFrame,
                             target_poi: int,
                             fixed_positions: Dict[int, Tuple[int, float]],
                             original_positions: Dict[int, Tuple[int, float]],
                             poi2_idx: int) -> Optional[Tuple[int, float]]:
        """
        Find a valid position for a POI given constraints from fixed POIs.
        """
        # Define search bounds based on fixed POIs
        min_idx = 0
        max_idx = len(df) - 1

        if poi2_idx != -1:
            min_idx = max(min_idx, poi2_idx + 1)

        if target_poi == 4:
            if 5 in fixed_positions:
                max_idx = min(max_idx, fixed_positions[5][0] - 1)
        elif target_poi == 5:
            if 4 in fixed_positions:
                min_idx = max(min_idx, fixed_positions[4][0] + 1)
            if 6 in fixed_positions:
                max_idx = min(max_idx, fixed_positions[6][0] - 1)
        elif target_poi == 6:
            if 5 in fixed_positions:
                min_idx = max(min_idx, fixed_positions[5][0] + 1)

        # If search range is too small, return None
        if max_idx - min_idx < 10:
            return None

        # Find best position in range (simplified - could use model features)
        # For now, use geometric positioning as heuristic
        if target_poi == 5:
            if 4 in fixed_positions and 6 in fixed_positions:
                # Place POI5 geometrically between POI4 and POI6
                poi4_idx = fixed_positions[4][0]
                poi6_idx = fixed_positions[6][0]
                # Weight toward POI6 (since POI5-POI6 distance typically >= POI4-POI5)
                suggested_idx = int(poi4_idx + (poi6_idx - poi4_idx) * 0.45)

                # Adjust confidence based on how different from original
                original_idx = original_positions[5][0]
                distance_penalty = abs(suggested_idx - original_idx) / len(df)
                new_conf = max(
                    0.4, original_positions[5][1] - distance_penalty)

                return (suggested_idx, new_conf)

        return None

    def _handle_partial_fill(self,
                             df: pd.DataFrame,
                             clf_results: Dict[int, Tuple[int, float]],
                             poi2_idx: int) -> Dict[int, Tuple[int, float]]:
        """
        Handle true partial fill cases where some POIs are missing.
        """
        # Identify which POIs are detected
        detected_pois = set()
        missing_pois = set()
        positions = {}

        for poi_num in [4, 5, 6]:
            if poi_num in clf_results and clf_results[poi_num][0] != -1:
                detected_pois.add(poi_num)
                positions[poi_num] = clf_results[poi_num]
            else:
                missing_pois.add(poi_num)

        # Apply backtracking logic for missing POIs
        if missing_pois:
            # If POI6 is detected but POI4 or POI5 missing, must find them
            if 6 in detected_pois:
                if 4 in missing_pois:
                    positions[4] = self._backtrack_find_poi(
                        df, 4, positions, poi2_idx)
                if 5 in missing_pois:
                    positions[5] = self._backtrack_find_poi(
                        df, 5, positions, poi2_idx)

            # If POI5 is detected but POI4 missing, must find POI4
            elif 5 in detected_pois and 4 in missing_pois:
                positions[4] = self._backtrack_find_poi(
                    df, 4, positions, poi2_idx)

        return positions

    def _backtrack_find_poi(self,
                            df: pd.DataFrame,
                            target_poi: int,
                            existing_positions: Dict[int, Tuple[int, float]],
                            poi2_idx: int) -> Tuple[int, float]:
        """
        Find a missing POI using backtracking with relaxed thresholds.
        """
        # Define search range
        min_idx = poi2_idx + 1 if poi2_idx != -1 else 0
        max_idx = len(df) - 1

        if target_poi == 4 and 5 in existing_positions:
            max_idx = existing_positions[5][0] - 1
        elif target_poi == 5:
            if 4 in existing_positions:
                min_idx = existing_positions[4][0] + 1
            if 6 in existing_positions:
                max_idx = existing_positions[6][0] - 1

        # Use heuristic positioning for backtracked POIs
        if target_poi == 4:
            if 5 in existing_positions and 6 in existing_positions:
                # Place POI4 before POI5 with appropriate spacing
                poi5_idx = existing_positions[5][0]
                poi6_idx = existing_positions[6][0]
                dist_5_6 = poi6_idx - poi5_idx
                # POI4 should be at least dist_5_6 before POI5 (constraint)
                suggested_idx = max(min_idx, poi5_idx - dist_5_6)
                return (suggested_idx, self.min_confidence_threshold * self.confidence_decay_factor)

        elif target_poi == 5:
            if 4 in existing_positions and 6 in existing_positions:
                # Place POI5 between POI4 and POI6
                poi4_idx = existing_positions[4][0]
                poi6_idx = existing_positions[6][0]
                # Weight toward middle but slightly closer to POI4
                suggested_idx = int(poi4_idx + (poi6_idx - poi4_idx) * 0.45)
                return (suggested_idx, self.min_confidence_threshold * self.confidence_decay_factor)

        # Default: place in middle of search range
        suggested_idx = (min_idx + max_idx) // 2
        return (suggested_idx, self.min_confidence_threshold * self.confidence_decay_factor * 0.8)

    def _calculate_dynamic_threshold(self,
                                     positions: Dict[int, Tuple[int, float]]) -> float:
        """
        Calculate dynamic threshold based on the distribution of confidences.
        """
        confidences = [conf for _, conf in positions.values()]

        if not confidences:
            return self.min_confidence_threshold

        max_conf = max(confidences)
        mean_conf = np.mean(confidences)

        if max_conf > 0.8:
            # High confidence detection, use stricter threshold
            dynamic_threshold = max(0.5, mean_conf * 0.8)
        elif max_conf > 0.6:
            # Medium confidence, use moderate threshold
            dynamic_threshold = max(0.4, mean_conf * 0.7)
        else:
            # Low confidence overall, use lenient threshold
            dynamic_threshold = max(
                self.min_confidence_threshold, mean_conf * 0.6)

        return dynamic_threshold


# Integration function to use with your existing code
def enhance_poi_predictions(df: pd.DataFrame,
                            clf_results: Dict[int, Tuple[int, float]],
                            poi2_idx: int = -1,
                            min_confidence: float = 0.3) -> Dict[int, Tuple[int, float]]:
    """
    Enhance POI predictions for partial fills.

    Args:
        df: DataFrame with time series data
        clf_results: Raw classification results for POI 4, 5, 6
        poi2_idx: Index of POI2 (end-of-fill)
        min_confidence: Minimum confidence threshold

    Returns:
        Enhanced predictions with backtracking and constraints applied
    """
    detector = PartialFillDetector(min_confidence_threshold=min_confidence)
    return detector.process_predictions(df, clf_results, poi2_idx)
