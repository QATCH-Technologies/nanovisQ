import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass


@dataclass
class POICandidate:
    index: int
    confidence: float
    time: float
    poi_num: int


class PartialFillDetector:
    def __init__(self,
                 min_confidence_threshold: float = 0.3,
                 confidence_decay_factor: float = 0.8,
                 min_relative_distance: float = 0.1,
                 confidence_boost_threshold: float = 0.45):
        self.min_confidence_threshold = min_confidence_threshold
        self.confidence_decay_factor = confidence_decay_factor
        self.min_relative_distance = min_relative_distance
        self.confidence_boost_threshold = confidence_boost_threshold

    def process_predictions(self,
                            df: pd.DataFrame,
                            clf_results: Dict[int, Tuple[int, float]],
                            poi2_idx: int = -1) -> Dict[int, Tuple[int, float]]:
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
        r_time = df['Relative_time']
        poi4_time = r_time[positions[4][0]]
        poi5_time = r_time[positions[5][0]]
        poi6_time = r_time[positions[6][0]]

        if not (poi4_time < poi5_time < poi6_time):
            return False

        total_range = poi6_time - poi4_time
        if total_range <= 0:
            return False
        dist_4_5 = poi5_time - poi4_time
        dist_5_6 = poi6_time - poi5_time
        min_segment = total_range * 0.05
        if dist_4_5 < min_segment or dist_5_6 < min_segment:
            return False

        if dist_5_6 < dist_4_5 * 0.8:
            return False

        if poi2_idx != -1:
            poi2_time = r_time[poi2_idx]
            if poi2_time < poi4_time:
                dist_2_4 = poi4_time - poi2_time
                if dist_4_5 < dist_2_4 * 0.8:
                    return False

        return True

    def _boost_valid_configuration(self,
                                   df: pd.DataFrame,
                                   positions: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        boosted_positions = {}

        for poi_num, (idx, conf) in positions.items():
            if conf < self.confidence_boost_threshold:
                boosted_conf = min(conf * 1.25, 0.95)
                boosted_positions[poi_num] = (idx, boosted_conf)
            else:
                boosted_positions[poi_num] = (idx, conf)

        return boosted_positions

    def _fix_invalid_configuration(self,
                                   df: pd.DataFrame,
                                   positions: Dict[int, Tuple[int, float]],
                                   poi2_idx: int) -> Dict[int, Tuple[int, float]]:
        high_conf_pois = {
            poi_num for poi_num, (_, conf) in positions.items()
            if conf > 0.7
        }

        if not high_conf_pois:
            confidences = [(conf, poi_num)
                           for poi_num, (_, conf) in positions.items()]
            confidences.sort(reverse=True)
            high_conf_pois = {confidences[0][1], confidences[1][1]}

        fixed_positions = {}

        for poi_num in high_conf_pois:
            fixed_positions[poi_num] = positions[poi_num]

        for poi_num in [4, 5, 6]:
            if poi_num not in high_conf_pois:
                new_pos = self._find_valid_position(
                    df, poi_num, fixed_positions, positions, poi2_idx
                )
                if new_pos:
                    fixed_positions[poi_num] = new_pos
                else:
                    fixed_positions[poi_num] = positions[poi_num]

        return fixed_positions

    def _find_valid_position(self,
                             df: pd.DataFrame,
                             target_poi: int,
                             fixed_positions: Dict[int, Tuple[int, float]],
                             original_positions: Dict[int, Tuple[int, float]],
                             poi2_idx: int) -> Optional[Tuple[int, float]]:
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
        if max_idx - min_idx < 10:
            return None
        if target_poi == 5:
            if 4 in fixed_positions and 6 in fixed_positions:
                poi4_idx = fixed_positions[4][0]
                poi6_idx = fixed_positions[6][0]
                suggested_idx = int(poi4_idx + (poi6_idx - poi4_idx) * 0.45)
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
        detected_pois = set()
        missing_pois = set()
        positions = {}

        for poi_num in [4, 5, 6]:
            if poi_num in clf_results and clf_results[poi_num][0] != -1:
                detected_pois.add(poi_num)
                positions[poi_num] = clf_results[poi_num]
            else:
                missing_pois.add(poi_num)
        if missing_pois:
            if 6 in detected_pois:
                if 4 in missing_pois:
                    positions[4] = self._backtrack_find_poi(
                        df, 4, positions, poi2_idx)
                if 5 in missing_pois:
                    positions[5] = self._backtrack_find_poi(
                        df, 5, positions, poi2_idx)
            elif 5 in detected_pois and 4 in missing_pois:
                positions[4] = self._backtrack_find_poi(
                    df, 4, positions, poi2_idx)

        return positions

    def _backtrack_find_poi(self,
                            df: pd.DataFrame,
                            target_poi: int,
                            existing_positions: Dict[int, Tuple[int, float]],
                            poi2_idx: int) -> Tuple[int, float]:
        min_idx = poi2_idx + 1 if poi2_idx != -1 else 0
        max_idx = len(df) - 1

        if target_poi == 4 and 5 in existing_positions:
            max_idx = existing_positions[5][0] - 1
        elif target_poi == 5:
            if 4 in existing_positions:
                min_idx = existing_positions[4][0] + 1
            if 6 in existing_positions:
                max_idx = existing_positions[6][0] - 1
        if target_poi == 4:
            if 5 in existing_positions and 6 in existing_positions:
                poi5_idx = existing_positions[5][0]
                poi6_idx = existing_positions[6][0]
                dist_5_6 = poi6_idx - poi5_idx
                suggested_idx = max(min_idx, poi5_idx - dist_5_6)
                return (suggested_idx, self.min_confidence_threshold * self.confidence_decay_factor)

        elif target_poi == 5:
            if 4 in existing_positions and 6 in existing_positions:
                poi4_idx = existing_positions[4][0]
                poi6_idx = existing_positions[6][0]
                suggested_idx = int(poi4_idx + (poi6_idx - poi4_idx) * 0.45)
                return (suggested_idx, self.min_confidence_threshold * self.confidence_decay_factor)

        # Default: place in middle of search range
        suggested_idx = (min_idx + max_idx) // 2
        return (suggested_idx, self.min_confidence_threshold * self.confidence_decay_factor * 0.8)

    def _calculate_dynamic_threshold(self,
                                     positions: Dict[int, Tuple[int, float]]) -> float:
        confidences = [conf for _, conf in positions.values()]

        if not confidences:
            return self.min_confidence_threshold

        max_conf = max(confidences)
        mean_conf = np.mean(confidences)

        if max_conf > 0.8:
            dynamic_threshold = max(0.5, mean_conf * 0.8)
        elif max_conf > 0.6:
            dynamic_threshold = max(0.4, mean_conf * 0.7)
        else:
            dynamic_threshold = max(
                self.min_confidence_threshold, mean_conf * 0.6)

        return dynamic_threshold
