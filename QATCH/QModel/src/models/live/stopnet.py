import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Optional
from typing import Dict
from collections import deque
import matplotlib.pyplot as plt
import joblib
from QATCH.common.logger import Logger as Log
WIN_SIZE = 128
TARGET_POIS = {1, 4, 5, 6}
LABEL_MAP = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4}
NUM_CLASSES = len(LABEL_MAP)
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
BACKGROUND_CLASS = 0
POI_CLASSES = set(range(1, NUM_CLASSES))
POI_SEQUENCE = [1, 4, 5, 6]


@staticmethod
def rebaseline(
    df: pd.DataFrame,
    window_size: int = WIN_SIZE,
    diss_col: str = "Dissipation",
    rf_col: str = "Resonance_Frequency"
) -> pd.DataFrame:
    base_d = df[diss_col].iloc[:window_size].mean()
    base_rf = df[rf_col].iloc[:window_size].mean()
    df[diss_col] = df[diss_col] - base_d
    df[rf_col] = -(df[rf_col] - base_rf)
    return df


class StopNetFeatureExtractor:
    def __init__(
        self,
        time_scaler,
        rf_scaler,
        diss_scaler,
        window_size: int = WIN_SIZE
    ):
        self.window_size = window_size
        self.scalers = {
            "Relative_time": time_scaler,
            "Dissipation": diss_scaler,
            "Resonance_Frequency": rf_scaler,
        }

    @classmethod
    def from_scaler_paths(
        cls,
        time_scaler_path: str,
        rf_scaler_path: str,
        diss_scaler_path: str,
        window_size: int = WIN_SIZE
    ) -> 'StopNetFeatureExtractor':
        time_scaler = joblib.load(time_scaler_path)
        rf_scaler = joblib.load(rf_scaler_path)
        diss_scaler = joblib.load(diss_scaler_path)
        return cls(time_scaler, rf_scaler, diss_scaler, window_size)

    def transform_window(
        self,
        win: pd.DataFrame,
        relative_time_poi1: float | None = None
    ) -> np.ndarray:
        """
        Transform a single window, adding a time_since_poi1 feature in seconds.

        Args:
            win: DataFrame for this window with a 'Relative_time' column.
            window_index: 0-based index of this window in the run.
            relative_time_poi1: timestamp (in same units as Relative_time) when POI1 occurred.
                                If None, time_since_poi1 will be zero.
        """
        # Capture original times before scaling
        raw_times = win["Relative_time"].values

        # 1) Scale signals
        df = win.copy()
        for col in ["Relative_time", "Dissipation", "Resonance_Frequency"]:
            df[col] = self.scalers[col].transform(df[[col]]).ravel()

        # 2) Derived per-sample features
        df["dissipation_change"] = df["Dissipation"].diff().fillna(0)
        df["rf_change"] = df["Resonance_Frequency"].diff().fillna(0)
        df["dissipation_change"] = np.clip(df["dissipation_change"], -5, 5)
        df["rf_change"] = np.clip(df["rf_change"], -5, 5)
        df["diss_x_rf"] = df["Dissipation"] * df["Resonance_Frequency"]
        df["change_prod"] = df["dissipation_change"] * df["rf_change"]

        # 3) Temporal position within window
        df["temporal_position"] = np.linspace(0, 1, len(df))

        # 4) Rolling statistics for stability
        df["diss_rolling_mean"] = (
            df["Dissipation"]
            .rolling(window=3, center=True)
            .mean()
            .fillna(df["Dissipation"])
        )
        df["rf_rolling_mean"] = (
            df["Resonance_Frequency"]
            .rolling(window=3, center=True)
            .mean()
            .fillna(df["Resonance_Frequency"])
        )

        # 5) time_since_poi1 in seconds (relative time)
        if relative_time_poi1 is None:
            df["time_since_poi1"] = 0.0
        else:
            delta = raw_times - relative_time_poi1
            df["time_since_poi1"] = np.clip(delta, 0.0, None)

        # Final column order
        cols = [
            "Relative_time", "Dissipation", "Resonance_Frequency",
            "dissipation_change", "rf_change", "diss_x_rf", "change_prod",
            "temporal_position", "diss_rolling_mean", "rf_rolling_mean",
            "time_since_poi1"
        ]
        return df[cols].to_numpy()


class StopNetTracker:
    def __init__(self, poi_sequence: List[int] = POI_SEQUENCE):
        self.poi_sequence = poi_sequence
        self.sequence_map = {poi: idx for idx, poi in enumerate(poi_sequence)}
        self.reset()

    def reset(self):
        self.current_stage = 0
        self.detected_pois = []
        self.missed_pois = []
        self.detection_history = []
        self.inference_history = []

    def get_expected_poi(self) -> Optional[int]:
        if self.current_stage < len(self.poi_sequence):
            return self.poi_sequence[self.current_stage]
        return None

    def get_allowed_pois(self) -> set:
        if self.current_stage < len(self.poi_sequence):
            allowed = {0, self.poi_sequence[self.current_stage]}
            for i in range(self.current_stage + 1, len(self.poi_sequence)):
                allowed.add(self.poi_sequence[i])

            return allowed
        return {0}

    def is_valid_prediction(self, predicted_poi: int) -> bool:
        if predicted_poi == 0:
            return True

        allowed_pois = self.get_allowed_pois()
        return predicted_poi in allowed_pois

    def _infer_missed_pois(self, detected_poi: int) -> List[int]:
        detected_stage = self.sequence_map[detected_poi]
        missed = []
        for stage in range(self.current_stage, detected_stage):
            missed_poi = self.poi_sequence[stage]
            missed.append(missed_poi)

        return missed

    def update_state(self, predicted_poi: int, confidence: float) -> bool:
        """Update tracker state with new prediction, handling backward inference"""
        self.detection_history.append((predicted_poi, confidence))
        if predicted_poi == 0:
            return False
        if not self.is_valid_prediction(predicted_poi):
            expected = self.get_expected_poi()
            print(
                f" Invalid POI {predicted_poi} rejected (expected: {expected})")
            return False

        detected_stage = self.sequence_map[predicted_poi]

        if detected_stage == self.current_stage:
            self.detected_pois.append(
                (predicted_poi, confidence, len(self.detection_history)))
            self.current_stage += 1
            print(
                f"Sequential POI {predicted_poi} detected (stage {self.current_stage}/{len(self.poi_sequence)})")
            return True

        elif detected_stage > self.current_stage:
            missed_pois = self._infer_missed_pois(predicted_poi)

            for missed_poi in missed_pois:
                self.missed_pois.append(
                    (missed_poi, 0.0, len(self.detection_history), "inferred"))
                print(
                    f"INFERRED: POI {missed_poi} was missed (due to detecting POI {predicted_poi})")

            self.inference_history.append({
                'detected_poi': predicted_poi,
                'inferred_missed': missed_pois,
                'window': len(self.detection_history),
                'confidence': confidence
            })

            self.detected_pois.append(
                (predicted_poi, confidence, len(self.detection_history)))

            self.current_stage = detected_stage + 1
            return True

        else:
            return False

    def get_progress(self) -> Tuple[int, int]:
        total_accounted = len(self.detected_pois) + len(self.missed_pois)
        return total_accounted, len(self.poi_sequence)

    def get_detailed_progress(self) -> Dict:
        return {
            'detected': len(self.detected_pois),
            'missed': len(self.missed_pois),
            'total': len(self.poi_sequence),
            'accounted': len(self.detected_pois) + len(self.missed_pois),
            'complete': self.is_complete()
        }

    def is_complete(self) -> bool:
        return self.current_stage >= len(self.poi_sequence)

    def get_final_sequence(self) -> List[Tuple[int, str, float]]:
        all_events = []
        for poi, conf, window in self.detected_pois:
            all_events.append((poi, 'detected', conf, window))
        for poi, conf, window, reason in self.missed_pois:
            all_events.append((poi, 'missed', conf, window))
        all_events.sort(key=lambda x: self.sequence_map[x[0]])

        return [(poi, status, conf) for poi, status, conf, window in all_events]


class StopNetPredictor:
    def __init__(self, model, confidence_threshold=0.65, sequence_bonus=0.1):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.sequence_bonus = sequence_bonus  # Bonus for expected POIs
        self.tracker = StopNetTracker()
        self.prediction_history = []

    def reset_for_new_run(self):
        self.tracker.reset()
        self.prediction_history = []

    def predict_with_sequential_constraint(self, features) -> Tuple[int, float, np.ndarray]:
        raw_probs = self.model.predict(features[np.newaxis, ...], verbose=0)[0]
        adjusted_probs = self._apply_sequential_constraints(raw_probs)
        predicted_class = np.argmax(adjusted_probs)
        confidence = adjusted_probs[predicted_class]
        predicted_poi = INV_LABEL_MAP[predicted_class]
        self.prediction_history.append({
            'raw_probs': raw_probs.copy(),
            'adjusted_probs': adjusted_probs.copy(),
            'predicted_class': predicted_class,
            'predicted_poi': predicted_poi,
            'confidence': confidence
        })
        return predicted_class, confidence, adjusted_probs

    def _apply_sequential_constraints(self, raw_probs: np.ndarray) -> np.ndarray:
        adjusted_probs = raw_probs.copy()
        allowed_pois = self.tracker.get_allowed_pois()
        expected_poi = self.tracker.get_expected_poi()

        for model_class in range(NUM_CLASSES):
            original_poi = INV_LABEL_MAP[model_class]
            if original_poi not in allowed_pois:
                adjusted_probs[model_class] = 0.0

        if expected_poi is not None:
            expected_class = LABEL_MAP[expected_poi]
            adjusted_probs[expected_class] *= (1.0 + self.sequence_bonus)

        prob_sum = np.sum(adjusted_probs)
        if prob_sum > 0:
            adjusted_probs = adjusted_probs / prob_sum
        else:
            for model_class in range(NUM_CLASSES):
                original_poi = INV_LABEL_MAP[model_class]
                if original_poi in allowed_pois:
                    adjusted_probs[model_class] = 1.0 / len(allowed_pois)

        return adjusted_probs

    def force_sequential_detection(self, window_position: int, total_windows: int) -> Tuple[Optional[int], Optional[float]]:
        """Enhanced force detection with backward inference capability"""
        expected_poi = self.tracker.get_expected_poi()

        if expected_poi is None:
            return None, None

        force_threshold = 0.75 - (0.1 * self.tracker.current_stage)

        if window_position > force_threshold * total_windows:
            recent_entries = self.prediction_history[-7:] if len(
                self.prediction_history) >= 3 else []

            expected_class = LABEL_MAP[expected_poi]
            max_expected_prob = 0
            for entry in recent_entries:
                prob = entry['raw_probs'][expected_class]
                if prob > max_expected_prob:
                    max_expected_prob = prob

            min_force_threshold = 0.15 - (0.02 * self.tracker.current_stage)

            if max_expected_prob > min_force_threshold:
                print(
                    f" Sequential force: POI {expected_poi} (prob: {max_expected_prob:.3f})")
                return LABEL_MAP[expected_poi], max_expected_prob

            if window_position > 0.85 * total_windows:
                remaining_pois = self.tracker.poi_sequence[self.tracker.current_stage:]

                best_later_poi = None
                best_later_prob = 0

                for poi in remaining_pois[1:]:
                    poi_class = LABEL_MAP[poi]
                    for entry in recent_entries:
                        prob = entry['raw_probs'][poi_class]
                        if prob > best_later_prob and prob > 0.1:
                            best_later_prob = prob
                            best_later_poi = poi

                if best_later_poi is not None:
                    return LABEL_MAP[best_later_poi], best_later_prob

        return None, None


class StopNet:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model=None,
        feature_extractor: Optional[StopNetFeatureExtractor] = None,
        time_scaler_path: Optional[str] = None,
        rf_scaler_path: Optional[str] = None,
        diss_scaler_path: Optional[str] = None,
        confidence_threshold: float = 0.65,
        sequence_bonus: float = 0.1
    ):
        if model is None:
            if not model_path:
                raise ValueError("Provide model_path or model instance.")
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            self.model = model

        if feature_extractor is None:
            if time_scaler_path and rf_scaler_path and diss_scaler_path:
                self.feature_extractor = StopNetFeatureExtractor.from_scaler_paths(
                    time_scaler_path, rf_scaler_path, diss_scaler_path, window_size=WIN_SIZE
                )
            else:
                raise ValueError(
                    "Provide a feature_extractor or all three scaler paths (time, rf, diss)."
                )
        else:
            self.feature_extractor = feature_extractor

        self.predictor = StopNetPredictor(
            self.model,
            confidence_threshold=confidence_threshold,
            sequence_bonus=sequence_bonus
        )
        self._buffer = deque(maxlen=WIN_SIZE)
        self._base_d: Optional[float] = None
        self._base_rf: Optional[float] = None
        self._t_poi = None

    def reset(self) -> None:
        """Clear buffer, reset POI state, and clear baseline offsets."""
        self._buffer.clear()
        self.predictor.reset_for_new_run()
        self._base_d = None
        self._base_rf = None
        self._t_poi = None

    def add_data_point(self, relative_time, dissipation, resonance_freq):
        self._buffer.append({
            'Relative_time':       relative_time,
            'Dissipation':         dissipation,
            'Resonance_Frequency': resonance_freq
        })

        if len(self._buffer) < WIN_SIZE:
            return None

        window_records = [self._buffer.popleft() for _ in range(WIN_SIZE)]
        window_df = pd.DataFrame(window_records)
        self._buffer.clear()
        if self._base_d is None:
            self._base_d = window_df['Dissipation'].mean()
            self._base_rf = window_df['Resonance_Frequency'].mean()
        window_df['Dissipation'] -= self._base_d
        window_df['Resonance_Frequency'] = -(
            window_df['Resonance_Frequency'] - self._base_rf
        )

        win_min, win_max = window_df['Relative_time'].min(
        ), window_df['Relative_time'].max()

        features = self.feature_extractor.transform_window(
            window_df, relative_time_poi1=self._t_poi
        )
        model_class, confidence, adjusted = (
            self.predictor.predict_with_sequential_constraint(features)
        )
        poi = INV_LABEL_MAP[model_class]

        if poi == 1 and self._t_poi is None:
            self._t_poi = window_df['Relative_time'].iloc[0]
        if poi != 0:
            self.predictor.tracker.update_state(poi, confidence)

        Log.i(
            f"Window [{win_min:.3f}, {win_max:.3f} (size={len(window_df)})] -> "
            f"class={model_class}, conf={confidence:.3f}, adjusted={adjusted}"
        )

        return poi, confidence, adjusted
