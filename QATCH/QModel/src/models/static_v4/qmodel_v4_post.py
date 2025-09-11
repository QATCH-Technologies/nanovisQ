import numpy as np
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
from QATCH.common.logger import Logger as Log
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from typing import List, Optional
PLOTTING = True


class V4PostProcess:
    @staticmethod
    def reasign(pois: dict, relative_time: np.ndarray):
        cand_4 = pois.get("POI4").get("indices", -1)
        cand_5 = pois.get("POI5").get("indices", -1)
        cand_6 = pois.get("POI6").get("indices", -1)
        if cand_4 == -1 or cand_5 == -1 or cand_6 == -1:
            return pois
        if max(relative_time) < 90:
            relative_time = np.arange(len(relative_time))
        max_4 = relative_time[cand_4[0]]
        max_6 = relative_time[cand_6[0]]

        first_split = int((max_6 - max_4) * 0.25) + max_4
        second_split = int((max_6 - max_4) * 0.75) + max_4

        all_cand = []
        all_cand.extend(cand_4)
        all_cand.extend(cand_5)
        all_cand.extend(cand_6)
        set_4 = []
        set_5 = []
        set_6 = []
        for c in all_cand:
            if relative_time[c] <= first_split:
                set_4.append(c)
            elif relative_time[c] >= second_split:
                set_6.append(c)
            else:
                set_5.append(c)
        pois["POI4"]["indices"] = set_4
        pois["POI5"]["indices"] = set_5
        pois["POI6"]["indices"] = set_6
        return pois

    @staticmethod
    def density_select(poi_indices: List[int], relative_time: np.ndarray,
                       confidences: List[float] = None,
                       eps: float = 0.5, min_samples: int = 2) -> Optional[int]:
        """
        Simplified version when confidence scores are provided directly.

        Args:
            poi_indices: List of POI indices to cluster
            relative_time: Array of relative times for each index
            confidences: List of confidence scores corresponding to poi_indices
            eps: Maximum distance between two samples for clustering (in time units)
            min_samples: Minimum number of samples in a neighborhood for clustering

        Returns:
            Index of the highest confidence POI in the largest cluster
        """
        if not poi_indices:
            return None

        if len(poi_indices) == 1:
            return poi_indices[0]

        # Extract time values for the given indices
        time_values = relative_time[poi_indices].reshape(-1, 1)

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(time_values)
        labels = clustering.labels_

        # Find the largest cluster
        unique_labels, counts = np.unique(
            labels[labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            # No clusters found, return highest confidence point
            if confidences:
                return poi_indices[np.argmax(confidences)]
            return poi_indices[0]

        # Get the largest cluster
        largest_cluster_label = unique_labels[np.argmax(counts)]
        cluster_mask = labels == largest_cluster_label

        # Get highest confidence index from largest cluster
        if confidences:
            cluster_confidences = [confidences[i]
                                   for i, mask in enumerate(cluster_mask) if mask]
            cluster_indices = [poi_indices[i]
                               for i, mask in enumerate(cluster_mask) if mask]
            return cluster_indices[np.argmax(cluster_confidences)]
        else:
            # No confidence scores, return first point in largest cluster
            for i, mask in enumerate(cluster_mask):
                if mask:
                    return poi_indices[i]

        return None

    @staticmethod
    def poi_1(indices: list[int],
              confidences: list[float],
              feature_vector: pd.DataFrame,
              relative_time: np.ndarray
              ) -> tuple[list[int], list[float]]:
        dissipation = feature_vector['Dissipation'].values
        resonance = feature_vector['Resonance_Frequency'].values
        diff_smooth = feature_vector['Difference'].values
        svm_scores = {
            col: feature_vector[col].values
            for col in feature_vector.columns
            if col.endswith('_DoG_SVM_Score')
        }
        orig_inds = list(indices)
        orig_confs = list(confidences)

        def pick_after(cands, times, t):
            return [i for i in cands if times[i] > t]
        grad_d = np.gradient(dissipation, relative_time)
        thr_d = grad_d.mean() + grad_d.std()
        sh_d = np.where(grad_d > thr_d)[0]
        if len(sh_d):
            t1 = relative_time[sh_d[0]]
            inds = pick_after(orig_inds, relative_time, t1)
        else:
            inds = orig_inds
        grad_r = np.gradient(resonance, relative_time)
        thr_r = grad_r.mean() - grad_r.std()
        sh_r = np.where(grad_r < thr_r)[0]
        if len(sh_r):
            t2 = relative_time[sh_r[0]]
            inds = pick_after(inds, relative_time, t2)
        grad_diff = np.gradient(diff_smooth, relative_time)
        thr_diff = grad_diff.mean() + grad_diff.std()
        sh_diff = np.where(grad_diff > thr_diff)[0]
        if len(sh_diff):
            t_diff = relative_time[sh_diff[0]]
            avg_interval = np.mean(np.diff(sorted(relative_time[orig_inds])))
            tol = avg_interval / 2
            aligned = [i for i in inds if abs(
                relative_time[i] - t_diff) <= tol]
            inds = aligned or inds
        if not inds:
            for scores in svm_scores.values():
                jumps = np.abs(np.diff(scores))
                thr_j = jumps.mean() + jumps.std()
                big = np.where(jumps > thr_j)[0]
                if len(big):
                    t3 = relative_time[big[0] + 1]
                    inds = pick_after(orig_inds, relative_time, t3)
                    if inds:
                        break
        inds = orig_inds.copy()

        # HEIGHT FILTER
        baseline_window = int(0.015*len(dissipation))
        base_vals = dissipation[int(0.005*len(dissipation)):baseline_window]
        height_thresh = base_vals.mean() + 2 * base_vals.std()
        height_filtered = [i for i in inds if dissipation[i] > height_thresh]
        inds_after_height = height_filtered or inds.copy()

        # PEAK-JUMP FILTER
        delta_d = np.diff(dissipation)
        thr_jump = delta_d.mean() + delta_d.std()
        jump_idxs = np.where(delta_d > thr_jump)[0] + 1

        if len(jump_idxs):
            injection_idx = jump_idxs[0]
        else:
            # fallback to global max if nothing exceeds thr_jump
            injection_idx = np.argmax(delta_d) + 1

        jump_filtered = [i for i in inds_after_height if i >= injection_idx]
        inds_after_jump = jump_filtered or inds_after_height.copy()

        # NO-INCLINE FILTER
        flat_filtered = [i for i in inds_after_jump if grad_d[i] <= thr_d]
        final_inds = flat_filtered or inds_after_jump.copy()

        if PLOTTING:
            plt.figure(figsize=(12, 6))
            plt.plot(relative_time, dissipation, color='black',
                     lw=1.2, label='Dissipation')
            plt.axhline(height_thresh, color='gray',
                        ls='--', label='height_thresh')
            plt.axvline(relative_time[injection_idx], color='blue', ls='--',
                        label=f'biggest jump @ {relative_time[injection_idx]:.2f}s')

            # plot each stage
            def scatter(idxs, marker, label, color):
                plt.scatter(relative_time[idxs], dissipation[idxs],
                            marker=marker, s=100, edgecolors=color,
                            facecolors='none', label=label)

            scatter(orig_inds,         'x', 'orig_inds',   'red')
            scatter(inds_after_height, 'o', 'after height', 'orange')
            scatter(inds_after_jump,   's', 'after jump',   'green')
            scatter(final_inds,        'D', 'after slope',  'blue')

            plt.xlabel("Time (s)")
            plt.ylabel("Dissipation")
            plt.title("Filter Debug - dissipation & candidates at each stage")
            plt.legend(loc="upper left", fontsize="small", ncol=2)
            plt.tight_layout()
            plt.show()

        final_confs = [orig_confs[orig_inds.index(i)] for i in final_inds]

        return final_inds[0]

    @staticmethod
    def poi_2():
        pass

    @staticmethod
    def poi_4(x, y, initial_guess, search_window=150, window_size=30, smooth_sigma=5):
        pass

    @staticmethod
    def poi_6(
        indices: list[int],
        confidences: list[float],
        feature_vector: pd.DataFrame,
        relative_time: np.ndarray,
        poi5_idx: int,
    ) -> tuple[list[int], list[float]]:
        """
        POI6: choose the best candidate after poi5_idx by equally weighting
        rough envelopes of all three raw curves plus the three DoG_SVM scores,
        with a debug plot of normalized envelopes and candidates.
        """
        # filter out-of-bounds
        filtered = [(i, c)
                    for i, c in zip(indices, confidences) if i > poi5_idx]
        if not filtered:
            return indices, confidences
        cand_idxs, cand_confs = zip(*filtered)
        cand_idxs = list(cand_idxs)
        cand_confs = list(cand_confs)
        # raw values & DoG_SVM scores
        diss_vals = feature_vector['Dissipation'].values
        rf_vals = feature_vector['Resonance_Frequency'].values
        diff_vals = feature_vector['Difference'].values

        diss_score = feature_vector['Dissipation_DoG_SVM_Score'].values
        rf_score = feature_vector['Resonance_Frequency_DoG_SVM_Score'].values
        diff_score = feature_vector['Difference_DoG_SVM_Score'].values

        # build rough "envelopes" by thresholding the gradients
        d_diss = np.gradient(diss_vals,  relative_time)
        d_rf = np.gradient(rf_vals,    relative_time)
        d_diff = np.gradient(diff_vals,  relative_time)

        def make_env(d):
            thr = np.std(d)
            return np.where(np.abs(d) >= thr, d, 0)

        diss_env = make_env(d_diss)
        rf_env = make_env(d_rf)
        diff_env = make_env(d_diff)

        # min–max normalize
        def _min_max(x: np.ndarray) -> np.ndarray:
            xmin, xmax = x.min(), x.max()
            return (x - xmin) / (xmax - xmin) if xmax != xmin else x

        norm_diss = _min_max(diss_env)
        norm_rf = _min_max(rf_env)
        norm_diff = _min_max(diff_env)

        # score each candidate
        scores = []
        for idx in cand_idxs:
            dog_score = abs(diss_score[idx]) + \
                abs(rf_score[idx]) + abs(diff_score[idx])
            score = (
                norm_diss[idx]      # + for big diss slope
                + (-norm_rf[idx])   # + for big negative RF slope
                + (-norm_diff[idx])  # + for big downward diff slope
                + dog_score
            )
            scores.append(score)

            # pick best
            best_pos = int(np.argmax(scores))
            best_idx = cand_idxs[best_pos]
            best_conf = cand_confs[best_pos]
            best_score = scores[best_pos]

            idxs = list(cand_idxs)
            times_cand = relative_time[idxs]

            # normalize scores for marker sizing using np.ptp
            score_arr = np.array(scores)
            min_s, max_s = 30, 150
            norm_sz = (score_arr - score_arr.min()) / \
                (np.ptp(score_arr) + 1e-6)
            sizes = norm_sz * (max_s - min_s) + min_s

            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

            # plot envelopes
            ax.plot(relative_time, norm_diss,
                    label='Norm Diss Env', linewidth=1)
            ax.plot(relative_time, norm_rf,
                    label='Norm RF Env',   linewidth=1)
            ax.plot(relative_time, norm_diff,
                    label='Norm Diff Env', linewidth=1)

            # scatter candidates sized by score
            ax.scatter(times_cand, norm_diss[idxs],
                       s=sizes, marker='o', label='Diss Candidates')
            ax.scatter(times_cand, norm_rf[idxs],
                       s=sizes, marker='s', label='RF Candidates')
            ax.scatter(times_cand, norm_diff[idxs],
                       s=sizes, marker='^', label='Diff Candidates')

            # annotate each with its numeric score
            for idx, score in zip(idxs, scores):
                y_env = max(norm_diss[idx], norm_rf[idx], norm_diff[idx])
                ax.text(
                    relative_time[idx],
                    y_env + 0.03,
                    f"{score:.1f}",
                    ha='center', va='bottom',
                    fontsize='x-small'
                )

            # highlight the selected candidate
            best_time = relative_time[best_idx]
            ax.axvline(best_time, color='red', linestyle='--',
                       label='Selected Candidate')
            ax.scatter(best_time, norm_diss[best_idx],
                       s=200, marker='*', color='red')
            ax.scatter(best_time, norm_rf[best_idx],
                       s=200, marker='*', color='red')
            ax.scatter(best_time, norm_diff[best_idx],
                       s=200, marker='*', color='red')

            # labels & legend
            ax.set_xlabel("Relative Time (s)")
            ax.set_ylabel("Normalized Envelope Value")
            ax.set_title("POI6 Debug: Envelopes, Scores & Selection")
            ax.legend(loc='upper right', fontsize='small')
            plt.tight_layout()
            plt.show()
            return best_idx
