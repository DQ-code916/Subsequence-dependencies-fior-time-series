"""
Rule-corrected imputation experiment 

Prerequisites (in order):

1. Complete and missing time-series CSVs
   - data/<dataname>.csv                          ground-truth series
   - data/<dataname>_missing.csv    same shape, NaN = missing
     (create missing data beforehand, e.g. random masking scripts)

2. Rule mining artifacts (motif_discovery + rule_discovery, task=1)
   - results/obj/<dataname>.pkl
   - results/valid_rules/<dataname>.pkl

3. Baseline imputation outputs
   - results/Arima/Arima_<dataname>.csv   optional;
   - results/rnn/rnn_<dataname>.csv
   - results/Timer/Timer_<dataname>.csv
   (typically produced via data_preparation_new.py or your DL imputation pipeline)

Pipeline per (method, SubD): load baseline fill -> SubD rule imputation plan on TEST
-> Correct_filled (replace baseline values outside rule band) -> RMSE on missing cells
(baseline vs rule-corrected). Saves summary to results/imputation_results/<dataname>.csv.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    Calc_cdd_rhs_score,
    fill_missing_with_arima,
    load_instance,
    predict_rhs_from_lhs_combination,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)


def _missing_csv_path(dataname: str, project_root: Optional[str] = None) -> str:
    """Path to data/<dataname>_missing.csv."""
    root = project_root or _ROOT
    return os.path.join(root, "data", f"{dataname}_missing.csv")


class Candidate_imputation:
    def __init__(self, c_id, t_2, delta_2, lambda_2, S_pred, avg_err, epsilon):
        self.c_id = c_id
        self.t_2 = t_2
        self.delta_2 = delta_2
        self.lambda_2 = lambda_2
        self.S_pred = S_pred
        self.avg_err = avg_err
        self.epsilon = epsilon
        self.score = -1.0


class Actual_imputation:
    """Rule fill on [Range[0], Range[1]): imputed_values[i] is the reference at offset i."""

    def __init__(self, Range, epsilon, imputed_values):
        self.Range = Range
        self.epsilon = epsilon
        self.imputed_values = imputed_values


def get_missing_segments_by_col(missing_db: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Continuous NaN runs per column; each segment is half-open [start, end)."""
    n_row, n_col = missing_db.shape
    out: List[List[Tuple[int, int]]] = []
    for col in range(n_col):
        segs: List[Tuple[int, int]] = []
        i = 0
        while i < n_row:
            if np.isnan(missing_db[i, col]):
                start = i
                while i < n_row and np.isnan(missing_db[i, col]):
                    i += 1
                segs.append((start, i))
            else:
                i += 1
        out.append(segs)
    return out


def imputation_with_candidates2(
    vr,
    md,
    missing_db: np.ndarray,
    ori_db: np.ndarray,
    c_skip: Optional[set] = None,
) -> Tuple[Dict[int, List[Actual_imputation]], np.ndarray, List[List[Tuple[int, int]]]]:
    """
    SubD: for each missing segment, pick best RHS rule candidate and fill iteratively.
    Returns (actual_imputation, filled_db_by_rule, original_missing_segments).
    """
    _ = c_skip  # reserved; kept for API compatibility
    candidate_imputations: Dict[int, List[Candidate_imputation]] = {}
    actual_imputation: Dict[int, List[Actual_imputation]] = {}

    filled_db = np.array(missing_db, dtype=np.float64, copy=True)
    original_missing_segments = get_missing_segments_by_col(missing_db)
    n_row, n_col = filled_db.shape

    rules_by_col: List[List[Any]] = [[] for _ in range(n_col)]
    for rhs, rule_list in vr.items():
        c_id_rhs = rhs[0]
        if 0 <= c_id_rhs < n_col:
            rules_by_col[c_id_rhs].extend(rule_list)

    for c_id in range(n_col):
        candidate_imputations[c_id] = []
        rules = rules_by_col[c_id]
        if not rules:
            continue
        for rule in rules:
            if rule.RHS[0] != c_id:
                continue
            for lhs_combination in rule.lhs_combinations:
                valid, t_2, delta_2, lambda_2, S_pred = predict_rhs_from_lhs_combination(
                    md, rule, lhs_combination, filled_db[:, c_id], is_only_motif=True
                )
                if not valid or S_pred is None:
                    continue
                candidate_imputations[c_id].append(
                    Candidate_imputation(
                        c_id, t_2, delta_2, lambda_2, S_pred,
                        avg_err=rule.avg_err, epsilon=rule.epsilon,
                    )
                )

    for c_id in range(len(original_missing_segments)):
        db_rhs = filled_db[:, c_id]
        segments = list(original_missing_segments[c_id])
        for seg_i in range(len(segments)):
            segment = list(segments[seg_i])
            while segment[1] - segment[0] > 0:
                prev_segment = tuple(segment)
                for cdd_imp in candidate_imputations[c_id]:
                    cdd_imp.score = Calc_cdd_rhs_score(cdd_imp, tuple(segment))
                best_list = [c for c in candidate_imputations[c_id] if getattr(c, "score", 0.0) > 0]
                if not best_list:
                    break
                cdd_imp_best = max(best_list, key=lambda x: x.score)

                if cdd_imp_best.t_2 < segment[0]:
                    int_start = segment[0]
                    int_end = min(segment[1], cdd_imp_best.t_2 + cdd_imp_best.delta_2)
                    S_pred = cdd_imp_best.S_pred + filled_db[cdd_imp_best.t_2, c_id]
                    fill_values = S_pred[int_start - cdd_imp_best.t_2 : int_end - cdd_imp_best.t_2]
                    filled_db[int_start:int_end, c_id] = fill_values
                else:
                    int_start = segment[0]
                    int_end = min(segment[1], segment[0] + cdd_imp_best.delta_2)
                    S_slice = cdd_imp_best.S_pred[0 : min(segment[1] - segment[0], cdd_imp_best.delta_2)]
                    if segment[0] <= 0 or np.isnan(db_rhs[segment[0] - 1]):
                        break
                    base = db_rhs[segment[0] - 1]
                    fill_values = S_slice + base
                    filled_db[int_start:int_end, c_id] = fill_values

                if c_id not in actual_imputation:
                    actual_imputation[c_id] = []
                actual_imputation[c_id].append(
                    Actual_imputation(
                        Range=(int_start, int_end),
                        epsilon=getattr(cdd_imp_best, "epsilon", None),
                        imputed_values=np.asarray(fill_values, dtype=np.float64).copy(),
                    )
                )
                if int_end >= segment[1]:
                    break
                segment = [int_end, segment[1]]
                if tuple(segment) == prev_segment:
                    break
            segments[seg_i] = tuple(segment)

    return actual_imputation, filled_db, original_missing_segments


def correct_filled(
    filled_db: np.ndarray,
    original_missing_segments: Sequence[Sequence[Tuple[int, int]]],
    actual_imputation: Dict[int, List[Actual_imputation]],
    *,
    epsilon_override: float = 0.01,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """
    For each missing segment, if baseline fill falls outside [rule_value ± eps],
    replace with rule imputed_values.
    """
    out = np.asarray(filled_db, dtype=np.float64, copy=True)
    n_row, n_col = out.shape
    corrected_segments: List[List[Tuple[int, int]]] = [[] for _ in range(n_col)]

    for c_id in range(min(len(original_missing_segments), n_col)):
        act_list = actual_imputation.get(c_id, [])
        if not act_list:
            continue
        for seg_start, seg_end in original_missing_segments[c_id]:
            for act in act_list:
                r0, r1 = act.Range[0], act.Range[1]
                int_start = max(seg_start, r0, 0)
                int_end = min(seg_end, r1, n_row)
                if int_end <= int_start:
                    continue
                eps = epsilon_override if act.epsilon is None else act.epsilon
                if eps is None or (isinstance(eps, (int, float)) and (np.isnan(eps) or eps < 0)):
                    continue
                iv = np.asarray(act.imputed_values, dtype=np.float64)
                rel_start = int_start - r0
                rel_end = int_end - r0
                if rel_start < 0 or rel_end > len(iv):
                    continue
                imputed_slice = iv[rel_start:rel_end]
                cur = out[int_start:int_end, c_id]
                min_len = min(len(imputed_slice), len(cur))
                if min_len <= 0:
                    continue
                if len(imputed_slice) != len(cur):
                    imputed_slice = imputed_slice[:min_len]
                    cur = cur[:min_len]
                low = imputed_slice - float(eps)
                high = imputed_slice + float(eps)
                mask = ~np.isnan(cur)
                out_mask = mask & ((cur < low) | (cur > high))
                if not np.any(out_mask):
                    continue
                cur = np.where(out_mask, imputed_slice, cur)
                out[int_start : int_start + min_len, c_id] = cur
                corrected_segments[c_id].append((int_start, int_start + min_len))

    return out, corrected_segments


def load_imputed_db_from_csv(
    dataname: str,
    imputation_type: str,
    extra: Optional[str] = None,
    *,
    project_root: Optional[str] = None,
) -> np.ndarray:
    root = project_root or _ROOT
    if extra is None:
        csv_path = os.path.join(root, "results", imputation_type, f"{imputation_type}_{dataname}.csv")
    else:
        csv_path = os.path.join(root, "results", imputation_type, f"{extra}_{dataname}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing baseline imputation CSV: {csv_path}")
    return pd.read_csv(csv_path).values


def evaluate_imputation_rmse(
    ori_db: np.ndarray,
    filled_baseline: np.ndarray,
    filled_final: np.ndarray,
    original_missing_segments: Sequence[Sequence[Tuple[int, int]]],
) -> Dict[str, Optional[float]]:
    """Normalized RMSE on originally missing cells: baseline vs rule-corrected."""
    ori_db = np.asarray(ori_db, dtype=np.float64)
    filled_baseline = np.asarray(filled_baseline, dtype=np.float64)
    filled_final = np.asarray(filled_final, dtype=np.float64)
    n_row = min(ori_db.shape[0], filled_baseline.shape[0], filled_final.shape[0])
    n_col = ori_db.shape[1]
    ori_db = ori_db[:n_row]
    filled_baseline = filled_baseline[:n_row]
    filled_final = filled_final[:n_row]

    norm_baseline: List[float] = []
    norm_final: List[float] = []

    for j in range(n_col):
        mask_missing = np.zeros(n_row, dtype=bool)
        if j < len(original_missing_segments):
            for s, e in original_missing_segments[j]:
                s0, e0 = max(0, s), min(n_row, e)
                if e0 > s0:
                    mask_missing[s0:e0] = True
        n_miss = int(np.sum(mask_missing))
        if n_miss == 0:
            continue
        ori_j = ori_db[:, j]
        rmse_base = float(np.sqrt(np.nanmean((filled_baseline[mask_missing, j] - ori_j[mask_missing]) ** 2)))
        rmse_fin = float(np.sqrt(np.nanmean((filled_final[mask_missing, j] - ori_j[mask_missing]) ** 2)))
        valid_ori = ori_j[~np.isnan(ori_j)]
        if valid_ori.size == 0:
            continue
        Mj = float(np.nanmax(valid_ori) - np.nanmin(valid_ori))
        if Mj > 0:
            norm_baseline.append(rmse_base / Mj)
            norm_final.append(rmse_fin / Mj)

    overall_base = float(np.mean(norm_baseline)) if norm_baseline else None
    overall_final = float(np.mean(norm_final)) if norm_final else None
    print("Overall normalized RMSE (rmse / range) over attributes:")
    print(f"  baseline       : {overall_base:.6f}" if overall_base is not None else "  baseline       : None")
    print(f"  rule-corrected : {overall_final:.6f}" if overall_final is not None else "  rule-corrected : None")
    return {"baseline": overall_base, "final": overall_final}


def run_subd_imputation_fusion(
    dataname: str,
    *,
    imputation_type: str = "Arima",
    c_skip: Optional[set] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    One run: baseline imputation + SubD rule correction + RMSE.
    Returns {"baseline": ..., "final": ...} (keys also exposed as "arima"/"final" for compatibility).
    """
    root = project_root or _ROOT
    ori_path = os.path.join(root, "data", f"{dataname}.csv")
    missing_path = _missing_csv_path(dataname, project_root=root)
    md_path = os.path.join(root, "results", "obj", f"{dataname}.pkl")
    vr_path = os.path.join(root, "results", "valid_rules", f"{dataname}.pkl")

    print(" 0.Loading md, vr ...")
    md = load_instance(md_path)
    vr = load_instance(vr_path)
    n_rules = sum(len(vr[rhs]) for rhs in vr)
    print(f"  rules loaded: {n_rules} rules for {len(vr)} RHS.")

    print(" 1.Loading data ...")
    ori_df = pd.read_csv(ori_path)
    ori_db = ori_df.values
    missing_db = pd.read_csv(missing_path).values
    columns = list(ori_df.columns)
    print(f"  shape {missing_db.shape[0]} x {missing_db.shape[1]}, missing cells: {np.isnan(missing_db).sum()}.")

    mtype = imputation_type.strip()
    if mtype.lower() == "arima":
        arima_csv = os.path.join(root, "results", "Arima", f"Arima_{dataname}.csv")
        if os.path.isfile(arima_csv):
            filled_db = pd.read_csv(arima_csv).values
            print(f"  Loaded cached Arima from: {arima_csv}")
        else:
            filled_db = fill_missing_with_arima(missing_db)
            os.makedirs(os.path.dirname(arima_csv), exist_ok=True)
            pd.DataFrame(filled_db, columns=columns).to_csv(arima_csv, index=False, encoding="utf-8")
            print(f"  Saved Arima to: {arima_csv}")
    elif mtype in ("rnn", "Timer"):
        filled_db = load_imputed_db_from_csv(dataname, mtype, project_root=root)
    else:
        raise ValueError(f"Unsupported imputation_type: {imputation_type!r}")

    filled_baseline = filled_db.copy()
    print(f"  Finish baseline imputation by {mtype}.")

    actual_imputation, _filled_by_rule, original_missing_segments = imputation_with_candidates2(
        vr, md, missing_db, ori_db, c_skip=c_skip or set()
    )
    print("  Finish SubD rule imputation plan.")

    if mtype == "rnn":
        try:
            filled_for_correct = load_imputed_db_from_csv(
                dataname, "rnn", "rnn_SubD", project_root=root
            )
        except FileNotFoundError:
            filled_for_correct = filled_baseline
    else:
        filled_for_correct = filled_baseline

    final_db, _corrected = correct_filled(
        filled_for_correct, original_missing_segments, actual_imputation
    )
    print("  Finish rule correction.")

    overall = evaluate_imputation_rmse(
        ori_db, filled_baseline, final_db, original_missing_segments
    )
    return {
        "baseline": overall["baseline"],
        "final": overall["final"],
        "arima": overall["baseline"],
    }


def run_imputation_grid_experiment(
    dataname: str,
    *,
    method_set: Optional[Sequence[str]] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """ each baseline method x SubD; save results/imputation_results/<dataname>.csv."""
    methods = list(method_set or ["Arima", "rnn", "Timer"])
    result_dict: Dict[str, Optional[float]] = {}

    for m in methods:
        overall = run_subd_imputation_fusion(
            dataname,
            imputation_type=m,
            project_root=project_root,
        )
        result_dict[m] = overall["baseline"]
        result_dict[f"{m}SubD"] = overall["final"]

    for k, v in result_dict.items():
        print(f"{k} : {v}")

    root = project_root or _ROOT
    save_dir = os.path.join(root, "results", "imputation_results")
    os.makedirs(save_dir, exist_ok=True)
    save_csv = os.path.join(save_dir, f"{dataname}.csv")
    pd.DataFrame(
        [{"key": k, "rmse": v} for k, v in result_dict.items()],
        columns=["key", "rmse"],
    ).to_csv(save_csv, index=False, encoding="utf-8")
    print(f"Saved imputation results to: {save_csv}")
    return result_dict


if __name__ == "__main__":
    run_imputation_grid_experiment(
        dataname="exchange_rate",
    )
