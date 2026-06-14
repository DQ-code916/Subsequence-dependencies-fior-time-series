"""
Rule + deep-learning union fusion for anomaly detection

Prerequisites (in order):

1. Test data and sample-level ground truth
   - data/<dataname>/<dataname>_TRAIN.csv   (column layout reference for TEST)
   - data/<dataname>/<dataname>_TEST.csv    (multivariate series to evaluate)
   - data/<dataname>/label.csv              (sample-level labels; >0 = anomaly)

2. Rule mining artifacts (motif_discovery / rule_discovery task=1)
   - results/obj/<rule_source>.pkl
   - results/valid_rules/<rule_source>.pkl
   rule_source defaults to dataname; may differ for cross-dataset rule transfer.

3. Deep-learning baseline predictions (train DL detectors separately, then export)
   - results/detection/<dataname>/<method>.csv
   CSV must contain a y_pred column, one row per TEST time step (same length as label.csv).


Pipeline: RHS rule-forecast error detection on TEST -> fuse with DL y_pred (point-wise union) ->
report DL-only and rule-union-DL F1, Precision, Recall.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _detection_csv_coerce_numeric_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align with MotifDiscovery.read_data: drop idx, coerce columns to numeric, fill NaN with 0.
    Do not drop unparseable columns so attribute indices stay consistent with rule LHS/RHS.
    """
    out = df.copy()
    if "idx" in out.columns:
        out = out.drop(columns=["idx"])
    if out.shape[1] == 0:
        raise ValueError("CSV has no feature columns after dropping idx.")
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _sample_label_csv_column(df: pd.DataFrame) -> str:
    """Pick the sample-level label column from label.csv."""
    for name in ("label", "y_true", "y", "anomaly"):
        if name in df.columns:
            return name
    skip = {"idx", "index", "time"}
    for c in df.columns:
        if str(c).lower() not in skip:
            return str(c)
    raise ValueError(
        "label.csv: no usable label column (expected label / y_true / y / anomaly or another numeric column)."
    )


def _load_test_aligned_to_train(
    dataname: str, data_root: Optional[str] = None
) -> np.ndarray:
    """Load {dataname}_TEST.csv with columns aligned to {dataname}_TRAIN.csv."""
    root = data_root or os.path.join(_ROOT, "data")
    dn = os.path.join(root, dataname)
    p_tr = os.path.join(dn, f"{dataname}_TRAIN.csv")
    p_te = os.path.join(dn, f"{dataname}_TEST.csv")
    if not os.path.isfile(p_tr):
        raise FileNotFoundError(f"Missing TRAIN (for column alignment): {p_tr}")
    if not os.path.isfile(p_te):
        raise FileNotFoundError(f"Missing TEST: {p_te}")
    tr_df = _detection_csv_coerce_numeric_fill(pd.read_csv(p_tr))
    te_df = _detection_csv_coerce_numeric_fill(pd.read_csv(p_te))
    missing_te = [c for c in tr_df.columns if c not in te_df.columns]
    if missing_te:
        raise ValueError(f"{dataname}: TEST missing columns present in TRAIN: {missing_te}")
    extra_te = [c for c in te_df.columns if c not in tr_df.columns]
    if extra_te:
        te_df = te_df.drop(columns=extra_te)
    te_df = te_df[[c for c in tr_df.columns]]
    return te_df.to_numpy(dtype=np.float64)


def _load_sample_label_binary(
    dataname: str, n_points: int, data_root: Optional[str] = None
) -> np.ndarray:
    """Load label.csv sample-level ground truth (>0 treated as anomaly)."""
    root = data_root or os.path.join(_ROOT, "data")
    p_lb = os.path.join(root, dataname, "label.csv")
    if not os.path.isfile(p_lb):
        raise FileNotFoundError(f"Missing label file: {p_lb}")
    df = pd.read_csv(p_lb)
    col_lab = _sample_label_csv_column(df)
    y = pd.to_numeric(df[col_lab], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y_bin = (y > 0).astype(np.int64)
    if y_bin.shape[0] != n_points:
        raise ValueError(
            f"label.csv row count {y_bin.shape[0]} != TEST row count {n_points}."
        )
    return y_bin


def _binary_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Point-level binary precision, recall, F1, and tp/fp/fn."""
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true/y_pred length mismatch: {yt.shape[0]} vs {yp.shape[0]}")
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def _print_binary_metrics_line(prefix: str, tag: str, m: Dict[str, Any]) -> None:
    print(
        f"{prefix} {tag}  F1={float(m['f1']):.6f}  "
        f"Precision={float(m['precision']):.6f}  Recall={float(m['recall']):.6f}  "
        f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})"
    )


def _load_task1_rule_assets(rule_source_name: str) -> Tuple[Any, Dict[Any, List[Any]]]:
    from utils import load_instance

    p_md = os.path.join(_ROOT, "results", "obj", f"{rule_source_name}.pkl")
    p_rules = os.path.join(_ROOT, "results", "valid_rules", f"{rule_source_name}.pkl")
    if not os.path.isfile(p_md):
        raise FileNotFoundError(f"Missing task1 md artifact: {p_md}")
    if not os.path.isfile(p_rules):
        raise FileNotFoundError(f"Missing task1 valid_rules artifact: {p_rules}")
    md = load_instance(p_md)
    valid_rules = load_instance(p_rules)
    if not isinstance(valid_rules, dict):
        raise ValueError(f"valid_rules must be a dict: {p_rules}")
    return md, valid_rules


def _rhs_col_from_rule_key(k: Any) -> Optional[int]:
    if isinstance(k, tuple) and len(k) >= 1:
        try:
            return int(k[0])
        except Exception:
            return None
    if isinstance(k, (int, np.integer)):
        return int(k)
    return None


def _positive_int_topk(topk: Any) -> Optional[int]:
    """Return positive int topk, or None to use all rules. bool is not treated as int."""
    if topk is None or isinstance(topk, bool):
        return None
    try:
        k = int(topk)
    except (TypeError, ValueError):
        return None
    return k if k > 0 else None


def _merge_half_open_intervals(ivs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = [(int(a), int(b)) for a, b in ivs if int(b) > int(a)]
    if not cleaned:
        return []
    cleaned.sort(key=lambda t: (t[0], t[1]))
    out: List[Tuple[int, int]] = []
    cs, ce = cleaned[0]
    for a, b in cleaned[1:]:
        if a <= ce:
            ce = max(ce, b)
        else:
            out.append((cs, ce))
            cs, ce = a, b
    out.append((cs, ce))
    return out


def _subtract_half_open_interval(
    lo: int, hi: int, blocked: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Maximal sub-intervals of [lo, hi) not covered by merged blocked intervals."""
    if hi <= lo:
        return []
    bm = _merge_half_open_intervals(blocked)
    cur = lo
    free: List[Tuple[int, int]] = []
    for x, y in bm:
        if y <= cur:
            continue
        if x >= hi:
            break
        if cur < x:
            free.append((cur, min(x, hi)))
        cur = max(cur, y)
        if cur >= hi:
            break
    if cur < hi:
        free.append((cur, hi))
    return [(u, v) for u, v in free if v > u]


def _collect_rhs_ordered_forecast_segments(
    *,
    md: Any,
    valid_rules: Dict[Any, List[Any]],
    x: np.ndarray,
    topk: Any = None,
    rhs_cols: Optional[List[int]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Per RHS column: aggregate rules, sort by avg_err, predict_rhs_from_lhs_combination,
    de-overlap half-open segments. Returns n_feats lists of segment dicts
    (t_start / t_end / predicted / rule_sort_rank).
    """
    from utils import predict_rhs_from_lhs_combination

    arr = np.asarray(x, dtype=np.float64)
    n_points, n_feats = arr.shape

    if rhs_cols is not None:
        cols_loop = sorted({int(c) for c in rhs_cols})
        if not cols_loop:
            raise ValueError("rhs_cols is empty; use None for all attributes.")
        for c in cols_loop:
            if c < 0 or c >= n_feats:
                raise ValueError(
                    f"rhs_cols index {c} out of range 0..{n_feats - 1} (n_feats={n_feats})."
                )
    else:
        cols_loop = list(range(n_feats))

    rules_by_col: List[List[Any]] = [[] for _ in range(n_feats)]
    for key, rule_list in valid_rules.items():
        rhs_c = _rhs_col_from_rule_key(key)
        if rhs_c is None or rhs_c < 0 or rhs_c >= n_feats:
            continue
        rules_by_col[rhs_c].extend(rule_list)
    for c_id in range(n_feats):
        rules_by_col[c_id].sort(key=lambda r: getattr(r, "avg_err", np.inf))

    cap = _positive_int_topk(topk)
    out: List[List[Dict[str, Any]]] = [[] for _ in range(n_feats)]

    for rhs_c in cols_loop:
        rules = rules_by_col[rhs_c]
        if cap is not None:
            rules = rules[:cap]
        if not rules:
            continue

        ser_r = np.asarray(arr[:, rhs_c], dtype=np.float64).ravel()
        occupied: List[Tuple[int, int]] = []
        all_segments: List[Dict[str, Any]] = []

        for rank, rule in enumerate(rules):
            if getattr(rule, "RHS", None) is None or int(rule.RHS[0]) != rhs_c:
                continue
            if not getattr(rule, "lhs_combinations", None):
                continue
            for combo in rule.lhs_combinations:
                combo_list = list(combo)
                ok, t2, d2, _lam, S_pred = predict_rhs_from_lhs_combination(
                    md, rule, combo_list, ser_r, is_only_motif=False
                )
                if (not ok) or S_pred is None:
                    continue
                t2 = int(t2)
                d2 = int(d2)
                if d2 <= 0:
                    continue
                t_end = t2 + d2
                t2c = max(0, t2)
                t_endc = min(n_points, t_end)
                if t_endc <= t2c:
                    continue
                free_parts = _subtract_half_open_interval(t2c, t_endc, occupied)
                for fa, fb in free_parts:
                    fa = max(0, fa)
                    fb = min(n_points, fb)
                    if fb <= fa:
                        continue
                    rel_lo = fa - t2
                    rel_hi = fb - t2
                    if rel_lo < 0 or rel_hi > len(S_pred):
                        continue
                    pred_slice = np.asarray(S_pred[rel_lo:rel_hi], dtype=np.float64).copy()
                    if pred_slice.size != fb - fa:
                        continue
                    all_segments.append(
                        {
                            "t_start": fa,
                            "t_end": fb,
                            "predicted": pred_slice,
                            "rule_sort_rank": rank,
                        }
                    )
                    occupied.append((fa, fb))
                occupied = _merge_half_open_intervals(occupied)

        out[rhs_c] = all_segments

    return out


def _rule_forecast_pred(
    *,
    dataname: str,
    rule_source_name: Optional[str] = None,
    epsilon: float = 1.0,
    topk: Any = None,
    rhs_cols: Optional[List[int]] = None,
    data_root: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    RHS rule-forecast error detection on TEST.
    Returns y_true, y_pred_rule (0/1), and meta dict for logging.
    """
    src = rule_source_name or dataname
    md, valid_rules = _load_task1_rule_assets(src)
    x_test = _load_test_aligned_to_train(dataname, data_root=data_root)
    n_points, n_feats = x_test.shape
    y_true = _load_sample_label_binary(dataname, n_points, data_root=data_root)

    if rhs_cols is None:
        cols_loop = list(range(n_feats))
    else:
        cols_loop = sorted({int(c) for c in rhs_cols})
        if not cols_loop:
            raise ValueError("rhs_cols is empty; use None for all attributes.")
        for c in cols_loop:
            if c < 0 or c >= n_feats:
                raise ValueError(
                    f"rhs_cols index {c} out of range 0..{n_feats - 1} (n_feats={n_feats})."
                )

    segments_per = _collect_rhs_ordered_forecast_segments(
        md=md,
        valid_rules=valid_rules,
        x=x_test,
        topk=topk,
        rhs_cols=rhs_cols,
    )

    pred_anom = np.zeros(n_points, dtype=bool)
    n_seg = 0
    for rhs_c in cols_loop:
        segs = segments_per[rhs_c]
        ser_r = np.asarray(x_test[:, rhs_c], dtype=np.float64).ravel()
        for seg in segs:
            fa = int(seg["t_start"])
            fb = int(seg["t_end"])
            pred_slice = np.asarray(seg["predicted"], dtype=np.float64).ravel()
            if fb <= fa or pred_slice.size != fb - fa:
                continue
            n_seg += 1
            actual = ser_r[fa:fb]
            hit = np.abs(pred_slice - actual) > float(epsilon)
            pred_anom[fa:fb] |= hit

    y_rule = pred_anom.astype(np.int64)
    cap = _positive_int_topk(topk)
    meta = {
        "dataname": dataname,
        "rule_source": src,
        "epsilon": float(epsilon),
        "n_points": int(n_points),
        "n_feats": int(n_feats),
        "n_forecast_segments": int(n_seg),
        "rhs_cols": list(cols_loop),
        "topk_rules": cap,
    }
    return y_true, y_rule, meta


def _load_dl_detection_y_pred_csv(csv_path: str, n_expected: int) -> np.ndarray:
    """Load y_pred from results/detection/.../{method}.csv."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing DL detection result: {csv_path}")
    df = pd.read_csv(csv_path)
    if "y_pred" not in df.columns:
        raise ValueError(f"{csv_path} missing y_pred column")
    y = df["y_pred"].to_numpy(dtype=np.int64)
    if y.shape[0] != n_expected:
        raise ValueError(
            f"{csv_path} row count {y.shape[0]} != TEST / label length {n_expected}"
        )
    return y


def run_rule_dl_fusion(
    *,
    dataname: str,
    rule_source_name: Optional[str] = None,
    epsilon: float = 1.0,
    topk: Any = None,
    rhs_cols: Optional[List[int]] = None,
    dl_methods: Optional[List[str]] = None,
    data_root: Optional[str] = None,
    detection_results_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rule + DL union fusion evaluation.

    1. Rule RHS forecast error detection on TEST (internal; used for fusion)
    2. Load results/detection/{dataname}/{method}.csv (y_pred from DL baselines)
    3. Per DL method: DL-only and rule-union-DL metrics
    """
    y_true, y_rule, meta = _rule_forecast_pred(
        dataname=dataname,
        rule_source_name=rule_source_name,
        epsilon=epsilon,
        topk=topk,
        rhs_cols=rhs_cols,
        data_root=data_root,
    )
    n_points = int(meta["n_points"])
    cap = meta["topk_rules"]
    topk_note = str(cap) if cap is not None else "all"
    rhs_note = (
        ",".join(str(c) for c in meta["rhs_cols"])
        if rhs_cols is not None
        else f"0..{meta['n_feats'] - 1}"
    )

    print(
        f"dataname={meta['dataname']} rule_src={meta['rule_source']} epsilon={meta['epsilon']} "
        f"rhs_cols=[{rhs_note}] topk_rules={topk_note} "
        f"n={n_points} forecast_segments={meta['n_forecast_segments']}"
    )

    base_out = os.path.join(
        detection_results_root
        if detection_results_root is not None
        else os.path.join(_ROOT, "results", "detection", dataname)
    )

    method_list = [x.lower() for x in (dl_methods or ["tranad", "timer", "timesnet"])]
    per_method: Dict[str, Dict[str, Any]] = {}

    for dm in method_list:
        p_csv = os.path.join(base_out, f"{dm}.csv")
        y_dl = _load_dl_detection_y_pred_csv(p_csv, n_points)
        m_dl = _binary_precision_recall_f1(y_true, y_dl)
        y_fuse = np.maximum(y_rule, y_dl).astype(np.int64)
        m_fuse = _binary_precision_recall_f1(y_true, y_fuse)

        _print_binary_metrics_line( f"{dm}_only", m_dl)
        _print_binary_metrics_line( f"rule_union_{dm}", m_fuse)

        per_method[dm] = {
            "dl_only": dict(m_dl),
            "rule_union": dict(m_fuse),
        }

    summary: Dict[str, Any] = {
        "dataname": dataname,
        "rule_meta": meta,
        "per_method": per_method,
    }
    return summary


if __name__ == "__main__":
    run_rule_dl_fusion(
        dataname="SKAB_VALVE2",
        rule_source_name=None,
        epsilon=1,
        topk=5,
        dl_methods=["timesnet"],
    )
