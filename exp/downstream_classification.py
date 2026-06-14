"""
Rule-boosted downstream classification

Prerequisites (in order):

1. Data CSVs
   - data/<dataname>/<dataname>_TRAIN.csv  training set (first column = class label)
   - data/<dataname>/<dataname>_TEST.csv    test set (first column = class label)

2. Motif mining results (motif_discovery.py task=2)
   - results/classification/<dataname>/motif_res/<sample_id>_<label>.pkl

3. Rule mining results (rule_discovery.py task=2 & 3)
   - results/classification/<dataname>/valid_rule_res/  serialized ValidRule files

4. Baseline test logits (train and export beforehand; e.g. GCC96 / MILLET96 / TIMESNET)
   - results/classification/<dataname>/<METHOD>_test_logits.npy
   - results/classification/<dataname>/<METHOD>_test_row_index.npy
   If baseline_smoke_quick=True, read from results/classification/<dataname>/smoke_quick/.

Pipeline: score every rule on a training sample -> pick top-k rules per class -> build rule
score matrix on the test set -> add baseline logits element-wise -> report baseline / fused accuracy.
"""

import os
from multiprocessing import Pool
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rule_discovery import rule_check_two_series_regions
from utils import load_instance, load_instance_list


def calc_rule_series_score(rule, ser_l, ser_r, md=None):
    """Score a single-LHS rule on a series. md must be the MotifDiscovery from the rule's source sample."""
    n_l, n_r = len(ser_l), len(ser_r)
    if not hasattr(rule, "LHS") or len(rule.LHS) != 1:
        raise ValueError("calc_rule_series_score: only single-LHS rules are supported")
    if md is None:
        raise ValueError("calc_rule_series_score: pass md=... (MotifDiscovery instance)")

    out = rule_check_two_series_regions(md, rule, ser_l, ser_r, plot=False)
    lhs_sat = out["lhs_sat_points"]
    rhs_vio_points = out["rhs_vio_points"]

    if lhs_sat <= 0:
        score = 0.0
    else:
        score = float(lhs_sat) / float(n_l) - float(rhs_vio_points) / float(n_r)
    return score, out["violation_regions"]


def sample_db_per_class(
    db: np.ndarray,
    sampling_rate: float,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified sampling by class label (first column).
    - If 0 < sampling_rate <= 1: per class, take max(1, ceil(class_size * sampling_rate)).
    - If sampling_rate > 1: per class, take min(int(sampling_rate), class_size).
    Returns (db_sampled, row_indices).
    """
    if db.size == 0:
        return db, np.zeros(0, dtype=np.int64)
    labels = db[:, 0]
    rng = np.random.default_rng(rng_seed)
    keep: List[int] = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        n = int(idx.size)
        if n == 0:
            continue
        if 0 < sampling_rate <= 1.0:
            take = max(1, int(np.ceil(n * float(sampling_rate))))
        else:
            take = min(int(sampling_rate), n)
        take = min(take, n)
        pick = rng.choice(idx, size=take, replace=False)
        keep.extend(int(x) for x in pick.tolist())
    keep.sort()
    row_indices = np.asarray(keep, dtype=np.int64)
    return db[row_indices], row_indices


_T6_DB: Optional[np.ndarray] = None
_T6_ROW_INDICES: Optional[np.ndarray] = None
_T6_MOTIF_RES_DIR: Optional[str] = None


def _task6_pool_init(
    db_sampled: np.ndarray,
    row_indices: np.ndarray,
    motif_res_dir: str,
) -> None:
    global _T6_DB, _T6_ROW_INDICES, _T6_MOTIF_RES_DIR
    _T6_DB = np.asarray(db_sampled, dtype=np.float64)
    _T6_ROW_INDICES = np.asarray(row_indices, dtype=np.int64)
    _T6_MOTIF_RES_DIR = motif_res_dir


def _task6_process_one_rule(item: Tuple[Any, int, Any]) -> Tuple[Any, int, float, float, float, float, int]:
    """One rule: accumulate score_same / score_diff over the injected training db."""
    import os as _os

    k, idx, rule = item
    t0 = perf_counter()
    pid = _os.getpid()
    assert _T6_DB is not None and _T6_ROW_INDICES is not None and _T6_MOTIF_RES_DIR is not None

    origin_sid = int(k[0])
    origin_lab = int(k[1])
    md_path = _os.path.join(_T6_MOTIF_RES_DIR, f"{origin_sid}_{origin_lab}.pkl")
    if not _os.path.isfile(md_path):
        elapsed = perf_counter() - t0
        return (k, idx, 0.0, 0.0, 0.0, elapsed, pid)
    md = load_instance(md_path)

    score_same = 0.0
    score_diff = 0.0
    target_label = k[1]
    n = _T6_DB.shape[0]
    for j in range(n):
        label = _T6_DB[j, 0]
        ser = np.asarray(_T6_DB[j, 1:], dtype=np.float64)
        score, _ = calc_rule_series_score(rule, ser, ser, md=md)
        if float(label) == float(target_label):
            score_same += score
        else:
            score_diff += score
    score_final = score_same / (score_diff + 1.0)
    elapsed = perf_counter() - t0
    return (k, idx, score_same, score_diff, score_final, elapsed, pid)


def score_rules_on_db_parallel(
    valid_rule_res: Dict[Any, List[Any]],
    db_sampled: np.ndarray,
    row_indices: np.ndarray,
    motif_res_dir: str,
    n_workers: int,
    verbose: bool = True,
) -> None:
    """Score every rule in valid_rule_res on db_sampled; write scores back onto each rule in place."""
    tasks: List[Tuple[Any, int, Any]] = []
    for k, rules in valid_rule_res.items():
        for idx, rule in enumerate(rules):
            tasks.append((k, idx, rule))
    total = len(tasks)
    if total == 0:
        return
    n_workers = max(1, int(n_workers))
    done = 0
    if n_workers == 1:
        _task6_pool_init(db_sampled, row_indices, motif_res_dir)
        for item in tasks:
            k, idx, s_same, s_diff, s_fin, elapsed, pid = _task6_process_one_rule(item)
            rule = valid_rule_res[k][idx]
            rule.score_same = s_same
            rule.score_diff = s_diff
            rule.score = s_fin
            done += 1
            if verbose:
                print(f"pid={pid} elapsed={elapsed:.4f}s  [{done}/{total}] key={k} idx={idx}")
        return

    with Pool(
        processes=n_workers,
        initializer=_task6_pool_init,
        initargs=(db_sampled, row_indices, motif_res_dir),
    ) as pool:
        for res in pool.imap_unordered(_task6_process_one_rule, tasks, chunksize=1):
            k, idx, s_same, s_diff, s_fin, elapsed, pid = res
            rule = valid_rule_res[k][idx]
            rule.score_same = s_same
            rule.score_diff = s_diff
            rule.score = s_fin
            done += 1
            if verbose:
                print(f"pid={pid} elapsed={elapsed:.4f}s  [{done}/{total}] key={k} idx={idx}")


def task6_rule_score_on_series(
    rule_key: Tuple[Any, ...],
    rule: Any,
    ser: np.ndarray,
    motif_res_dir: str,
) -> float:
    """Load md from the rule's source sample and score one rule on ser."""
    import os as _os

    origin_sid = int(rule_key[0])
    origin_lab = int(rule_key[1])
    md_path = _os.path.join(motif_res_dir, f"{origin_sid}_{origin_lab}.pkl")
    if not _os.path.isfile(md_path):
        return 0.0
    md = load_instance(md_path)
    score, _ = calc_rule_series_score(rule, ser, ser, md=md)
    return float(score)


def task6_sorted_class_order(*label_arrays: np.ndarray) -> List[float]:
    """All class labels seen in train/test, sorted ascending."""
    parts = [np.asarray(a[:, 0]).ravel() for a in label_arrays if a is not None and a.size]
    if not parts:
        return []
    labs = np.unique(np.concatenate(parts))
    return sorted(float(x) for x in labs.tolist())


def select_topk_rules_per_class(
    valid_rule_res: Dict[Any, List[Any]],
    top_k: int,
) -> Dict[float, List[Tuple[Any, int, Any]]]:
    """Group by key[1]; per class, take top_k rules by rule.score descending."""
    by_class: Dict[float, List[Tuple[Any, int, Any, float]]] = {}
    for rk, rules in valid_rule_res.items():
        j = float(rk[1])
        for idx, rule in enumerate(rules):
            sc = float(getattr(rule, "score", 0.0) or 0.0)
            by_class.setdefault(j, []).append((rk, idx, rule, sc))
    out: Dict[float, List[Tuple[Any, int, Any]]] = {}
    tk = max(0, int(top_k))
    for j, items in by_class.items():
        items.sort(key=lambda t: -t[3])
        out[j] = [(t[0], t[1], t[2]) for t in items[:tk]]
    return out


def task6_sample_class_score_vector(
    ser: np.ndarray,
    class_order: Sequence[float],
    repr_by_class: Dict[float, List[Tuple[Any, int, Any]]],
    motif_res_dir: str,
) -> np.ndarray:
    """One test series: mean top-k rule score per class."""
    ser = np.asarray(ser, dtype=np.float64).ravel()
    vec = np.zeros(len(class_order), dtype=np.float64)
    for ji, j in enumerate(class_order):
        jf = float(j)
        items = repr_by_class.get(jf, [])
        if not items:
            continue
        acc = 0.0
        for rk, _idx, rule in items:
            acc += task6_rule_score_on_series(rk, rule, ser, motif_res_dir)
        vec[ji] = acc / len(items)
    return vec


_T6_TEST_ROWS: Optional[np.ndarray] = None
_T6_CLASS_ORDER: Optional[List[float]] = None
_T6_REPR_BY_CLASS: Optional[Dict[float, List[Tuple[Any, int, Any]]]] = None
_T6_MOTIF_DIR_TEST: Optional[str] = None


def _task6_test_pool_init(
    db_test: np.ndarray,
    class_order: Sequence[float],
    repr_by_class: Dict[float, List[Tuple[Any, int, Any]]],
    motif_res_dir: str,
) -> None:
    global _T6_TEST_ROWS, _T6_CLASS_ORDER, _T6_REPR_BY_CLASS, _T6_MOTIF_DIR_TEST
    _T6_TEST_ROWS = np.asarray(db_test, dtype=np.float64)
    _T6_CLASS_ORDER = list(float(x) for x in class_order)
    _T6_REPR_BY_CLASS = repr_by_class
    _T6_MOTIF_DIR_TEST = motif_res_dir


def _task6_test_one_row(row_index: int) -> Tuple[int, np.ndarray]:
    assert _T6_TEST_ROWS is not None and _T6_CLASS_ORDER is not None
    assert _T6_REPR_BY_CLASS is not None and _T6_MOTIF_DIR_TEST is not None
    row = _T6_TEST_ROWS[row_index]
    ser = np.asarray(row[1:], dtype=np.float64)
    vec = task6_sample_class_score_vector(
        ser, _T6_CLASS_ORDER, _T6_REPR_BY_CLASS, _T6_MOTIF_DIR_TEST
    )
    return row_index, vec


def score_test_class_vectors_parallel(
    db_test: np.ndarray,
    class_order: Sequence[float],
    repr_by_class: Dict[float, List[Tuple[Any, int, Any]]],
    motif_res_dir: str,
    n_workers: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score every test row in parallel; return (score_matrix, y_true)."""
    n = int(db_test.shape[0])
    n_cls = len(class_order)
    score_matrix = np.zeros((n, n_cls), dtype=np.float64)
    y_true = np.asarray(db_test[:, 0], dtype=np.float64)
    if n == 0:
        return score_matrix, y_true
    n_workers = max(1, int(n_workers))
    done = 0
    if n_workers == 1:
        _task6_test_pool_init(db_test, class_order, repr_by_class, motif_res_dir)
        for i in range(n):
            ri, vec = _task6_test_one_row(i)
            score_matrix[ri] = vec
            done += 1
            if verbose:
                print(f"[test scoring] [{done}/{n}] row={ri}")
        return score_matrix, y_true

    with Pool(
        processes=n_workers,
        initializer=_task6_test_pool_init,
        initargs=(db_test, class_order, repr_by_class, motif_res_dir),
    ) as pool:
        for ri, vec in pool.imap_unordered(_task6_test_one_row, range(n), chunksize=1):
            score_matrix[ri] = vec
            done += 1
            if verbose:
                print(f"[test scoring] [{done}/{n}] row={ri}")
    return score_matrix, y_true


def task6_predict_labels_from_score_matrix(
    score_matrix: np.ndarray,
    class_order: Sequence[float],
) -> np.ndarray:
    """Argmax per row to obtain predicted class labels."""
    order = np.asarray(list(class_order), dtype=np.float64)
    idx = np.argmax(score_matrix, axis=1)
    return order[idx]


def task6_classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true, dtype=np.float64) == np.asarray(y_pred, dtype=np.float64)))


def load_baseline_test_logits(
    dataname: str,
    method: str,
    *,
    project_root: Optional[str] = None,
    smoke_quick: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load baseline test logits and row indices from disk."""
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    m = (method or "").strip().upper().replace("-", "")
    prefix = "TIMESNET" if m in ("TIMESNET", "TIMESNET96") else m
    out_tail = os.path.join(dataname, "smoke_quick") if smoke_quick else dataname
    out_dir = os.path.join(root, "results", "classification", out_tail)
    logits_path = os.path.join(out_dir, f"{prefix}_test_logits.npy")
    row_path = os.path.join(out_dir, f"{prefix}_test_row_index.npy")
    logits = np.load(logits_path)
    row_index = np.load(row_path).astype(np.int64)
    n_total = int(row_index.max()) + 1 if row_index.size else int(logits.shape[0])
    full = np.zeros((n_total, int(logits.shape[1])), dtype=np.float64)
    full[row_index] = logits.astype(np.float64)
    return full, row_index


def fuse_rule_scores_with_baseline_logits(
    rule_score_matrix: np.ndarray,
    baseline_logits: np.ndarray,
) -> np.ndarray:
    """Fuse by element-wise addition."""
    if rule_score_matrix.shape != baseline_logits.shape:
        raise ValueError(
            f"shape mismatch: rule={rule_score_matrix.shape}, baseline={baseline_logits.shape}"
        )
    return rule_score_matrix.astype(np.float64) + baseline_logits.astype(np.float64)


def align_logits_to_class_order(
    logits: np.ndarray,
    class_order: Sequence[float],
) -> np.ndarray:
    """Align baseline logit columns to class_order (e.g. Trace labels 1..4 -> columns 1..4)."""
    logits = np.asarray(logits, dtype=np.float64)
    col_ids = [int(float(x)) for x in class_order]
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N,C); got shape={logits.shape}")
    c = int(logits.shape[1])
    if any((j < 0 or j >= c) for j in col_ids):
        raise ValueError(f"class_order={list(class_order)} out of logits column range C={c}")
    return logits[:, col_ids]


def run_rule_boost_experiment(
    datanames: Sequence[str],
    baseline_methods: Sequence[str],
    *,
    sampling_rate: float = 0.3,
    repr_top_k: int = 3,
    n_workers: Optional[int] = None,
    baseline_smoke_quick: bool = False,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rule-boost experiment over datanames x baseline_methods.
    Rule-side results per dataname are computed once and reused; each baseline reports acc_base / acc_fused.
    """
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    nw = int(n_workers) if n_workers is not None else min(8, os.cpu_count() or 1)

    cache_by_dn: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {"settings": {"sampling_rate": sampling_rate, "repr_top_k": repr_top_k, "n_workers": nw}}

    for dn in datanames:
        dn = str(dn)
        if dn not in cache_by_dn:
            db = pd.read_csv(os.path.join(root, "data", dn, f"{dn}_TRAIN.csv")).values
            db_test = pd.read_csv(os.path.join(root, "data", dn, f"{dn}_TEST.csv")).values
            dir_path = os.path.join(root, "results", "classification", dn, "valid_rule_res")
            valid_rule_res = load_instance_list(dir_path)

            db_sampled, row_indices = sample_db_per_class(db, sampling_rate, rng_seed=0)
            print(f"[{dn}] train={db.shape[0]} sampled={db_sampled.shape[0]} sampling_rate={sampling_rate}")

            motif_res_dir = os.path.join(root, "results", "classification", dn, "motif_res")

            score_rules_on_db_parallel(
                valid_rule_res,
                db_sampled,
                row_indices,
                motif_res_dir,
                n_workers=nw,
                verbose=True,
            )

            class_order = task6_sorted_class_order(db, db_test)
            repr_by_class = select_topk_rules_per_class(valid_rule_res, repr_top_k)
            score_mat, y_true = score_test_class_vectors_parallel(
                db_test,
                class_order,
                repr_by_class,
                motif_res_dir,
                n_workers=nw,
                verbose=True,
            )

            cache_by_dn[dn] = {
                "class_order": class_order,
                "score_mat": score_mat,
                "y_true": y_true,
            }

        c = cache_by_dn[dn]
        score_mat = c["score_mat"]
        y_true = c["y_true"]
        class_order = c["class_order"]

        report.setdefault(dn, {})
        for method in baseline_methods:
            m = str(method)
            try:
                base_logits, _ = load_baseline_test_logits(
                    dn,
                    m,
                    project_root=root,
                    smoke_quick=baseline_smoke_quick,
                )
                base_logits = base_logits[: score_mat.shape[0]]
                base_logits = align_logits_to_class_order(base_logits, class_order)
                y_pred_base = task6_predict_labels_from_score_matrix(base_logits, class_order)
                acc_base = task6_classification_accuracy(y_true, y_pred_base)

                fused_logits = fuse_rule_scores_with_baseline_logits(score_mat, base_logits)
                y_pred_fused = task6_predict_labels_from_score_matrix(fused_logits, class_order)
                acc_fused = task6_classification_accuracy(y_true, y_pred_fused)

                report[dn][m] = {"acc_base": acc_base, "acc_fused": acc_fused}
                print(f"[{dn} + {m}] base={acc_base:.4f} fused={acc_fused:.4f}")
            except Exception as e:
                report[dn][m] = {"error": str(e)}
                print(f"[{dn} + {m}] skipped (failed to load logits or align): {e}")

    return report


if __name__ == "__main__":

    DATANAMES = ["Coffee"]
    BASELINE_METHODS = ["TIMESNET"]

    rep = run_rule_boost_experiment(
        DATANAMES,
        BASELINE_METHODS,
        sampling_rate=0.3,
        repr_top_k=3,
        n_workers=min(8, os.cpu_count() or 1),
        baseline_smoke_quick=False,
        project_root=os.path.dirname(os.path.abspath(__file__)),
    )
    print("Summary:", rep)
