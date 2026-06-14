from typing import Any
from utils import *
from motif_discovery import *
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import random
from multiprocessing import Pool
import glob
import shutil
from numba import njit
import os
from datetime import datetime
import csv


class Node():
    def __init__(self, lhs, rhs, father, ignore, node_id,):
        self.lhs=lhs
        self.rhs=rhs
        self.father=father
        self.ignore=ignore
        self.node_id=node_id

class ValidRule():
    def __init__(self, LHS, RHS, C_phi, supp, params, lhs_combinations, avg_err,epsilon):
        self.LHS=LHS
        self.RHS=RHS
        self.C_phi=C_phi
        self.supp=supp
        self.params=params
        self.lhs_combinations = lhs_combinations
        self.avg_err = avg_err
        self.epsilon = epsilon


def _intervals_from_boolean_mask(mask):
    n = int(len(mask))
    out = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def scan_lhs_matches_double_matching4_style(md, data_1d, c_id, class_id):
    data = np.asarray(data_1d, dtype=np.float64).ravel()
    n = len(data)
    sub_sq_set = md.G[c_id][class_id]
    class_id = sub_sq_set.class_id
    c_id = sub_sq_set.c_id
    para, kind = sub_sq_set.f
    gamma = sub_sq_set.gamma
    epsilonM = sub_sq_set.epsilon
    window_rate = md.window_rate
    delta_range = sub_sq_set.delta_range
    lambda_lower_bound = sub_sq_set.lambda_lower_bound

    if kind != "piecewise":
        para = np.asarray(para, dtype=np.float64)
        breaks = slopes = intercepts = None
    else:
        breaks = np.asarray(para["breaks"], dtype=np.float64)
        slopes = np.asarray(para["slopes"], dtype=np.float64)
        intercepts = np.asarray(para["intercepts"], dtype=np.float64)

    predict_fn = select_predict_nb(kind)
    delta_list = list(range(delta_range[1], delta_range[0] - 1, -1))
    x_norm_cache = {d: np.linspace(0, 1, d, dtype=np.float64) for d in delta_list}
    f_cache = {}
    for d in delta_list:
        xn = x_norm_cache[d]
        if kind != "piecewise":
            f_cache[d] = predict_fn(xn, para)
        else:
            f_cache[d] = predict_fn(xn, breaks=breaks, slopes=slopes, intercepts=intercepts)
    window_cache = {d: int(window_rate * d) for d in delta_list}

    matches = []
    md.forbidden_area = set()
    md.forbidden_area2 = []
    local_forbidden = md.forbidden_area

    for t1 in range(n):
        if t1 in local_forbidden:
            continue
        base = data[t1]
        for delta in delta_list:
            if t1 + delta > n:
                continue
            S = data[t1 : t1 + delta] - base
            if len(S) < delta_range[0]:
                continue
            lam = float(np.max(S) - np.min(S))
            if lam < lambda_lower_bound:
                continue
            
            S_pred = f_cache[delta] * lam
            error = calc_error_local_alignment4(
                S, S_pred, window_cache[delta], md.gamma_penalty
            )
            err_agg = md._error_aggregate(error)
            temp = err_agg
            if lam * gamma != 0:
                ratio = lam / delta / gamma
                temp = err_agg * max(ratio, 1.0 / ratio) ** md.gamma_penalty

            if temp <= epsilonM:
                matches.append(SubSq(t1, delta, [lam, lam], class_id, c_id))
                forbid_end = t1 + int(delta * md.minimum_interval) + 1
                local_forbidden.update(range(t1, forbid_end))
                md.forbidden_area2.append((t1, forbid_end))
                break
        
    return matches


def plot_series_regions(series_1d, irrelevant_regions, violation_regions, satisfaction_regions, *, title=None, show=False, save_path=None):
    s = np.asarray(series_1d, dtype=np.float64).ravel()
    n = len(s)
    x = np.arange(n)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, s, "k-", lw=1.0, alpha=0.8, label="series (irrelevant/base)")

    def _plot_ranges(ranges, color, label):
        first = True
        for a, b in ranges:
            a = max(0, int(a))
            b = min(n, int(b))
            if a >= b:
                continue
            ax.plot(x[a:b], s[a:b], color=color, lw=2.2, alpha=0.9, label=(label if first else None))
            first = False

    _plot_ranges(satisfaction_regions, "b", "satisfaction")
    _plot_ranges(violation_regions, "r", "violation")

    if title:
        ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def rule_check_two_series_regions(
    md,
    rule,
    ser_l,
    ser_r,
    *,
    point_epsilon=None,
    max_multi_lhs_combos=5000,
    plot=False,
):
    lhs_data = np.asarray(ser_l, dtype=np.float64).ravel()
    rhs_data = np.asarray(ser_r, dtype=np.float64).ravel()
    n_l = int(len(lhs_data))
    n_r = int(len(rhs_data))
    thr = float(point_epsilon) if point_epsilon is not None else float(rule.epsilon)
    if not np.isfinite(thr):
        thr = float(md.epsilonM)

    lhs_tuples = list(rule.LHS)
    snapshots = {(c, k): md.G[c][k].SubSq_obj_list[:] for c, k in lhs_tuples}
    match_lists = [
        scan_lhs_matches_double_matching4_style(md, lhs_data, c, k) for c, k in lhs_tuples
    ]

    lhs_cover = np.zeros(n_l, dtype=bool)
    for ml in match_lists:
        for sub in ml:
            t1, d = int(sub.t1), int(sub.delta)
            te = min(t1 + d, n_l)
            if t1 < te:
                lhs_cover[t1:te] = True

    if (not lhs_tuples) or any(len(m) == 0 for m in match_lists) or (len(lhs_tuples) > 1):
        for (c, k), snap in snapshots.items():
            md.G[c][k].SubSq_obj_list = snap
        out = {
            "irrelevant_regions": _intervals_from_boolean_mask(~lhs_cover),
            "violation_regions": [],
            "satisfaction_regions": [],
            "lhs_sat_points": int(np.sum(lhs_cover)),
            "rhs_vio_points": 0,
        }
        if plot:
            plot_series_regions(lhs_data, out["irrelevant_regions"], [], [], title="LHS regions")
        return out

    rhs_touched = np.zeros(n_r, dtype=bool)
    vio = np.zeros(n_r, dtype=bool)
    ncomb = 0
    for combo in itertools.product(*[range(len(m)) for m in match_lists]):
        ncomb += 1
        if ncomb > max_multi_lhs_combos:
            break
        for i, (c, k) in enumerate(lhs_tuples):
            md.G[c][k].SubSq_obj_list = match_lists[i]
        ok, t2, d2, _lam2, S_pred = predict_rhs_from_lhs_combination(
            md, rule, list(combo), rhs_data
        )
        if (not ok) or (S_pred is None):
            continue
        d2 = int(d2)
        t2 = int(t2)
        if t2 < 0 or t2 + d2 > n_r or d2 <= 0:
            continue
        actual = rhs_data[t2 : t2 + d2]
        min_len = min(int(len(S_pred)), int(len(actual)))
        if min_len <= 0:
            continue
        window = 0
        if min_len > 1:
            window = max(0, min(int(d2 * md.window_rate), min_len - 1))
        err = calc_error_local_alignment4(
            np.asarray(S_pred[:min_len], dtype=np.float64),
            np.asarray(actual[:min_len], dtype=np.float64),
            window,
            md.gamma_penalty,
        )
        if err is None:
            continue
        rhs_touched[t2 : t2 + min_len] = True
        for j in range(min_len):
            if err[j] > thr:
                vio[t2 + j] = True

    for (c, k), snap in snapshots.items():
        md.G[c][k].SubSq_obj_list = snap

    sat = rhs_touched & ~vio
    out = {
        "irrelevant_regions": _intervals_from_boolean_mask(~lhs_cover),
        "violation_regions": _intervals_from_boolean_mask(vio),
        "satisfaction_regions": _intervals_from_boolean_mask(sat),
        "lhs_sat_points": int(np.sum(lhs_cover)),
        "rhs_vio_points": int(np.sum(vio)),
    }
    if plot:
        plot_series_regions(rhs_data, [], out["violation_regions"], out["satisfaction_regions"], title="RHS regions")
    return out



class Motif_pair():
    def __init__(
                self,c_ids,class_ids,
                 ):
        self.c_ids=c_ids
        self.class_ids=class_ids
        self.best_match=None
        self.best_score=None
        self.best_supp=None
        self.best_params=None

class RuleDiscovery():
    def __init__(self, md, supp_threshold=0.7, max_iter=5000, is_plot=False,is_print=True, para_model_type='linear',confidence_threshold=0.95,epsilon_rate=0.8, plot_save_prefix=None, ransac7_discard_match_if_min_eps_rate_gt_budget=False, max_lhs_len=None):
        self.md=md
        self.is_plot=is_plot 
        self.max_iter=max_iter 
        self.supp_threshold=supp_threshold
        self.para_model_type=para_model_type
        if para_model_type=="linear":
            self.C_phi_threshold=2 
        else: 
            self.C_phi_threshold=5 
        self.mp=None
        self.confidence_threshold = confidence_threshold
        self.is_print=is_print
        self.epsilon_rate=epsilon_rate
        self.plot_save_prefix = plot_save_prefix
        self.ransac7_discard_match_if_min_eps_rate_gt_budget = ransac7_discard_match_if_min_eps_rate_gt_budget
        self.max_lhs_len = max_lhs_len

    def _prefixed_temp_rule_plot_filename(self, rule_core_stem):
        pre = getattr(self, "plot_save_prefix", None)
        if pre is not None and str(pre) != "":
            safe = (
                str(pre)
                .replace("\\", "_")
                .replace("/", "_")
                .replace(":", "_")
            )
            if os.altsep:
                safe = safe.replace(os.altsep, "_")
            return f"{safe}_{rule_core_stem}.png"
        return f"{rule_core_stem}.png"




    def find_best_motif_match_linear_ransac6(
        self,
        c_ids,
        class_ids,
        is_plot=False,
        task='3par',
        seed=42,
    ):
        n_lhs = len(c_ids) - 1
        min_regression_samples = get_min_regression_samples(n_lhs)
        sq_sets = []
        for i in range(len(c_ids)):
            temp = self.md.G[c_ids[i]][class_ids[i]].SubSq_obj_list
            if len(temp) < min_regression_samples:
                return 0, [], 0, {}, [], float("inf"), float("inf")
            sq_sets.append(temp)

        D = len(c_ids)
        best_score = D - 1
        best_num_lhs = D - 1
        best_match = []
        best_params = {}
        best_lhs_combinations = []
        best_avg_err = float("inf")
        best_epsilon = float("inf")

        t1_arrays = [np.array([s.t1 for s in sq_sets[i]], dtype=np.float64) for i in range(len(sq_sets))]
        sq_lens = [len(s) for s in sq_sets]
        n_sample = max(D, min_regression_samples)
        max_iter = self.max_iter
        _ = 0
        ta = time()


        while _ <= max_iter:
            _ += 1
            if self.is_print and _ % 500 == 0:
                print("     rounds=", _)
                print("     tb-ta=", round(time() - ta, 3))
                print("     best_score=", best_score)

            t2=perf_counter()
            combination0 = [random.randrange(sq_lens[s_id]) for s_id in range(len(sq_sets))]
            combination0_obj = [sq_sets[s_id][combination0[s_id]] for s_id in range(len(sq_sets))]
            combinations = []
            combination_objs = []
            for _id in range(1, n_sample):
                i0 = random.randrange(sq_lens[0])
                while i0 == combination0[0] or i0 in [c[0] for c in combinations]:
                    i0 = random.randrange(sq_lens[0])
                combinations.append([i0])
                combination_objs.append([sq_sets[0][i0]])
                for id2 in range(1, len(c_ids)):
                    t_reference = combination0_obj[id2].t1 + combination_objs[-1][0].t1 - combination0_obj[0].t1
                    arr = t1_arrays[id2]
                    pos = np.searchsorted(arr, t_reference)
                    if pos == 0:
                        id3 = 0
                    elif pos == len(arr):
                        id3 = len(arr) - 1
                    else:
                        id3 = pos if abs(arr[pos] - t_reference) < abs(arr[pos - 1] - t_reference) else pos - 1
                    combinations[-1].append(id3)
                    combination_objs[-1].append(sq_sets[id2][id3])
            combinations = [combination0] + combinations
            combination_objs = [combination0_obj] + combination_objs
            t3=perf_counter()
            indep_subsqs = [[x[i] for i in range(len(x) - 1)] for x in combination_objs]
            dep_subsqs = [x[-1] for x in combination_objs]
            
            try:
                para, t1_col_fit = sample_parameter_from_pair5(indep_subsqs, dep_subsqs)
            except Exception as e:
                print(e)
                continue
            t4=perf_counter()
            valid_pairs, num_lhs, params, avg_err, epsilon = self.compute_valid_pair5(
                c_ids, class_ids, sq_sets, para, t1_col_fit,
            )
            t5=perf_counter()
            ub = upper_bound_from_valid_pairs(valid_pairs)
            if ub <= best_score:
                continue
            if not valid_pairs or not params:
                continue
            t6=perf_counter()

            score = len(valid_pairs)
            match = valid_pairs
            update_best = (
                (score > best_score)
                or (score == best_score and (epsilon < best_epsilon or avg_err < best_avg_err))
            )

            if update_best:
                best_score = score
                best_match = match
                best_num_lhs = num_lhs
                best_params = params
                best_lhs_combinations = valid_pairs
                best_avg_err = avg_err
                best_epsilon = epsilon
                if score > best_score:
                    max_iter = Calc_max_turns(
                        self.confidence_threshold, D, best_score, sq_lens[0], sq_lens[-1],
                    )
                    if self.is_print:
                        print("max_iter=", max_iter)
                    max_iter = min(self.max_iter, max_iter)
                    if max_iter < 500:
                        max_iter = 500

        supp = best_score / best_num_lhs if best_num_lhs > 0 else 0
        
        if (
            # self.is_plot      #**********
             supp >= self.supp_threshold
            and best_score >= min_regression_samples
        ):
            temp = [(c_ids[x], class_ids[x]) for x in range(len(c_ids))]
            print("-" * 100)
            print("rule=", temp[:-1], temp[-1])
            print("best_score=", best_score, "best_num_lhs=", best_num_lhs)
            print("avg_err=", best_avg_err, "epsilon=", best_epsilon)
            self.plot_best_match_sequences(c_ids, class_ids, best_match, best_params, best_avg_err, best_epsilon)
            print("-" * 100)

        best_lhs_combinations = [x[:-1] for x in best_match]
        return best_score, best_match, supp, best_params, best_lhs_combinations, best_avg_err, best_epsilon

    def find_best_motif_match_linear_ransac7(
        self,
        c_ids,
        class_ids,
        is_plot=False,
        task='3par',
        seed=42,
    ):
        n_lhs = len(c_ids) - 1
        min_regression_samples = get_min_regression_samples(n_lhs)
        sq_sets = []
        for i in range(len(c_ids)):
            temp = self.md.G[c_ids[i]][class_ids[i]].SubSq_obj_list
            if len(temp) < min_regression_samples:
                return 0, [], 0, {}, [], float("inf"), float("inf"), float("inf")
            sq_sets.append(temp)

        D = len(c_ids)
        best_score = D - 1
        best_num_lhs = D - 1
        best_match = []
        best_params = {}
        best_lhs_combinations = []
        best_avg_err = float("inf")
        best_epsilon = float("inf")
        best_min_eps_rate = float("inf")
        best_eps_rate_stats = {
            "max": float("inf"),
            "q90": float("inf"),
            "q80": float("inf"),
            "q70": float("inf"),
            "q60": float("inf"),
            "q50": float("inf"),
        }

        t1_arrays = [np.array([s.t1 for s in sq_sets[i]], dtype=np.float64) for i in range(len(sq_sets))]
        sq_lens = [len(s) for s in sq_sets]
        n_sample = max(D, min_regression_samples)
        max_iter = self.max_iter
        _ = 0
        ta = time()


        while _ <= max_iter:
            _ += 1
            if self.is_print and _ % 500 == 0:
                print("     rounds=", _)
                print("     tb-ta=", round(time() - ta, 3))
                print("     best_score=", best_score)

            t2=perf_counter()
            combination0 = [random.randrange(sq_lens[s_id]) for s_id in range(len(sq_sets))]
            combination0_obj = [sq_sets[s_id][combination0[s_id]] for s_id in range(len(sq_sets))]
            combinations = []
            combination_objs = []
            for _id in range(1, n_sample):
                i0 = random.randrange(sq_lens[0])
                while i0 == combination0[0] or i0 in [c[0] for c in combinations]:
                    i0 = random.randrange(sq_lens[0])
                combinations.append([i0])
                combination_objs.append([sq_sets[0][i0]])
                for id2 in range(1, len(c_ids)):
                    t_reference = combination0_obj[id2].t1 + combination_objs[-1][0].t1 - combination0_obj[0].t1
                    arr = t1_arrays[id2]
                    pos = np.searchsorted(arr, t_reference)
                    if pos == 0:
                        id3 = 0
                    elif pos == len(arr):
                        id3 = len(arr) - 1
                    else:
                        id3 = pos if abs(arr[pos] - t_reference) < abs(arr[pos - 1] - t_reference) else pos - 1
                    combinations[-1].append(id3)
                    combination_objs[-1].append(sq_sets[id2][id3])
            combinations = [combination0] + combinations
            combination_objs = [combination0_obj] + combination_objs
            t3=perf_counter()
            indep_subsqs = [[x[i] for i in range(len(x) - 1)] for x in combination_objs]
            dep_subsqs = [x[-1] for x in combination_objs]
            
            try:
                para, t1_col_fit = sample_parameter_from_pair5(indep_subsqs, dep_subsqs)
            except Exception as e:
                print(e)
                input("Press Enter to continue...")
                continue
            t4=perf_counter()
            valid_pairs, num_lhs, params, avg_err, epsilon, eps_rate_need, eps_rate_stats = self.compute_valid_pair6(
                c_ids, class_ids, sq_sets, para, t1_col_fit,
            )
            t5=perf_counter()
            ub = upper_bound_from_valid_pairs(valid_pairs)
            if ub <= best_score:
                continue
            if not valid_pairs or not params:
                continue
            t6=perf_counter()

            score = len(valid_pairs)
            match = valid_pairs
            update_best = (
                score > best_score
                or (
                    score == best_score
                    and (
                        eps_rate_need < best_min_eps_rate
                        or (
                            eps_rate_need == best_min_eps_rate
                            and (epsilon < best_epsilon or avg_err < best_avg_err)
                        )
                    )
                )
            )

            if update_best:
                prev_best_score = best_score
                best_score = score
                best_match = match
                best_num_lhs = num_lhs
                best_params = params
                best_lhs_combinations = valid_pairs
                best_avg_err = avg_err
                best_epsilon = epsilon
                best_min_eps_rate = eps_rate_need
                best_eps_rate_stats = eps_rate_stats
                if score > prev_best_score:
                    max_iter = Calc_max_turns(
                        self.confidence_threshold, D, best_score, sq_lens[0], sq_lens[-1],
                    )
                    if self.is_print:
                        print("max_iter=", max_iter)
                    max_iter = min(self.max_iter, max_iter)
                    if max_iter < 500:
                        max_iter = 500

        supp = best_score / best_num_lhs if best_num_lhs > 0 else 0

        if supp >= self.supp_threshold and best_score >= min_regression_samples:
            temp = [(c_ids[x], class_ids[x]) for x in range(len(c_ids))]
            print("-" * 100)
            print("rule=", temp[:-1], temp[-1])
            print("best_score=", best_score, "best_num_lhs=", best_num_lhs)
            print("avg_err=", best_avg_err, "epsilon=", best_epsilon)
            print("min_epsilon_rate (for DP match) =", best_min_eps_rate)
            print(
                "eps_rate quantiles:",
                {k: (round(v, 6) if np.isfinite(v) else v) for k, v in best_eps_rate_stats.items()},
            )
            print("-" * 100)

        if self.is_plot:
            self.plot_best_match_sequences_ransac7(
                c_ids,
                class_ids,
                best_match,
                best_params,
                best_avg_err,
                best_epsilon,
                best_min_eps_rate,
                best_eps_rate_stats,
                supp,
            )

        if getattr(self, "ransac7_discard_match_if_min_eps_rate_gt_budget", False):
            retain = (
                best_match
                and best_params
                and np.isfinite(best_min_eps_rate)
                and best_min_eps_rate <= self.epsilon_rate
            )
            if not retain and best_match:
                best_score = 0
                best_match = []
                best_params = {}
                best_avg_err = float("inf")
                best_epsilon = float("inf")
                best_min_eps_rate = float("inf")
                best_eps_rate_stats = {
                    "max": float("inf"),
                    "q90": float("inf"),
                    "q80": float("inf"),
                    "q70": float("inf"),
                    "q60": float("inf"),
                    "q50": float("inf"),
                }
                supp = 0.0

        best_lhs_combinations = [x[:-1] for x in best_match]
        return (
            best_score,
            best_match,
            supp,
            best_params,
            best_lhs_combinations,
            best_avg_err,
            best_epsilon,
            best_min_eps_rate,
        )

    def plot_best_match_sequences_ransac7(
        self,
        c_ids,
        class_ids,
        best_match,
        best_params,
        avg_err,
        best_epsilon,
        min_epsilon_rate_need,
        eps_rate_stats,
        supp,
    ):
        if best_match and best_params:
            self.plot_best_match_sequences(
                c_ids,
                class_ids,
                best_match,
                best_params,
                avg_err,
                best_epsilon,
                min_epsilon_rate_need=min_epsilon_rate_need,
                eps_rate_stats=eps_rate_stats,
            )
            return

        db = self.md.db
        T = db.shape[0]
        x = np.arange(T)
        n_attr = len(c_ids)
        fig, axes = plt.subplots(n_attr, 1, sharex=True, figsize=(10, 2 * n_attr))
        if n_attr == 1:
            axes = [axes]

        title_parts = [
            f"avg_err = {avg_err:.6g}" if np.isfinite(avg_err) else "avg_err = inf",
            f"epsilon = {best_epsilon:.6g}" if np.isfinite(best_epsilon) else "epsilon = inf",
            f"min_eps_rate = {min_epsilon_rate_need:.6g}" if np.isfinite(min_epsilon_rate_need) else "min_eps_rate = inf",
            f"supp = {supp:.6g}" if np.isfinite(supp) else "supp = inf",
            (
                "eps_q: "
                f"max={eps_rate_stats.get('max', float('inf')):.4g},"
                f"q90={eps_rate_stats.get('q90', float('inf')):.4g},"
                f"q80={eps_rate_stats.get('q80', float('inf')):.4g},"
                f"q70={eps_rate_stats.get('q70', float('inf')):.4g},"
                f"q60={eps_rate_stats.get('q60', float('inf')):.4g},"
                f"q50={eps_rate_stats.get('q50', float('inf')):.4g}"
            ),
            "no_valid_match_segments",
        ]
        fig.suptitle("   |   ".join(title_parts), fontsize=10)

        for j in range(n_attr):
            c_id = c_ids[j]
            cls = class_ids[j]
            series = db[:, c_id]
            ax = axes[j]
            ax.plot(x, series, "b-", lw=1, label="series")
            role = "RHS" if j == n_attr - 1 else "LHS"
            ax.set_title(f"Attr {c_id}, class {cls} ({role})", fontsize=9)
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time index")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs("results/temp_valid_rules", exist_ok=True)
        name_parts = [f"({c_ids[i]}_{class_ids[i]})" for i in range(len(c_ids))]
        fname = self._prefixed_temp_rule_plot_filename("_".join(name_parts))
        save_path = os.path.join("results/temp_valid_rules", fname)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)





    def plot_best_match_sequences(
        self,
        c_ids,
        class_ids,
        best_match,
        best_params,
        avg_err,
        best_epsilon,
        min_epsilon_rate_need=None,
        eps_rate_stats=None,
    ):
        if not best_match or best_params is None:
            return

        n_attr = len(c_ids)
        n_lhs = n_attr - 1

        db = self.md.db
        T = db.shape[0]

        class _TmpRule:
            pass

        tmp_rule = _TmpRule()
        tmp_rule.LHS = [(c_ids[j], class_ids[j]) for j in range(n_lhs)]
        tmp_rule.RHS = (c_ids[-1], class_ids[-1])
        tmp_rule.params = {
            "t1": best_params["t1"],
            "delta": best_params["delta"],
            "lambda": best_params["lambda"],
        }
        if best_params.get("predict_from_t1_only"):
            tmp_rule.params["predict_from_t1_only"] = True

        n_rows = n_attr
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, 2 * n_rows))
        if n_rows == 1:
            axes = [axes]
        title_parts = [
            f"avg_err = {avg_err:.6g}",
            f"epsilon = {best_epsilon:.6g}",
        ]
        if min_epsilon_rate_need is not None:
            if np.isfinite(min_epsilon_rate_need):
                title_parts.append(f"min_eps_rate = {min_epsilon_rate_need:.6g}")
            else:
                title_parts.append("min_eps_rate = inf")
        if eps_rate_stats is not None:
            title_parts.append(
                "eps_q: "
                f"max={eps_rate_stats.get('max', float('inf')):.4g},"
                f"q90={eps_rate_stats.get('q90', float('inf')):.4g},"
                f"q80={eps_rate_stats.get('q80', float('inf')):.4g},"
                f"q70={eps_rate_stats.get('q70', float('inf')):.4g},"
                f"q60={eps_rate_stats.get('q60', float('inf')):.4g},"
                f"q50={eps_rate_stats.get('q50', float('inf')):.4g}"
            )
        fig.suptitle("   |   ".join(title_parts), fontsize=10)
        x = np.arange(T)

        def _annotate_red_range(ax, start, end, y_series):
            try:
                start_i, end_i = int(start), int(end)
            except Exception:
                return
            if end_i <= start_i or start_i < 0 or start_i >= len(y_series):
                return
            y0 = y_series[start_i]
            if not np.isfinite(y0):
                return
            ax.annotate(
                f"({start_i},{end_i})",
                xy=(start_i, y0),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7,
                color="red",
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
            )

        all_lhs_indices = [list(comb[:n_lhs]) for comb in best_match]
        all_rhs_indices = [comb[-1] for comb in best_match]

        for j in range(n_attr):
            ax = axes[j]
            c_id = c_ids[j]
            cls = class_ids[j]
            series = db[:, c_id]
            ax.plot(x, series, "b-", lw=1, label="series")

            sqset = self.md.G[c_id][cls]
            if j < n_lhs:
                for k, lhs_indices in enumerate(all_lhs_indices):
                    sq_idx = lhs_indices[j]
                    if 0 <= sq_idx < len(sqset.SubSq_obj_list):
                        s = sqset.SubSq_obj_list[sq_idx]
                        start = s.t1
                        end = s.t1 + s.delta
                        if 0 <= start < end <= T:
                            ax.plot(
                                x[start:end],
                                series[start:end],
                                "r-",
                                lw=2,
                                alpha=0.7,
                                label="LHS subseq" if (j == 0 and k == 0) else None,
                            )
                            _annotate_red_range(ax, start, end, series)
                ax.set_title(f"Attr {c_id}, class {cls} (LHS)", fontsize=9)
            else:
                for k, (lhs_indices, rhs_index) in enumerate(zip(all_lhs_indices, all_rhs_indices)):
                    if 0 <= rhs_index < len(sqset.SubSq_obj_list):
                        s_rhs = sqset.SubSq_obj_list[rhs_index]
                        r_start = s_rhs.t1
                        r_end = s_rhs.t1 + s_rhs.delta
                        if 0 <= r_start < r_end <= T:
                            ax.plot(
                                x[r_start:r_end],
                                series[r_start:r_end],
                                "r-",
                                lw=2,
                                alpha=0.7,
                                label="RHS true" if k == 0 else None,
                            )
                            _annotate_red_range(ax, r_start, r_end, series)

                    db_rhs = series
                    valid, t_2, delta_2, lambda_2, S_pred = predict_rhs_from_lhs_combination(
                        self.md, tmp_rule, lhs_indices, db_rhs
                    )
                    if valid and S_pred is not None:
                        p_start = t_2
                        p_end = t_2 + len(S_pred)
                        p_start_clip = max(0, p_start)
                        p_end_clip = min(T, p_end)
                        if p_start_clip < p_end_clip:
                            rel0 = p_start_clip - p_start
                            rel1 = rel0 + (p_end_clip - p_start_clip)
                            ax.plot(
                                x[p_start_clip:p_end_clip],
                                S_pred[rel0:rel1],
                                "g-",
                                lw=2,
                                alpha=0.7,
                                label="RHS predicted" if k == 0 else None,
                            )
                ax.set_title(f"Attr {c_id}, class {cls} (RHS)", fontsize=9)

            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time index")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs("results/temp_valid_rules", exist_ok=True)
        name_parts = [f"({c_ids[i]}_{class_ids[i]})" for i in range(len(c_ids))]
        fname = self._prefixed_temp_rule_plot_filename("_".join(name_parts))
        save_path = os.path.join("results/temp_valid_rules", fname)
        plt.savefig(save_path, dpi=150)


    def find_all_best_motif_match3(self, rhs_attribute_list=None, n_workers=4):
        if rhs_attribute_list is None:
            c_id_pairs = [(i, j) for i in self.md.G.keys() for j in self.md.G.keys() if i != j]
        else:
            c_id_pairs = [(i, j) for i in self.md.G.keys() for j in self.md.G.keys() if i != j and j in rhs_attribute_list]
        self.motif_pairs = {}
        print("number of motif pairs for each attribute:")
        for c_id in self.md.G.keys():
            print(c_id, " : ", len(self.md.G[c_id]))
        t_start = time()

        tasks = []
        for c_ids in c_id_pairs:
            c_id1, c_id2 = c_ids
            for class_id1 in range(len(self.md.G[c_id1])):
                min_lhs = get_min_regression_samples(1, use_t1_only=True)
                if len(self.md.G[c_id1][class_id1].SubSq_obj_list) <= min_lhs:
                    continue
                for class_id2 in range(len(self.md.G[c_id2])):
                    if len(self.md.G[c_id2][class_id2].SubSq_obj_list) / len(self.md.G[c_id1][class_id1].SubSq_obj_list) < self.supp_threshold:
                        continue
                    key = ((c_id1, class_id1), (c_id2, class_id2))
                    tasks.append((key, c_ids, (class_id1, class_id2)))
        for c_id in self.md.G.keys():
            if rhs_attribute_list is not None and c_id not in rhs_attribute_list:
                continue
            for class_id1 in range(len(self.md.G[c_id])):
                min_lhs = get_min_regression_samples(1, use_t1_only=True)
                if len(self.md.G[c_id][class_id1].SubSq_obj_list) <= min_lhs:
                    continue
                for class_id2 in range(len(self.md.G[c_id])):
                    if class_id1 == class_id2:
                        continue
                    if len(self.md.G[c_id][class_id2].SubSq_obj_list) / len(self.md.G[c_id][class_id1].SubSq_obj_list) < self.supp_threshold:
                        continue
                    key = ((c_id, class_id1), (c_id, class_id2))
                    tasks.append((key, (c_id, c_id), (class_id1, class_id2)))

        total = len(tasks)
        if total == 0:
            print("No motif pairs to process.")
            t_end = time()
            print("mining time:", round(t_end - t_start))
            return
        rd_kwargs = dict(
            supp_threshold=self.supp_threshold,
            max_iter=self.max_iter,
            is_plot=False,
            is_print=False,
            para_model_type=self.para_model_type,
            confidence_threshold=self.confidence_threshold,
        )
        with Pool(
            processes=n_workers,
            initializer=_init_worker_motif_match,
            initargs=(self.md, rd_kwargs),
        ) as pool:
            for done_count, (key, temp) in enumerate(pool.imap_unordered(_process_one_motif_pair, tasks), 1):
                self.motif_pairs[key] = temp
                print(f"Processed {done_count} / {total} (current key={key})")
        t_end = time()
        
        out_dir = os.path.join("results", "obj_motif_pair")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{self.md.dataname}.pkl")
        save_instance(self.motif_pairs, save_path)
        print(f"motif_pairs saved to: {save_path}")
        print("mining time:", round(t_end - t_start))

    def compute_valid_pair5(
        self,
        c_ids,
        class_ids,
        sq_sets,
        para,
        t1_col_fit,
    ):
        n_attr = len(sq_sets)
        n_lhs = n_attr - 1
        n_anchor = len(sq_sets[0])

        db_rhs = np.asarray(self.md.db[:, c_ids[-1]], dtype=np.float64)
        n_rhs = len(db_rhs)

        rhs_t1 = np.array([s.t1 for s in sq_sets[-1]], dtype=np.float64)
        rhs_delta = np.array([s.delta for s in sq_sets[-1]], dtype=np.float64)

        lhs_all_t1 = [np.array([s.t1 for s in sq_sets[j]], dtype=np.float64) for j in range(n_lhs)]
        lhs_all_delta = [np.array([s.delta for s in sq_sets[j]], dtype=np.float64) for j in range(n_lhs)]
        lhs_all_lambda = [np.array([s.lambda_range[0] for s in sq_sets[j]], dtype=np.float64) for j in range(n_lhs)]

        coef_t1, _     = para["t1"]
        coef_delta, _  = para["delta"]
        coef_lambda, _ = para["lambda"]

        ta=perf_counter()
        if n_lhs > 1:
            j_range = list(range(1, n_attr - 1))
            lhs_t1_arrays    = [lhs_all_t1[j]    for j in range(1, n_lhs)]
            lhs_delta_arrays = [lhs_all_delta[j] for j in range(1, n_lhs)]
            t1_combos = []

            for i0, s1 in enumerate(sq_sets[0]):
                s1_t1 = s1.t1
                matched = {}
                skip = False

                for idx, j in enumerate(j_range):
                    if j not in t1_col_fit:
                        skip = True
                        break

                    para_j = t1_col_fit[j][0]
                    tj = linear_predict(s1_t1, para_j)

                    arr_t1 = lhs_t1_arrays[idx]
                    arr_delta = lhs_delta_arrays[idx]

                    overlap_start = np.maximum(tj, arr_t1)
                    overlap_end = np.minimum(tj + arr_delta, arr_t1 + arr_delta)
                    overlap_len = overlap_end - overlap_start

                    best_idx = int(np.argmax(overlap_len))
                    if overlap_len[best_idx] <= 0:
                        skip = True
                        break

                    matched[j] = best_idx

                if not skip and len(matched) == len(j_range):
                    t1_combos.append([i0] + [matched[j] for j in j_range])
        else:
            t1_combos = [[i0] for i0 in range(n_anchor)]


        num_lhs = len(t1_combos)
        if num_lhs < get_min_regression_samples(len(c_ids)-1):
            return [], num_lhs, {}, float("inf"), float("inf")
        tb=perf_counter()


        param, kind = self.md.G[c_ids[-1]][class_ids[-1]].f
        predict_fn = select_predict_nb(kind)

        if kind != "piecewise":
            param = np.asarray(param, dtype=np.float64)
        else:
            breaks = np.asarray(param["breaks"], dtype=np.float64)
            slopes = np.asarray(param["slopes"], dtype=np.float64)
            intercepts = np.asarray(param["intercepts"], dtype=np.float64)

        window_rate = self.md.window_rate



        X_one = np.empty(3 * n_lhs, dtype=np.float64)
        x_norm_cache = {}
        pred_cache = []

        valid_pairs_raw = []
        weights_raw = []
        tc=perf_counter()

        X_one = np.empty(3 * n_lhs, dtype=np.float64)
        x_norm_cache = {}
        pred_cache_dict = {}

        for combo in t1_combos:

            if n_lhs == 1:
                idx0 = combo[0]
                X_one[0] = lhs_all_t1[0][idx0]
                X_one[1] = lhs_all_delta[0][idx0]
                X_one[2] = lhs_all_lambda[0][idx0]
            else:
                for j in range(n_lhs):
                    idx = combo[j]
                    X_one[j] = lhs_all_t1[j][idx]
                    X_one[n_lhs + j] = lhs_all_delta[j][idx]
                    X_one[2 * n_lhs + j] = lhs_all_lambda[j][idx]

            t_2     = int(round(multi_linear_predict(X_one, coef_t1)))
            delta_2 = int(round(multi_linear_predict(X_one, coef_delta)))
            lambda_2=       multi_linear_predict(X_one, coef_lambda)
            

            if t_2 < 0 or delta_2 <= 0 or t_2 + delta_2 > n_rhs:
                continue

            x_norm = x_norm_cache.get(delta_2)
            if x_norm is None:
                if delta_2 > 1:
                    x_norm = np.arange(delta_2, dtype=np.float64) / (delta_2 - 1)
                else:
                    x_norm = np.zeros(1, dtype=np.float64)
                x_norm_cache[delta_2] = x_norm

            pred_key = (delta_2, lambda_2, t_2)
            S_pred = pred_cache_dict.get(pred_key)
            if S_pred is None:
                if kind != "piecewise":
                    S_pred = predict_fn(x_norm, param) * lambda_2 + db_rhs[t_2]
                else:
                    S_pred = predict_fn(
                        x_norm,
                        breaks=breaks,
                        slopes=slopes,
                        intercepts=intercepts
                    ) * lambda_2 + db_rhs[t_2]
                pred_cache_dict[pred_key] = S_pred

            end = t_2 + delta_2
            S_actual = db_rhs[t_2:end]

            min_len = min(len(S_pred), len(S_actual))
            if min_len == 0:
                continue

            if min_len > 1:
                window = max(0, min(int(delta_2 * window_rate), min_len - 1))
            else:
                window = 0

            err_vec = calc_error_local_alignment3(
                S_pred[:min_len],
                S_actual[:min_len],
                window=window,
                gamma_penalty=self.md.gamma_penalty
            )
            if err_vec is None:
                continue

            err_agg = float(np.max(err_vec)) if getattr(self.md, "error_type", "max") == "max" else float(np.mean(err_vec))

            seg = S_actual[:min_len]
            M_seg = float(np.max(seg) - np.min(seg)) if min_len > 0 else 0.0
            if M_seg <= 0:
                continue

            if err_agg <= self.epsilon_rate * M_seg:

                o_start = np.maximum(t_2, rhs_t1)
                o_end   = np.minimum(t_2 + delta_2, rhs_t1 + rhs_delta)
                o_len   = o_end - o_start

                best_k = int(np.argmax(o_len))
                best_overlap = o_len[best_k]

                if best_overlap > 0:
                    valid_pairs_raw.append(combo + [best_k])
                    weights_raw.append(best_overlap)
                    pred_cache.append((tuple(combo), int(t_2), int(delta_2), float(lambda_2)))
        if not valid_pairs_raw:
            return [], num_lhs, {}, float("inf"), float("inf")

        _, valid_pairs = ordered_dp_max_matching2_weighted(valid_pairs_raw, weights_raw)
        td=perf_counter()
        
        max_err = 0.0
        total_err_weighted = 0.0
        total_weight = 0.0

        if len(pred_cache) >= get_min_regression_samples(len(c_ids)-1) and len(pred_cache)/num_lhs >= self.supp_threshold:
            for combo, t_2, delta_2, lambda_2 in pred_cache:
                if t_2 < 0 or delta_2 <= 0 or t_2 + delta_2 > n_rhs:
                    continue

                x_norm = x_norm_cache.get(delta_2)
                if x_norm is None:
                    continue
                if kind != "piecewise":
                    S_pred = predict_fn(x_norm, param) * lambda_2 + db_rhs[t_2]
                else:
                    S_pred = predict_fn(x_norm, breaks=breaks, slopes=slopes, intercepts=intercepts) * lambda_2 + db_rhs[t_2]

                S_actual = db_rhs[t_2 : t_2 + delta_2]

                min_len = min(len(S_pred), len(S_actual))
                if min_len == 0:
                    continue

                window = max(0, min(int(delta_2 * window_rate), min_len - 1))
                err_vec = calc_error_local_alignment3(
                    S_pred[:min_len],
                    S_actual[:min_len],
                    window=window,
                    gamma_penalty=self.md.gamma_penalty,
                )
                if err_vec is None:
                    continue
                err_agg = float(np.max(err_vec))
                mean_err = float(np.mean(err_vec))

                if err_agg > max_err:
                    max_err = err_agg

                total_err_weighted += mean_err * min_len
                total_weight += min_len


        epsilon = max_err if max_err > 0 else float("inf")
        avg_err = total_err_weighted / total_weight if total_weight > 0 else float("inf")
        te=perf_counter()

        return valid_pairs, num_lhs, para, avg_err, epsilon


    def compute_valid_pair6(
        self,
        c_ids,
        class_ids,
        sq_sets,
        para,
        t1_col_fit,
    ):
        def _build_eps_rate_stats(ratios):
            if not ratios:
                return {
                    "max": float("inf"),
                    "q90": float("inf"),
                    "q80": float("inf"),
                    "q70": float("inf"),
                    "q60": float("inf"),
                    "q50": float("inf"),
                }
            arr = np.asarray(ratios, dtype=np.float64)
            return {
                "max": float(np.max(arr)),
                "q90": float(np.quantile(arr, 0.9)),
                "q80": float(np.quantile(arr, 0.8)),
                "q70": float(np.quantile(arr, 0.7)),
                "q60": float(np.quantile(arr, 0.6)),
                "q50": float(np.quantile(arr, 0.5)),
            }

        n_attr = len(sq_sets)
        n_lhs = n_attr - 1
        n_anchor = len(sq_sets[0])

        db_rhs = np.asarray(self.md.db[:, c_ids[-1]], dtype=np.float64)
        n_rhs = len(db_rhs)

        rhs_t1 = np.array([s.t1 for s in sq_sets[-1]], dtype=np.float64)
        rhs_delta = np.array([s.delta for s in sq_sets[-1]], dtype=np.float64)

        lhs_all_t1 = [np.array([s.t1 for s in sq_sets[j]], dtype=np.float64) for j in range(n_lhs)]
        lhs_all_delta = [np.array([s.delta for s in sq_sets[j]], dtype=np.float64) for j in range(n_lhs)]
        lhs_all_lambda = [np.array([s.lambda_range[0] for s in sq_sets[j]], dtype=np.float64) for j in range(n_lhs)]

        coef_t1, _     = para["t1"]
        coef_delta, _  = para["delta"]
        coef_lambda, _ = para["lambda"]

        ta=perf_counter()
        if n_lhs > 1:
            j_range = list(range(1, n_attr - 1))
            lhs_t1_arrays    = [lhs_all_t1[j]    for j in range(1, n_lhs)]
            lhs_delta_arrays = [lhs_all_delta[j] for j in range(1, n_lhs)]
            t1_combos = []

            for i0, s1 in enumerate(sq_sets[0]):
                s1_t1 = s1.t1
                matched = {}
                skip = False

                for idx, j in enumerate(j_range):
                    if j not in t1_col_fit:
                        skip = True
                        break

                    para_j = t1_col_fit[j][0]
                    tj = linear_predict(s1_t1, para_j)

                    arr_t1 = lhs_t1_arrays[idx]
                    arr_delta = lhs_delta_arrays[idx]

                    overlap_start = np.maximum(tj, arr_t1)
                    overlap_end = np.minimum(tj + arr_delta, arr_t1 + arr_delta)
                    overlap_len = overlap_end - overlap_start

                    best_idx = int(np.argmax(overlap_len))
                    if overlap_len[best_idx] <= 0:
                        skip = True
                        break

                    matched[j] = best_idx

                if not skip and len(matched) == len(j_range):
                    t1_combos.append([i0] + [matched[j] for j in j_range])
        else:
            t1_combos = [[i0] for i0 in range(n_anchor)]


        num_lhs = len(t1_combos)
        if num_lhs < get_min_regression_samples(len(c_ids)-1):
            return [], num_lhs, {}, float("inf"), float("inf"), float("inf"), _build_eps_rate_stats([])
        tb=perf_counter()


        param, kind = self.md.G[c_ids[-1]][class_ids[-1]].f
        predict_fn = select_predict_nb(kind)

        if kind != "piecewise":
            param = np.asarray(param, dtype=np.float64)
        else:
            breaks = np.asarray(param["breaks"], dtype=np.float64)
            slopes = np.asarray(param["slopes"], dtype=np.float64)
            intercepts = np.asarray(param["intercepts"], dtype=np.float64)

        window_rate = self.md.window_rate



        X_one = np.empty(3 * n_lhs, dtype=np.float64)
        x_norm_cache = {}
        pred_cache = []

        valid_pairs_raw = []
        weights_raw = []
        pair_to_ratio = {}
        all_pairs_raw = []
        all_weights_raw = []
        all_pair_to_ratio = {}
        tc=perf_counter()

        X_one = np.empty(3 * n_lhs, dtype=np.float64)
        x_norm_cache = {}
        pred_cache_dict = {}

        for combo in t1_combos:

            if n_lhs == 1:
                idx0 = combo[0]
                X_one[0] = lhs_all_t1[0][idx0]
                X_one[1] = lhs_all_delta[0][idx0]
                X_one[2] = lhs_all_lambda[0][idx0]
            else:
                for j in range(n_lhs):
                    idx = combo[j]
                    X_one[j] = lhs_all_t1[j][idx]
                    X_one[n_lhs + j] = lhs_all_delta[j][idx]
                    X_one[2 * n_lhs + j] = lhs_all_lambda[j][idx]

            t_2     = int(round(multi_linear_predict(X_one, coef_t1)))
            delta_2 = int(round(multi_linear_predict(X_one, coef_delta)))
            lambda_2=       multi_linear_predict(X_one, coef_lambda)
            

            if t_2 < 0 or delta_2 <= 0 or t_2 + delta_2 > n_rhs:
                continue

            x_norm = x_norm_cache.get(delta_2)
            if x_norm is None:
                if delta_2 > 1:
                    x_norm = np.arange(delta_2, dtype=np.float64) / (delta_2 - 1)
                else:
                    x_norm = np.zeros(1, dtype=np.float64)
                x_norm_cache[delta_2] = x_norm

            pred_key = (delta_2, lambda_2, t_2)
            S_pred = pred_cache_dict.get(pred_key)
            if S_pred is None:
                if kind != "piecewise":
                    S_pred = predict_fn(x_norm, param) * lambda_2 + db_rhs[t_2]
                else:
                    S_pred = predict_fn(
                        x_norm,
                        breaks=breaks,
                        slopes=slopes,
                        intercepts=intercepts
                    ) * lambda_2 + db_rhs[t_2]
                pred_cache_dict[pred_key] = S_pred

            end = t_2 + delta_2
            S_actual = db_rhs[t_2:end]

            min_len = min(len(S_pred), len(S_actual))
            if min_len == 0:
                continue

            if min_len > 1:
                window = max(0, min(int(delta_2 * window_rate), min_len - 1))
            else:
                window = 0

            err_vec = calc_error_local_alignment3(
                S_pred[:min_len],
                S_actual[:min_len],
                window=window,
                gamma_penalty=self.md.gamma_penalty
            )
            if err_vec is None:
                continue

            err_agg = float(np.max(err_vec)) if getattr(self.md, "error_type", "max") == "max" else float(np.mean(err_vec))

            seg = S_actual[:min_len]
            M_seg = float(np.max(seg) - np.min(seg)) if min_len > 0 else 0.0
            if M_seg <= 0:
                continue

            o_start = np.maximum(t_2, rhs_t1)
            o_end   = np.minimum(t_2 + delta_2, rhs_t1 + rhs_delta)
            o_len   = o_end - o_start

            best_k = int(np.argmax(o_len))
            best_overlap = o_len[best_k]
            ratio = err_agg / M_seg

            if best_overlap > 0:
                pair_key = tuple(combo + [best_k])
                all_pairs_raw.append(combo + [best_k])
                all_weights_raw.append(best_overlap)
                all_pair_to_ratio[pair_key] = ratio

            if err_agg <= self.epsilon_rate * M_seg and best_overlap > 0:
                pair_key = tuple(combo + [best_k])
                valid_pairs_raw.append(combo + [best_k])
                weights_raw.append(best_overlap)
                pair_to_ratio[pair_key] = ratio
                pred_cache.append((tuple(combo), int(t_2), int(delta_2), float(lambda_2)))
        if not all_pairs_raw:
            return [], num_lhs, para, float("inf"), float("inf"), float("inf"), _build_eps_rate_stats([])

        if valid_pairs_raw:
            _, valid_pairs = ordered_dp_max_matching2_weighted(valid_pairs_raw, weights_raw)
            eps_rate_need = max(pair_to_ratio[tuple(p)] for p in valid_pairs) if valid_pairs else float("inf")
            eps_rate_stats = _build_eps_rate_stats([pair_to_ratio[tuple(p)] for p in valid_pairs] if valid_pairs else [])
        else:
            _, valid_pairs = ordered_dp_max_matching2_weighted(all_pairs_raw, all_weights_raw)
            eps_rate_need = max(all_pair_to_ratio[tuple(p)] for p in valid_pairs) if valid_pairs else float("inf")
            eps_rate_stats = _build_eps_rate_stats([all_pair_to_ratio[tuple(p)] for p in valid_pairs] if valid_pairs else [])
        td=perf_counter()
        
        max_err = 0.0
        total_err_weighted = 0.0
        total_weight = 0.0

        if len(pred_cache) >= get_min_regression_samples(len(c_ids)-1) and len(pred_cache)/num_lhs >= self.supp_threshold:
            for combo, t_2, delta_2, lambda_2 in pred_cache:
                if t_2 < 0 or delta_2 <= 0 or t_2 + delta_2 > n_rhs:
                    continue

                x_norm = x_norm_cache.get(delta_2)
                if x_norm is None:
                    continue
                if kind != "piecewise":
                    S_pred = predict_fn(x_norm, param) * lambda_2 + db_rhs[t_2]
                else:
                    S_pred = predict_fn(x_norm, breaks=breaks, slopes=slopes, intercepts=intercepts) * lambda_2 + db_rhs[t_2]

                S_actual = db_rhs[t_2 : t_2 + delta_2]

                min_len = min(len(S_pred), len(S_actual))
                if min_len == 0:
                    continue

                window = max(0, min(int(delta_2 * window_rate), min_len - 1))
                err_vec = calc_error_local_alignment3(
                    S_pred[:min_len],
                    S_actual[:min_len],
                    window=window,
                    gamma_penalty=self.md.gamma_penalty,
                )
                if err_vec is None:
                    continue
                err_agg = float(np.max(err_vec))
                mean_err = float(np.mean(err_vec))

                if err_agg > max_err:
                    max_err = err_agg

                total_err_weighted += mean_err * min_len
                total_weight += min_len


        epsilon = max_err if max_err > 0 else float("inf")
        avg_err = total_err_weighted / total_weight if total_weight > 0 else float("inf")
        te=perf_counter()

        return valid_pairs, num_lhs, para, avg_err, epsilon, eps_rate_need, eps_rate_stats






    def find_t1_par_co_occurrence(self, mp):
        G=self.md.G
        for c_id in G.keys():
            for sqset in G[c_id]:
                sqset.t1_co_occurrence = set()
                sqset.par_co_occurrence = set()

        for pair in mp.keys():
            pair_obj = mp[pair]
            c_id1, class_id1 = pair[0]
            c_id2, class_id2 = pair[1]
            
            min_t1 = get_min_regression_samples(1, use_t1_only=True)
            if pair_obj.C_phi_t >= min_t1:
                G[c_id1][class_id1].t1_co_occurrence.add((c_id2, class_id2))   
                
            
            min_all = get_min_regression_samples(1, use_t1_only=False)
            if pair_obj.C_phi >= min_all and pair_obj.supp >= self.supp_threshold:
                G[c_id2][class_id2].par_co_occurrence.add((c_id1, class_id1))

        for c_id in G.keys():
            for sqset in G[c_id]:
                sqset.t1_co_occurrence = sorted(
                    sqset.t1_co_occurrence, key=lambda x: (x[0], x[1])
                )
                sqset.par_co_occurrence = sorted(
                    sqset.par_co_occurrence, key=lambda x: (x[0], x[1])
                )
                sqset.par_co_occurrence_set = set(sqset.par_co_occurrence)

    def find_all_valid_lhs_for_rhs(self, RHS, all_motif_pairs,mp):
        self.valid_rules = []
        self.NodeList = []
        self._node_id_counter = [0]
        c_id_r, class_id_r = RHS
        sqset = self.md.G[c_id_r][class_id_r]
        initial = all_motif_pairs
        ignore = [RHS]
        id = 0
        for m in initial:
            id += 1
            par_set = getattr(self.md.G[RHS[0]][RHS[1]], 'par_co_occurrence_set', None)
            par_coll = self.md.G[RHS[0]][RHS[1]].par_co_occurrence

            if par_set is not None:
                rhs_not_in_par = m not in par_set
            else:
                rhs_not_in_par = m not in par_coll
            if rhs_not_in_par:
                self.node_processing(LHS=[m], RHS=RHS, Father=-1, ignore=ignore + [m])
            
            else:
                temp = mp[(m,RHS)]
                C_phi, supp, params,lhs_combinations,avg_err,epsilon = temp.C_phi, temp.supp, temp.params, temp.best_lhs_combinations, temp.avg_err, temp.epsilon
                self.valid_rules.append(ValidRule([m,], RHS, C_phi, supp, params, lhs_combinations, avg_err, epsilon))    
                print("[0.Valid rule]:", [m, RHS])
        return self.valid_rules


    def node_processing(self, LHS, RHS, Father, ignore):
        if RHS in LHS:
            print("[Failed node]:", LHS + [RHS], "RHS in LHS")
            return
        self._node_id_counter[0] += 1
        current_node_id = self._node_id_counter[0]
        self.NodeList.append(Node(LHS, RHS, Father, ignore, current_node_id))

        c_ids = [L[0] for L in LHS] + [RHS[0]]
        class_ids = [L[1] for L in LHS] + [RHS[1]]
        po = self.mp.get((LHS[0], RHS)) if len(LHS) == 1 else None
        if po is not None:
            C_phi, best_match, supp = po.C_phi, po.best_match, po.supp
            params, best_lhs_combinations, avg_err = po.params, po.best_lhs_combinations, po.avg_err
            epsilon = getattr(po, "epsilon", float("inf"))
        elif len(LHS)==1 and po is None:
            print("[Failed node]:", LHS + [RHS], "LHS is single, but not in mp")
            return
        else:
            C_phi, best_match, supp, params, best_lhs_combinations, avg_err, epsilon = self.find_best_motif_match_linear_ransac6(
                c_ids, class_ids, is_plot=self.is_plot
            )




        if C_phi >= get_min_regression_samples(len(LHS)) and supp >= self.supp_threshold:
            self.valid_rules.append(ValidRule(LHS, RHS, C_phi, supp, params, best_lhs_combinations, avg_err,epsilon))
            print("[1.Valid rule]:", LHS + [RHS], "C_phi=", C_phi, "supp=", supp)
            return
        num_lhs = C_phi / supp if supp > 0 else 0
        if num_lhs < get_min_regression_samples(len(LHS)):
            print("[3.Failed node]:", LHS + [RHS], "C_phi=", C_phi, "supp=", supp, f"End with insufficient lhs_combinations.          supp_threshold:{self.supp_threshold}; C_phi_threshold: {get_min_regression_samples(len(LHS))}")
            return
        if num_lhs >= get_min_regression_samples(len(LHS)) and supp < self.supp_threshold:
            max_lhs_len = getattr(self, "max_lhs_len", None)
            
            if max_lhs_len is not None and len(LHS) >= max_lhs_len:
                print("[2-0.Failed node]:", LHS + [RHS], "C_phi=", C_phi, "supp=", supp, f"End with max_lhs_len={max_lhs_len}")
                return
            anchor = LHS[0]
            sqset = self.md.G[anchor[0]][anchor[1]]
            if hasattr(sqset, 't1_co_occurrence'):
                ignore_set = set(ignore)
                m_set = [x for x in sqset.t1_co_occurrence if x not in ignore_set]
                if len(m_set) == 0:
                    print("[2-1.Failed node]:", LHS + [RHS], "C_phi=", C_phi, "supp=", supp, f"End with no available LHS.   supp_threshold:{self.supp_threshold}; C_phi_threshold: {get_min_regression_samples(len(LHS))}")
                    return
                if C_phi < get_min_regression_samples(len(LHS)+1):
                    print("[2-2.Failed node]:", LHS + [RHS], "C_phi=", C_phi, "supp=", supp, f"End with insufficient lhs_combinations.   supp_threshold:{self.supp_threshold}; C_phi_threshold: {get_min_regression_samples(len(LHS)+1)}")
                    return
                temp = []
                for m in m_set:
                    temp.append(m)
                    self.node_processing(
                        LHS=LHS + [m],
                        RHS=RHS,
                        Father=current_node_id,
                        ignore=ignore + temp,
                    )
        

 







def ordered_dp_max_matching2(valid_pairs):
    n = len(valid_pairs)
    if n == 0:
        return 0, []

    def strictly_less(u, v):
        return all(u[k] < v[k] for k in range(len(u)))

    dp = [1] * n
    parent = [-1] * n

    for i in range(n):
        for j in range(i):
            if strictly_less(valid_pairs[j], valid_pairs[i]) and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    best_idx = max(range(n), key=lambda i: dp[i])
    best_len = dp[best_idx]
    match = []
    cur = best_idx
    while cur != -1:
        match.append(valid_pairs[cur])
        cur = parent[cur]
    match.reverse()
    return best_len, match


def ordered_dp_max_matching2_weighted(valid_pairs, weights):
    n = len(valid_pairs)
    if n == 0:
        return 0.0, []
    if len(weights) != n:
        raise ValueError("valid_pairs and weights must have same length")

    def strictly_less(u, v):
        return all(u[k] < v[k] for k in range(len(u)))

    dp = [float(w) for w in weights]
    parent = [-1] * n

    for i in range(n):
        for j in range(i):
            if strictly_less(valid_pairs[j], valid_pairs[i]) and dp[j] + weights[i] > dp[i]:
                dp[i] = dp[j] + weights[i]
                parent[i] = j

    best_idx = max(range(n), key=lambda i: dp[i])
    total_weight = dp[best_idx]
    match = []
    cur = best_idx
    while cur != -1:
        match.append(valid_pairs[cur])
        cur = parent[cur]
    match.reverse()
    return total_weight, match




_md = None
_mp = None
_all_motif_pairs = None
_rd_kwargs = None

_md_mm = None
_rd_kwargs_mm = None


def _init_worker_motif_match(md_inst, rd_kwargs):
    global _md_mm, _rd_kwargs_mm
    _md_mm = md_inst
    _rd_kwargs_mm = rd_kwargs


def _process_one_motif_pair(task):
    key, c_ids, class_ids = task
    t0 = time()
    rd_local = RuleDiscovery(_md_mm, **_rd_kwargs_mm)
    temp = Motif_pair(c_ids, class_ids)
    
    temp.best_score, temp.best_match, temp.supp, temp.params, temp.best_lhs_combinations, temp.avg_err, temp.epsilon = rd_local.find_best_motif_match_linear_ransac6(
        c_ids, class_ids, is_plot=rd_local.is_plot
    )
    temp.C_phi = temp.best_score
    c_id1, c_id2 = c_ids
    n_lhs = len(_md_mm.G[c_id1][class_ids[0]].SubSq_obj_list)
    temp.supp = temp.best_score / n_lhs if n_lhs > 0 else 0
    temp.C_phi_t = temp.best_score
    temp.best_match_t = temp.best_match
    t1 = time()
    print(f"[Worker PID {os.getpid()}] {key} time={round(t1 - t0, 3)} s")
    return (key, temp)


def _process_one_motif_pair2(task):
    key, c_ids, class_ids = task
    t0 = time()
    rd_local = RuleDiscovery(_md_mm, **_rd_kwargs_mm)
    temp = Motif_pair(c_ids, class_ids)
    

    temp.best_score, temp.best_match, temp.supp, temp.params, temp.best_lhs_combinations, temp.avg_err, temp.epsilon, best_min_eps_rate = rd_local.find_best_motif_match_linear_ransac7(
        c_ids, class_ids, is_plot=rd_local.is_plot
    )
    temp.C_phi = temp.best_score
    c_id1, c_id2 = c_ids
    n_lhs = len(_md_mm.G[c_id1][class_ids[0]].SubSq_obj_list)
    temp.supp = temp.best_score / n_lhs if n_lhs > 0 else 0
    temp.C_phi_t = temp.best_score
    temp.best_match_t = temp.best_match
    t1 = time()
    print(f"[Worker PID {os.getpid()}] {key} time={round(t1 - t0, 3)} s")
    return (key, temp)


def find_all_best_motif_match5(
    md,
    *,
    n_workers_pairs,
    rhs_attribute_list=None,
    supp_threshold=0.79,
    epsilon_rate=1.0,
    is_plot=False,
    parallel_pairs=True,
    plot_save_prefix=None,
    use_ransac7=True,
    ransac7_discard_match_if_min_eps_rate_gt_budget=False,
):
    if rhs_attribute_list is None:
        c_id_pairs = [(i, j) for i in md.G.keys() for j in md.G.keys() if i != j]
    else:
        c_id_pairs = [
            (i, j)
            for i in md.G.keys()
            for j in md.G.keys()
            if i != j and j in rhs_attribute_list
        ]

    motif_pairs = {}
    rd = RuleDiscovery(
        md,
        supp_threshold=supp_threshold,
        is_plot=is_plot,
        is_print=False,
        epsilon_rate=epsilon_rate,
        plot_save_prefix=plot_save_prefix,
        ransac7_discard_match_if_min_eps_rate_gt_budget=ransac7_discard_match_if_min_eps_rate_gt_budget,
    )
    rd_kwargs = dict(
        supp_threshold=rd.supp_threshold,
        max_iter=rd.max_iter,
        is_plot=False,
        is_print=False,
        para_model_type=rd.para_model_type,
        confidence_threshold=rd.confidence_threshold,
        plot_save_prefix=plot_save_prefix,
        ransac7_discard_match_if_min_eps_rate_gt_budget=ransac7_discard_match_if_min_eps_rate_gt_budget,
    )

    tasks = []
    for c_ids in c_id_pairs:
        c_id1, c_id2 = c_ids
        for class_id1 in range(len(md.G[c_id1])):
            min_lhs = get_min_regression_samples(1, use_t1_only=True)
            if len(md.G[c_id1][class_id1].SubSq_obj_list) <= min_lhs:
                continue
            for class_id2 in range(len(md.G[c_id2])):
                if (
                    len(md.G[c_id2][class_id2].SubSq_obj_list)
                    / len(md.G[c_id1][class_id1].SubSq_obj_list)
                    < rd.supp_threshold
                ):
                    continue
                key = ((c_id1, class_id1), (c_id2, class_id2))
                tasks.append((key, c_ids, (class_id1, class_id2)))
    for c_id in md.G.keys():
        if rhs_attribute_list is not None and c_id not in rhs_attribute_list:
            continue
        for class_id1 in range(len(md.G[c_id])):
            min_lhs = get_min_regression_samples(1, use_t1_only=True)
            if len(md.G[c_id][class_id1].SubSq_obj_list) <= min_lhs:
                continue
            for class_id2 in range(len(md.G[c_id])):
                if class_id1 == class_id2:
                    continue
                if (
                    len(md.G[c_id][class_id2].SubSq_obj_list)
                    / len(md.G[c_id][class_id1].SubSq_obj_list)
                    < rd.supp_threshold
                ):
                    continue
                key = ((c_id, class_id1), (c_id, class_id2))
                tasks.append((key, (c_id, c_id), (class_id1, class_id2)))

    if len(tasks) == 0:
        return motif_pairs

    process_one = _process_one_motif_pair2 if use_ransac7 else _process_one_motif_pair
    if parallel_pairs:
        with Pool(
            processes=n_workers_pairs,
            initializer=_init_worker_motif_match,
            initargs=(md, rd_kwargs),
        ) as pool:
            for key, temp in pool.imap_unordered(process_one, tasks):
                motif_pairs[key] = temp
    else:
        _init_worker_motif_match(md, rd_kwargs)
        for task in tasks:
            key, temp = process_one(task)
            motif_pairs[key] = temp
    return motif_pairs


def _task13_worker_one_classification_pkl(args):
    (
        pkl_in,
        pkl_out,
        n_workers_pairs,
        epsilon_rate,
        supp_threshold,
        parallel_pairs,
        use_ransac7,
        ransac7_discard_match_if_min_eps_rate_gt_budget,
    ) = args
    md = load_instance(pkl_in)
    n_sets = len(md.G.get(0, []))
    print(
        f"[task13] {os.path.basename(pkl_in)}  md.G[0] num_subseq_sets={n_sets}"
    )
    pkl_stem = os.path.splitext(os.path.basename(pkl_in))[0]
    mp = find_all_best_motif_match5(
        md,
        n_workers_pairs=n_workers_pairs,
        rhs_attribute_list=[0],
        supp_threshold=supp_threshold,
        epsilon_rate=epsilon_rate,
        is_plot=False,
        parallel_pairs=parallel_pairs,
        plot_save_prefix=pkl_stem,
        use_ransac7=use_ransac7,
        ransac7_discard_match_if_min_eps_rate_gt_budget=ransac7_discard_match_if_min_eps_rate_gt_budget,
    )
    os.makedirs(os.path.dirname(pkl_out), exist_ok=True)
    save_instance(mp, pkl_out)
    return pkl_out


def run_classification_motif_pairs(
    dataset_name,
    *,
    project_root=None,
    n_workers_pkl=6,
    n_workers_pairs=8,
    epsilon_rate=1.0,
    supp_threshold=0.79,
    clear_output_dir=False,
    parallel_pairs_inside_pkl=False,
    use_ransac7=True,
    ransac7_discard_match_if_min_eps_rate_gt_budget=True,
):
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    motif_res_dir = os.path.join(
        root, "results", "classification", dataset_name, "motif_res"
    )
    out_dir = os.path.join(
        root, "results", "classification", dataset_name, "motif_pair_res"
    )
    pkls = sorted(glob.glob(os.path.join(motif_res_dir, "*.pkl")))
    if not pkls:
        print(f"[task13] no pkl under {motif_res_dir}")
        return []

    if clear_output_dir and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    worker_args = []
    for pkl_in in pkls:
        base = os.path.splitext(os.path.basename(pkl_in))[0]
        pkl_out = os.path.join(out_dir, base + ".pkl")
        worker_args.append(
            (
                pkl_in,
                pkl_out,
                n_workers_pairs,
                epsilon_rate,
                supp_threshold,
                parallel_pairs_inside_pkl,
                use_ransac7,
                ransac7_discard_match_if_min_eps_rate_gt_budget,
            )
        )

    done = []
    with Pool(processes=n_workers_pkl) as pool:
        for i, out_path in enumerate(
            pool.imap_unordered(_task13_worker_one_classification_pkl, worker_args), 1
        ):
            done.append(out_path)
            print(f"[task13] {i}/{len(worker_args)} -> {out_path}")
    return done


def _init_worker(md_inst, mp_inst, all_pairs, kwargs):
    global _md, _mp, _all_motif_pairs, _rd_kwargs
    _md = md_inst
    _mp = mp_inst
    _all_motif_pairs = all_pairs
    _rd_kwargs = kwargs


def _process_one_rhs(rhs):
    t0 = perf_counter()
    rd_local = RuleDiscovery(_md, **_rd_kwargs)
    rd_local.mp = _mp
    rules = rd_local.find_all_valid_lhs_for_rhs(
        RHS=rhs, all_motif_pairs=_all_motif_pairs, mp=_mp
    )
    t1 = perf_counter()
    print(f"[Worker PID {os.getpid()}] [RHS]={rhs} time={round(t1 - t0, 3)} s")
    return (rhs, rules)


def find_all_valid_rules_parallel(md, mp, rd, n_workers=4, dataset_name=None, parallel_rhs=True):
    all_rhs_list = []
    for c_id in md.G:
        for class_id in range(len(md.G[c_id])):
            all_rhs_list.append((c_id, class_id))
    total_rhs = len(all_rhs_list)
    if total_rhs == 0:
        print("No RHS to process.")
        return {}

    rd.find_t1_par_co_occurrence(mp)
    all_motif_pairs = list(all_rhs_list)
    rd_kwargs = dict(
        supp_threshold=rd.supp_threshold,
        max_iter=rd.max_iter,
        is_plot=rd.is_plot,
        is_print=rd.is_print,
        para_model_type=rd.para_model_type,
        confidence_threshold=rd.confidence_threshold,
        epsilon_rate=rd.epsilon_rate,
        plot_save_prefix=rd.plot_save_prefix,
        ransac7_discard_match_if_min_eps_rate_gt_budget=rd.ransac7_discard_match_if_min_eps_rate_gt_budget,
        max_lhs_len=getattr(rd, "max_lhs_len", None),
    )

    valid_rules_dict = {}
    t_start = perf_counter()
    if parallel_rhs:
        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(md, mp, all_motif_pairs, rd_kwargs),
        ) as pool:
            for done_count, (rhs, rules) in enumerate(pool.imap_unordered(_process_one_rhs, all_rhs_list), 1):
                valid_rules_dict[rhs] = rules
                print(f"[Processed] {done_count} / {total_rhs} RHS (current RHS={rhs})")
    else:
        _init_worker(md, mp, all_motif_pairs, rd_kwargs)
        for done_count, rhs in enumerate(all_rhs_list, 1):
            rhs_key, rules = _process_one_rhs(rhs)
            valid_rules_dict[rhs_key] = rules
            print(f"[Processed] {done_count} / {total_rhs} RHS (current RHS={rhs_key})")
    t_end = perf_counter()
    
    

    if dataset_name:
        out_dir = os.path.join("results", "valid_rules")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{dataset_name}.pkl")
        save_instance(valid_rules_dict, save_path)
        print(f"Valid rules saved to: {save_path}")
    print(f"Total time: {round(t_end - t_start, 3)} s")
    print(f"Total rules (all RHS): {sum(len(r) for r in valid_rules_dict.values())}")

    return valid_rules_dict


def _task14_worker_one_classification_sample(args):
    pkl_md, pkl_mp, pkl_out, n_workers_rhs, supp_threshold, epsilon_rate, parallel_rhs = args
    md = load_instance(pkl_md)
    mp = load_instance(pkl_mp)
    stem = os.path.splitext(os.path.basename(pkl_md))[0]
    n_classes = len(md.G.get(0, []))
    print(f"[task14] {stem}.pkl  md.G[0] num_classes={n_classes}")
    rd = RuleDiscovery(
        md=md,
        is_plot=False,
        is_print=False,
        supp_threshold=supp_threshold,
        epsilon_rate=epsilon_rate,
    )
    rd.mp = mp
    valid_rules_dict = find_all_valid_rules_parallel(
        md,
        mp,
        rd,
        n_workers=n_workers_rhs,
        dataset_name=None,
        parallel_rhs=parallel_rhs,
    )
    os.makedirs(os.path.dirname(pkl_out), exist_ok=True)
    save_instance(valid_rules_dict, pkl_out)
    n_rules = sum(len(r) for r in valid_rules_dict.values())
    print(f"[task14] {stem}.pkl  saved {n_rules} rules -> {pkl_out}")
    return pkl_out


def run_classification_valid_rules(
    dataset_name,
    *,
    project_root=None,
    n_workers_pkl=6,
    n_workers_rhs=6,
    supp_threshold=0.8,
    epsilon_rate=1.0,
    clear_output_dir=False,
    parallel_rhs_inside_sample=False,
):
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    motif_res_dir = os.path.join(
        root, "results", "classification", dataset_name, "motif_res"
    )
    motif_pair_dir = os.path.join(
        root, "results", "classification", dataset_name, "motif_pair_res"
    )
    out_dir = os.path.join(
        root, "results", "classification", dataset_name, "valid_rule_res"
    )
    pkls_md = sorted(glob.glob(os.path.join(motif_res_dir, "*.pkl")))
    if not pkls_md:
        print(f"[task14] no md pkl under {motif_res_dir}")
        return []

    if clear_output_dir and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    worker_args = []
    skipped = []
    for pkl_md in pkls_md:
        stem = os.path.splitext(os.path.basename(pkl_md))[0]
        pkl_mp = os.path.join(motif_pair_dir, stem + ".pkl")
        if not os.path.isfile(pkl_mp):
            skipped.append(stem)
            continue
        pkl_out = os.path.join(out_dir, stem + ".pkl")
        worker_args.append(
            (
                pkl_md,
                pkl_mp,
                pkl_out,
                n_workers_rhs,
                supp_threshold,
                epsilon_rate,
                parallel_rhs_inside_sample,
            )
        )
    if skipped:
        print(f"[task14] skip {len(skipped)} samples (no motif_pair_res): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if not worker_args:
        print("[task14] no paired md+mp to process")
        return []

    done = []
    with Pool(processes=n_workers_pkl) as pool:
        for i, out_path in enumerate(
            pool.imap_unordered(_task14_worker_one_classification_sample, worker_args), 1
        ):
            done.append(out_path)
            print(f"[task14] {i}/{len(worker_args)} -> {out_path}")
    return done



def run_full_rule_mining_pipeline(dataset_name, error_type="max", n_workers=6, rhs_attribute_list=None, supp_threshold=0.8,epsilon_rate=0.8):
    t1 = perf_counter()
    print("[1/3] Motif discovery ...")
    motif_discovery2(dataset_name, error_type, is_plot=False, n_workers=n_workers)
    md = load_instance(os.path.join("results", "obj", f"{dataset_name}.pkl"))
    t2 = perf_counter()
    print("Motif discovery time (s):", round(t2 - t1, 3))

    print("[2/3] Motif-pair matching ...")
    t3 = perf_counter()
    rd = RuleDiscovery(md=md, is_plot=False, is_print=False, supp_threshold=supp_threshold, epsilon_rate=epsilon_rate)
    if getattr(md, "m", 0) > 1:
        rd.max_lhs_len = 1
        print(f"[config] m={md.m} > 15, max_lhs_len capped at {rd.max_lhs_len}")
    rd.find_all_best_motif_match3(rhs_attribute_list=rhs_attribute_list, n_workers=n_workers)
    mp = rd.motif_pairs
    rd.mp = mp
    t4 = perf_counter()
    print("Motif-pair matching time (s):", round(t4 - t3, 3))

    print("[3/3] Valid-rule mining ...")
    valid_rules_dict = find_all_valid_rules_parallel(md, mp, rd, n_workers=n_workers, dataset_name=dataset_name)
    t5 = perf_counter()
    print("Valid-rule mining time (s):", round(t5 - t4, 3))
    print("Total pipeline time (s):", round(t5 - t1, 3))
    amount, avg_rmse = check_vr_results(valid_rules_dict)
    print("Rule count:", amount)
    print("Avg error:", avg_rmse)
    return md, mp, valid_rules_dict, amount, avg_rmse, round(t5 - t1, 3)

def sample_parameter_from_pair5(indep_subsqs, dep_subsqs):
    d = len(indep_subsqs)
    if d != len(dep_subsqs) or d == 0:
        raise ValueError("indep_subsqs and dep_subsqs must have same positive length")
    
    k = len(indep_subsqs[0])

    X = np.empty((d, 3 * k), dtype=np.float64)
    X_t1 = np.empty((d, k), dtype=np.float64)
    Y = np.empty((d, 3), dtype=np.float64)
    X_local = X
    X_t1_local = X_t1
    Y_local = Y

    for i in range(d):
        row = indep_subsqs[i]
        Xi = X_local[i]
        Xt1i = X_t1_local[i]

        base = 0
        for j in range(k):
            s = row[j]

            t1 = s.t1
            delta = s.delta
            lam = s.lambda_range[0]

            Xi[base]     = t1
            Xi[base + 1] = delta
            Xi[base + 2] = lam

            Xt1i[j] = t1
            base += 3

        dep = dep_subsqs[i]
        Y_local[i, 0] = dep.t1
        Y_local[i, 1] = dep.delta
        Y_local[i, 2] = dep.lambda_range[0]


    X_design = np.empty((d, 3 * k + 1), dtype=np.float64)
    X_design[:, 0] = 1.0
    X_design[:, 1:] = X_local
    coef, *_ = np.linalg.lstsq(X_design, Y_local, rcond=None)
    para_t1     = coef[:, 0]
    para_delta  = coef[:, 1]
    para_lambda = coef[:, 2]

    a_all, b_all = t1_col_fit_ab_from_X_nb(X_t1_local)
    t1_col_fit = {
        j: ((float(a_all[j - 1]), float(b_all[j - 1])), "linear")
        for j in range(1, k)
    }

    para = {
        "t1":     (para_t1, "multi_linear"),
        "delta":  (para_delta, "multi_linear"),
        "lambda": (para_lambda, "multi_linear"),
        "t1_col_fit": t1_col_fit,
    }
    return para, t1_col_fit




def check_vr_results(vr):
    vr={x:vr[x] for x in vr if vr[x]}
    amount=0
    avg_rmse=0
    for x in vr:
        for y in vr[x]:
            amount+=1
            avg_rmse+=y.avg_err
    return amount, avg_rmse/amount if amount>0 else float('inf')


def upper_bound_from_valid_pairs(valid_pairs):
    s_ids={}
    for i in range(len(valid_pairs)):
        for j in range(len(valid_pairs[0])):
            if j not in s_ids:
                s_ids[j]=[valid_pairs[i][j],]
            else: 
                s_ids[j].append(valid_pairs[i][j])
    s_ids_length=[len(set(s_ids[j])) for j in s_ids]
    return min(s_ids_length) if len(s_ids_length)>0 else 0


if __name__ == "__main__":
    task = 1
    # task 1: full pipeline (motif -> pairs -> rules)
    if task == 1:
        dataset_name = "glucose_T1_3"
        epsilon_rate = 0.6
        supp_threshold = 0.8
        error_type = "max"
        n_workers = 40
        t1 = time()
        md, mp, valid_rules_dict, rule_amount, avg_rmse, pipeline_time_s = run_full_rule_mining_pipeline(
            dataset_name,
            error_type=error_type,
            n_workers=n_workers,
            rhs_attribute_list=None,
            supp_threshold=supp_threshold,
            epsilon_rate=epsilon_rate,
        )
        t2 = time()
        print("Running time (s):", round(t2 - t1, 3))
        print("Saved under results/obj, results/obj_motif_pair, results/valid_rules")
    # task 2: classification motif-pair mining
    elif task == 2:
        dataset_name = "Trace"
        n_workers_pkl = 6
        n_workers_pairs = 8
        run_classification_motif_pairs(
            dataset_name,
            n_workers_pkl=n_workers_pkl,
            n_workers_pairs=n_workers_pairs,
            epsilon_rate=1,
            supp_threshold=0.79,
            clear_output_dir=True,
            parallel_pairs_inside_pkl=False,
            use_ransac7=True,
            ransac7_discard_match_if_min_eps_rate_gt_budget=True,
        )
    
    # task 3: classification valid-rule mining
    elif task == 3:
        dataset_name = "Trace"
        n_workers_pkl = 6
        n_workers_rhs = 6
        run_classification_valid_rules(
            dataset_name,
            n_workers_pkl=n_workers_pkl,
            n_workers_rhs=n_workers_rhs,
            supp_threshold=0.79,
            epsilon_rate=1.0,
            clear_output_dir=True,
            parallel_rhs_inside_sample=False,
        )

