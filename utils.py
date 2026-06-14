import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import pwlf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
from numba import njit, float64
import json
from math import comb, log
import glob
import os

@njit
def _ensure_min_len(x):
    return x

def linear_regression(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x_std = np.std(x)
    if x_std < 1e-10:
        y_mean = y.mean()
        a = 0.0
        b = float(y_mean)
        expression = f'S = {b:.6g}'
        return (((a, b), 'linear'), expression)
    x_mean = x.mean()
    y_mean = y.mean()
    a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - a * x_mean
    expression = f'S = {a:.6g} * t + {b:.6g}'
    return (((a, b), 'linear'), expression)

def linear_predict(x, para):
    (a, b) = para
    x_np = np.asarray(x)
    result = a * x_np + b
    if np.isscalar(x):
        return float(result)
    return result

def multi_linear_regression(X, y):
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    (n_samples, n_features) = X.shape

    X_design = np.column_stack([np.ones(n_samples), X])
    (coef, residuals, rank, s) = np.linalg.lstsq(X_design, y, rcond=None)
    terms = [f'{coef[0]:.6g}']
    for i in range(1, len(coef)):
        terms.append(f'{coef[i]:.6g}*x{i}')
    expression = 'y = ' + ' + '.join(terms)
    return ((coef, 'multi_linear'), expression)

@njit(cache=True)
def multi_linear_predict(X, coef):
    y = coef[0]
    for i in range(len(X)):
        y += X[i] * coef[i + 1]
    return y

@njit(cache=True)
def multi_linear_predict_three_nb(x_t1, x_delta, x_lambda, coef_t1, coef_delta, coef_lambda):
    n = len(x_t1)
    t2 = coef_t1[0]
    for i in range(n):
        t2 += x_t1[i] * coef_t1[i + 1]
    delta2 = coef_delta[0]
    for i in range(n):
        delta2 += x_delta[i] * coef_delta[i + 1]
    lam2 = coef_lambda[0]
    for i in range(n):
        lam2 += x_lambda[i] * coef_lambda[i + 1]
    return (t2, delta2, lam2)

def poly_regression(x, y, degree=3):
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.vander(x, degree + 1, increasing=True)
    para = np.linalg.lstsq(X, y, rcond=None)[0]
    terms = []
    for (i, c) in enumerate(para):
        if i == 0:
            terms.append(f'{c:.6g}')
        elif i == 1:
            terms.append(f'{c:.6g}*t')
        else:
            terms.append(f'{c:.6g}*t^{i}')
    expression = 'S = ' + ' + '.join(terms)
    return ((para, 'poly'), expression)

def poly_predict(x, para):
    x_np = np.asarray(x)
    degree = len(para) - 1
    S = np.zeros_like(x_np, dtype=float)
    for (i, c) in enumerate(para):
        S += c * x_np ** i
    if np.isscalar(x):
        return float(S)
    return S

def trig_model(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def trig_regression(x, y, p0=None):
    x = np.asarray(x)
    y = np.asarray(y)
    if p0 is None:
        a0 = (np.max(y) - np.min(y)) / 2
        d0 = (np.max(y) + np.min(y)) / 2
        mid_value = (np.max(y) + np.min(y)) / 2
        threshold = a0 * 0.1
        y_mean = (y[:-1] + y[1:]) / 2
        zero_crossing_candidates = np.where(np.abs(y_mean - mid_value) < threshold)[0]
        if len(zero_crossing_candidates) >= 2:
            candidate_x = x[zero_crossing_candidates]
            distances = np.diff(candidate_x)
            min_distance = (x[-1] - x[0]) / 20 if len(x) > 1 else 1.0
            valid_distances = distances[distances > min_distance]
            if len(valid_distances) > 0:
                avg_distance = np.mean(valid_distances)
                period = 2 * avg_distance
            else:
                period = (x[-1] - x[0]) / 2
        else:
            period = (x[-1] - x[0]) / 2
        if period <= 0 or not np.isfinite(period):
            period = (x[-1] - x[0]) / 2 if len(x) > 1 else 1.0
        b0 = 2 * np.pi / period
        max_x = x[np.argmax(y)]
        c0 = -b0 * max_x
        p0 = [a0, b0, c0, d0]

    (para, _) = curve_fit(trig_model, x, y, p0=p0, maxfev=50000)
    (a, b, c, d) = para
    expression = f'S = {a:.6g} * sin({b:.6g} * t + {c:.6g}) + {d:.6g}'
    return ((para, 'trig'), expression)

def trig_predict(x, para):

    (a, b, c, d) = para
    x_np = np.asarray(x)
    S = a * np.sin(b * x_np + c) + d
    if np.isscalar(x):
        return float(S)
    return S

def log_model(x, a, b, c, d):
    return a * np.log(b * x + c) + d

def log_regression(x, y, p0=None):
    x = np.asarray(x)
    y = np.asarray(y)
    if p0 is None:
        x_shift = x - np.min(x) + 1
        b0 = 1.0
        c0 = 1.0 - np.min(x) if np.min(x) < 0 else 1.0
        log_x = np.log(x_shift)
        if np.std(log_x) > 1e-10:
            a0 = np.sum((log_x - log_x.mean()) * (y - y.mean())) / np.sum((log_x - log_x.mean()) ** 2)
            d0 = y.mean() - a0 * log_x.mean()
        else:
            a0 = 1.0
            d0 = y.mean()
        p0 = [a0, b0, c0, d0]
    try:
        (para, _) = curve_fit(log_model, x, y, p0=p0, maxfev=50000, bounds=([-np.inf, 1e-10, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    except Exception as e:
        (para, _) = curve_fit(log_model, x, y, p0=[1.0, 1.0, 1.0, y.mean()], maxfev=50000)
    (a, b, c, d) = para
    expression = f'S = {a:.6g} * log({b:.6g} * t + {c:.6g}) + {d:.6g}'
    return ((para, 'log'), expression)

def log_predict(x, para):

    (a, b, c, d) = para
    x_np = np.asarray(x)
    inner = b * x_np + c

    S = a * np.log(inner) + d
    if np.isscalar(x):
        return float(S)
    return S

def exp_model(x, a, b, c, d):
    return a * np.exp(b * x + c) + d

def exp_regression(x, y, p0=None):
    x = np.asarray(x)
    y = np.asarray(y)
    if p0 is None:
        d0 = np.min(y) if np.min(y) > 0 else np.mean(y)
        y_adj = y - d0
        y_adj = np.maximum(y_adj, 1e-10)
        log_y = np.log(y_adj)
        if np.std(x) > 1e-10:
            b0 = np.sum((x - x.mean()) * (log_y - log_y.mean())) / np.sum((x - x.mean()) ** 2)
            log_a_plus_c = log_y.mean() - b0 * x.mean()
            a0 = np.exp(log_a_plus_c)
            c0 = 0.0
        else:
            a0 = np.exp(log_y.mean())
            b0 = 0.0
            c0 = 0.0
        p0 = [a0, b0, c0, d0]
    try:
        (para, _) = curve_fit(exp_model, x, y, p0=p0, maxfev=50000)
    except Exception as e:
        (para, _) = curve_fit(exp_model, x, y, p0=[1.0, 0.0, 0.0, np.mean(y)], maxfev=50000)
    (a, b, c, d) = para
    expression = f'S = {a:.6g} * exp({b:.6g} * t + {c:.6g}) + {d:.6g}'
    return ((para, 'exp'), expression)

def exp_predict(x, para):

    (a, b, c, d) = para
    x_np = np.asarray(x)
    S = a * np.exp(b * x_np + c) + d
    if np.isscalar(x):
        return float(S)
    return S

def inverse_model(x, a, b, c):
    return 1.0 / (a * x + b) + c

def inverse_regression(x, y, p0=None):
    x = np.asarray(x)
    y = np.asarray(y)
    if p0 is None:
        c0 = np.mean(y)
        y_adj = y - c0
        if np.any(np.abs(y_adj) < 1e-10):
            c0 = np.median(y)
            y_adj = y - c0
        inv_y = 1.0 / y_adj
        if np.std(x) > 1e-10:
            a0 = np.sum((x - x.mean()) * (inv_y - inv_y.mean())) / np.sum((x - x.mean()) ** 2)
            b0 = inv_y.mean() - a0 * x.mean()
        else:
            a0 = 1.0
            b0 = inv_y.mean()
        p0 = [a0, b0, c0]
    try:
        (para, _) = curve_fit(inverse_model, x, y, p0=p0, maxfev=50000)
    except Exception as e:
        (para, _) = curve_fit(inverse_model, x, y, p0=[1.0, 1.0, np.mean(y)], maxfev=50000)
    (a, b, c) = para
    expression = f'S = 1 / ({a:.6g} * t + {b:.6g}) + {c:.6g}'
    return ((para, 'inverse'), expression)

def inverse_predict(x, para):

    (a, b, c) = para
    x_np = np.asarray(x)
    denominator = a * x_np + b

    S = 1.0 / denominator + c
    if np.isscalar(x):
        return float(S)
    return S

def piecewise_regression(x, y, n_segments=10):
    model = pwlf.PiecewiseLinFit(x, y)
    breaks = model.fit(n_segments)
    slopes = model.slopes
    intercepts = model.intercepts

    def f(t):
        return model.predict(np.asarray(t))
    return (({'breaks': breaks, 'slopes': slopes, 'intercepts': intercepts}, 'piecewise'), f)

def piecewise_regression2(x, y, n_segments=10):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    km = KMeans(n_clusters=n_segments, n_init=10)
    labels = km.fit_predict(x)
    segments = []
    for k in range(n_segments):
        mask = labels == k
        xi = x[mask].flatten()
        yi = y[mask]
        if len(xi) < 2:
            m = 0
            b = yi.mean()
        else:
            A = np.vstack([xi, np.ones_like(xi)]).T
            (m, b) = np.linalg.lstsq(A, yi, rcond=None)[0]
        left = xi.min()
        right = xi.max()
        segments.append((left, right, m, b))
    segments.sort(key=lambda s: s[0])
    breaks = [segments[0][0]]
    slopes = []
    intercepts = []
    for (left, right, m, b) in segments:
        slopes.append(m)
        intercepts.append(b)
        breaks.append(right)

    def f(t):
        return piecewise_predict2(t, breaks, slopes, intercepts)
    return (({'breaks': breaks, 'slopes': slopes, 'intercepts': intercepts}, 'piecewise'), f)

def piecewise_regression3(x, y, n_segments=10, window=5):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    slopes = np.zeros(n)
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window)
        xi = x[left:right]
        yi = y[left:right]
        A = np.vstack([xi, np.ones_like(xi)]).T
        (m, b) = np.linalg.lstsq(A, yi, rcond=None)[0]
        slopes[i] = m
    F = np.vstack([(x - x.mean()) / x.std(), (y - y.mean()) / y.std(), (slopes - slopes.mean()) / (slopes.std() + 1e-09)]).T
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            block = F[i:j + 1]
            center = block.mean(axis=0)
            cost[i, j] = ((block - center) ** 2).sum()
    DP = np.full((n_segments + 1, n + 1), np.inf)
    back = np.zeros((n_segments + 1, n + 1), dtype=int)
    DP[0, 0] = 0
    for k in range(1, n_segments + 1):
        for j in range(1, n + 1):
            for i in range(j):
                new_cost = DP[k - 1, i] + cost[i, j - 1]
                if new_cost < DP[k, j]:
                    DP[k, j] = new_cost
                    back[k, j] = i
    segments_idx = []
    cur = n
    for k in range(n_segments, 0, -1):
        prev = back[k, cur]
        segments_idx.append((prev, cur))
        cur = prev
    segments_idx.reverse()
    breaks = []
    slopes_out = []
    intercepts_out = []
    for (l, r) in segments_idx:
        xi = x[l:r]
        yi = y[l:r]
        if len(xi) == 0 or len(yi) == 0:
            continue
        min_len = min(len(xi), len(yi))
        if min_len == 0:
            continue
        xi = xi[:min_len]
        yi = yi[:min_len]
        A = np.vstack([xi, np.ones_like(xi)]).T
        (m, b) = np.linalg.lstsq(A, yi, rcond=None)[0]
        slopes_out.append(m)
        intercepts_out.append(b)
        if len(breaks) == 0:
            breaks.append(xi.min())
        breaks.append(xi.max())
    if len(breaks) == 0 or len(slopes_out) == 0:
        A = np.vstack([x, np.ones_like(x)]).T
        (m, b) = np.linalg.lstsq(A, y, rcond=None)[0]
        breaks = [x.min(), x.max()]
        slopes_out = [m]
        intercepts_out = [b]

    def f(t):
        return piecewise_predict2(t, breaks, slopes_out, intercepts_out)
    breaks = [x for x in breaks[1:]] + [np.float64(1.0)]
    return (({'breaks': breaks, 'slopes': slopes_out, 'intercepts': intercepts_out}, 'piecewise'), f)

def piecewise_regression4(x, y, n_segments=10, window=5):
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)
    min_required = n_segments + 1
    if N <= 2:
        print('[auto] extreme small: single segment')
        A = np.vstack([x, np.ones_like(x)]).T
        (m, b) = np.linalg.lstsq(A, y, rcond=None)[0]
        breaks = [x.min(), x.max()]
        slopes_out = [m]
        intercepts_out = [b]

        def f(t):
            return m * t + b
        return (({'breaks': breaks, 'slopes': slopes_out, 'intercepts': intercepts_out}, 'piecewise'), f)
    if N < min_required:
        old = n_segments
        n_segments = max(1, N - 1)
        min_required = N
        print(f'[auto] shrink segments {old} → {n_segments}')
    if N == min_required:
        breaks = [x.min()]
        slopes_out = []
        intercepts_out = []
        for i in range(n_segments):
            xi = x[i:i + 2]
            yi = y[i:i + 2]
            A = np.vstack([xi, np.ones_like(xi)]).T
            (m, b) = np.linalg.lstsq(A, yi, rcond=None)[0]
            slopes_out.append(m)
            intercepts_out.append(b)
            breaks.append(xi.max())

        def f(t):
            return piecewise_predict2(t, breaks, slopes_out, intercepts_out)
        return (({'breaks': breaks, 'slopes': slopes_out, 'intercepts': intercepts_out}, 'piecewise'), f)
    print('[auto] using full piecewise_regression3')
    return piecewise_regression3(x, y, n_segments=n_segments, window=window)

def piecewise_predict(x, breaks, slopes, intercepts):
    x_np = np.asarray(x)
    y = np.zeros_like(x_np, dtype=float)
    for i in range(len(slopes)):
        left = breaks[i]
        right = breaks[i + 1]
        mask = (x_np >= left) & (x_np < right)
        y[mask] = slopes[i] * x_np[mask] + intercepts[i]
    if np.isscalar(x):
        return float(y)
    return y

def piecewise_predict2(x, breaks, slopes, intercepts):
    x_np = np.asarray(x)
    y = np.zeros_like(x_np, dtype=float)
    for i in range(len(x_np)):
        for j in range(len(slopes)):
            if x_np[i] >= breaks[j] and x_np[i] < breaks[j + 1]:
                y[i] = x_np[i] * slopes[j] + intercepts[j]
                break
            if x_np[i] == breaks[-1]:
                y[i] = x_np[i] * slopes[-1] + intercepts[-1]
                break
    return y

def normalize_shift_scale(arr, scale=1):
    arr = np.asarray(arr, dtype=float)
    shifted = arr - arr[0]
    rng = shifted.max() - shifted.min()
    if rng == 0:
        return shifted * 0
    normalized = shifted / rng * scale
    return normalized

def Predict(x, param, n_x=None):
    x = np.asarray(x)
    if n_x is None:
        x_ = x / len(x)
    else:
        x_ = x / n_x
    (para, kind) = param
    if kind == 'linear':
        S_pred = linear_predict(x_, para)
    elif kind == 'poly':
        S_pred = poly_predict(x_, para)
    elif kind == 'trig':
        S_pred = trig_predict(x_, para)
    elif kind == 'log':
        S_pred = log_predict(x_, para)
    elif kind == 'exp':
        S_pred = exp_predict(x_, para)
    elif kind == 'inverse':
        S_pred = inverse_predict(x_, para)
    elif kind == 'piecewise':
        S_pred = piecewise_predict2(x_, para['breaks'], para['slopes'], para['intercepts'])
    return S_pred

@njit
def linear_predict_nb(x, para):
    a = para[0]
    b = para[1]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = a * x[i] + b
    return out

@njit
def poly_predict_nb(x, para):
    n = x.shape[0]
    m = para.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xx = x[i]
        v = 0.0
        p = 1.0
        for j in range(m):
            v += para[j] * p
            p *= xx
        out[i] = v
    return out

@njit
def trig_predict_nb(x, para):
    a = para[0]
    b = para[1]
    c = para[2]
    d = para[3]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = a * np.sin(b * x[i] + c) + d
    return out

@njit
def log_predict_nb(x, para):
    a = para[0]
    b = para[1]
    c = para[2]
    d = para[3]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = b * x[i] + c
        if v <= 0.0:
            out[i] = d
        else:
            out[i] = a * np.log(v) + d
    return out

@njit
def exp_predict_nb(x, para):
    a = para[0]
    b = para[1]
    c = para[2]
    d = para[3]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = a * np.exp(b * x[i] + c) + d
    return out

@njit
def inverse_predict_nb(x, para):
    a = para[0]
    b = para[1]
    c = para[2]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        denom = a * x[i] + b
        if np.abs(denom) < 1e-12:
            out[i] = c
        else:
            out[i] = 1.0 / denom + c
    return out

@njit
def piecewise_predict2_nb(x, breaks, slopes, intercepts, is_check=False):
    n = x.shape[0]
    k = slopes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xx = x[i]
        for j in range(k):
            if xx >= breaks[j] and xx < breaks[j + 1]:
                out[i] = xx * slopes[j] + intercepts[j]
                break
            if abs(xx - breaks[-1]) < 1e-05:
                out[i] = xx * slopes[k - 1] + intercepts[k - 1]
                break

    return out

@njit(cache=True)
def piecewise_predict2_nb_exact_match(x, breaks, slopes, intercepts):
    n = x.shape[0]
    k = slopes.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last_break = breaks[k]
    for i in range(n):
        xx = x[i]
        for j in range(k):
            if xx >= breaks[j] and xx < breaks[j + 1]:
                out[i] = xx * slopes[j] + intercepts[j]
                break
            if xx == last_break:
                out[i] = xx * slopes[k - 1] + intercepts[k - 1]
                break
    return out

@njit
def lam_of(S):
    m1 = S[0]
    m2 = S[0]
    for v in S:
        if v < m1:
            m1 = v
        if v > m2:
            m2 = v
    return m2 - m1

def select_predict_nb(kind):
    if kind == 'linear':
        return linear_predict_nb
    elif kind == 'poly':
        return poly_predict_nb
    elif kind == 'trig':
        return trig_predict_nb
    elif kind == 'log':
        return log_predict_nb
    elif kind == 'exp':
        return exp_predict_nb
    elif kind == 'inverse':
        return inverse_predict_nb
    elif kind == 'piecewise':
        return piecewise_predict2_nb
    else:
        raise ValueError('unknown kind')

@njit
def calc_error_local_alignment3(S1, S2, window=5, gamma_penalty=0):
    s0 = S1[0]
    n1 = len(S1)
    n2 = len(S2)
    if n1 == 0 or n2 == 0:
        return None
    min_len = n1 if n1 < n2 else n2
    if min_len == 0:
        return None
    if window < 0:
        window = 0
    window = int(window)
    if min_len > 1:
        if window > min_len - 1:
            window = min_len - 1
    else:
        window = 0
    error = np.empty(min_len, dtype=np.float64)
    for i in range(min_len):
        min_val1 = np.inf
        min_val2 = np.inf
        s2_i = S2[i]
        s1_i = S1[i]
        left = i - window
        if left < 0:
            left = 0
        right = i + window + 1
        if right > min_len:
            right = min_len
        for j in range(left, right):
            diff = S1[j] - s2_i
            if diff < 0:
                diff = -diff
            if diff < min_val1:
                min_val1 = diff
        for j in range(left, right):
            diff = s1_i - S2[j]
            if diff < 0:
                diff = -diff
            if diff < min_val2:
                min_val2 = diff
        if min_val1 > min_val2:
            error[i] = min_val1
        else:
            error[i] = min_val2
    return error

@njit
def normalize_seq(S, s0):
    n = len(S)
    if n == 0:
        return S
    smin = S[0]
    smax = S[0]
    for i in range(1, n):
        v = S[i]
        if v < smin:
            smin = v
        elif v > smax:
            smax = v
    rng = smax - smin
    out = np.empty(n, dtype=np.float64)
    if rng == 0.0:
        out.fill(0.0)
    else:
        inv_rng = 1.0 / rng
        for i in range(n):
            out[i] = (S[i] - s0) * inv_rng
    return out

@njit
def calc_error_local_alignment4(S1, S2, window=5, gamma_penalty=0):
    s0 = S1[0]
    S1 = normalize_seq(S1, s0=s0)
    S2 = normalize_seq(S2, s0=s0)
    n1 = len(S1)
    n2 = len(S2)
    if n1 == 0 or n2 == 0:
        return None
    min_len = n1 if n1 < n2 else n2
    if min_len == 0:
        return None
    if window < 0:
        window = 0
    window = int(window)
    if min_len > 1:
        if window > min_len - 1:
            window = min_len - 1
    else:
        window = 0
    error = np.empty(min_len, dtype=np.float64)
    for i in range(min_len):
        min_val1 = np.inf
        min_val2 = np.inf
        s2_i = S2[i]
        s1_i = S1[i]
        left = i - window
        if left < 0:
            left = 0
        right = i + window + 1
        if right > min_len:
            right = min_len
        for j in range(left, right):
            diff = S1[j] - s2_i
            if diff < 0:
                diff = -diff
            if diff < min_val1:
                min_val1 = diff
        for j in range(left, right):
            diff = s1_i - S2[j]
            if diff < 0:
                diff = -diff
            if diff < min_val2:
                min_val2 = diff
        if min_val1 > min_val2:
            error[i] = min_val1
        else:
            error[i] = min_val2
    return error


def truncate_by_percentile(error, p=0.9):
    threshold = np.percentile(error, p * 100)
    return np.minimum(error, threshold)

def truncate_two_sided(arr, r):
    arr = np.asarray(arr, dtype=float)

    low = np.percentile(arr, r * 100)
    high = np.percentile(arr, (1 - r) * 100)
    return np.clip(arr, low, high)

def similarity_matrix_3(time_series, r=15, similarity_metric='euclidean', local_align_window=2, is_plot=False, skip_regions=[]):
    time_series = np.asarray(time_series).flatten()
    n = len(time_series)
    print('(1) Generating feature vectors...')
    feature_vectors = []
    valid_indices = []
    skip_mask = np.zeros(n, dtype=bool)
    if skip_regions:
        for (skip_start, skip_end) in skip_regions:
            skip_start = max(0, int(skip_start))
            skip_end = min(n, int(skip_end))
            skip_mask[skip_start:skip_end] = True
    for i in range(n):
        if skip_mask[i]:
            continue
        start_idx = max(0, i - r)
        end_idx = min(n, i + r + 1)
        window = time_series[start_idx:end_idx]
        if len(window) >= r + 1:
            normalized_window = normalize_shift_scale(window, scale=1)
            feature_vectors.append(normalized_window)
            valid_indices.append(i)

    print('(2) Calculating distance matrix...')
    n_features = len(feature_vectors)
    distance_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                vec1 = feature_vectors[i]
                vec2 = feature_vectors[j]
                min_len = min(len(vec1), len(vec2))
                vec1_aligned = vec1[:min_len]
                vec2_aligned = vec2[:min_len]
                distance = np.sqrt(np.sum((vec1_aligned - vec2_aligned) ** 2))
                distance_matrix[i, j] = distance
    for i in range(n_features):
        for j in range(i + 1, n_features):
            distance_matrix[j, i] = distance_matrix[i, j]
    print('(3) Aligning distance matrix...')
    distance_matrix_aligned = distance_matrix.copy()
    print('(4) Calculating similarity matrix...')
    similarity_matrix = 1.0 / (1.0 + np.power(distance_matrix_aligned, 0.5))
    np.fill_diagonal(similarity_matrix, 1.0)
    similarity_matrix = similarity_matrix.astype(np.float32)
    if is_plot:
        print('(5) Drawing similarity matrix...')
        d = 6
        r = 1.2
        colorbar_width = 0.5
        total_width = r + d + colorbar_width
        total_height = r + d
        fig = plt.figure(figsize=(total_width, total_height))
        gs = fig.add_gridspec(2, 3, height_ratios=[r, d], width_ratios=[r, d, colorbar_width], hspace=0.3, wspace=0.3)
        ax_top = fig.add_subplot(gs[0, 1])
        ax_top.plot(time_series[valid_indices], 'b-', linewidth=1)
        ax_top.set_title('Original Time Series (Top)', fontsize=10)
        ax_top.set_xlabel('Index')
        ax_top.set_ylabel('Value')
        ax_top.grid(True, alpha=0.3)
        x_range = len(valid_indices) if len(valid_indices) > 0 else 1
        y_range = np.max(time_series[valid_indices]) - np.min(time_series[valid_indices])
        y_range = y_range if y_range > 0 else 1
        aspect_ratio = r / d * (x_range / y_range)
        ax_top.set_aspect(aspect_ratio, adjustable='box')
        ax_left = fig.add_subplot(gs[1, 0])
        indices = np.arange(len(valid_indices))
        ax_left.plot(time_series[valid_indices], indices, 'b-', linewidth=1)
        ax_left.set_title('Original Time Series (Left)', fontsize=10, rotation=90, y=0.5, x=-0.1)
        ax_left.set_xlabel('Value', rotation=90)
        ax_left.set_ylabel('Index')
        ax_left.grid(True, alpha=0.3)
        ax_left.invert_yaxis()
        y_range_left = len(valid_indices) if len(valid_indices) > 0 else 1
        x_range_left = np.max(time_series[valid_indices]) - np.min(time_series[valid_indices])
        x_range_left = x_range_left if x_range_left > 0 else 1
        aspect_ratio_left = d / r * (y_range_left / x_range_left)
        ax_left.set_aspect(aspect_ratio_left, adjustable='box')
        ax_matrix = fig.add_subplot(gs[1, 1])
        im = ax_matrix.imshow(similarity_matrix, cmap='gray_r', aspect='equal', origin='upper', interpolation='nearest')
        ax_matrix.set_title('Self-Similarity Matrix\n(Brighter = Higher Similarity)', fontsize=12)
        ax_matrix.set_xlabel('Index j')
        ax_matrix.set_ylabel('Index i')
        ax_cbar = fig.add_subplot(gs[1, 2])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Similarity', rotation=270, labelpad=15)
        plt.tight_layout()
        plt.show()
    return similarity_matrix

def filter_motif_sets(motif_sets, motif_length, minimum_interval):
    motif_sets = sorted(motif_sets)
    threshold = int(motif_length * minimum_interval)
    result = []
    for a in motif_sets:
        if result and a < result[-1] + threshold:
            continue
        result.append(a)
    return result

def _ensure_parent_dir(filename):
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_instance(obj, filename):
    _ensure_parent_dir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_instance(filename):

    class _CompatUnpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)
    with open(filename, 'rb') as f:
        return _CompatUnpickler(f).load()

def load_instance_list(dir_path, keep_empty=False):
    pkl_paths = sorted(glob.glob(os.path.join(dir_path, '*.pkl')))
    out = {}
    for p in pkl_paths:
        fname = os.path.basename(p)
        stem = os.path.splitext(fname)[0]
        (sample_id_str, label_str) = stem.split('_', 1)
        sample_id = int(sample_id_str)
        label = int(label_str)
        data = load_instance(p)
        if not isinstance(data, dict):
            continue
        for ((c_id, class_id), rules) in data.items():
            if not keep_empty and (not rules):
                continue
            out[sample_id, label, int(c_id), int(class_id)] = rules
    return out

def save_data(data, filename):

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for (key, value) in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, set):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    data = convert_numpy_types(data)
    if isinstance(data, set):
        data = list(data)
    _ensure_parent_dir(filename)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_data(filename, data_type=None):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data_type == set:
        return set(data)
    elif data_type == list:
        return list(data)
    elif data_type == dict:
        return dict(data)
    else:
        return data

def intersects(A, B):
    return not A.isdisjoint(B)

def longest_element(lst):
    return max(lst, key=len)

def Calc_max_turns(confidence_threshold, degree, current_best_score, m1, m2):
    P = comb(current_best_score, degree) / (comb(m1, degree) * comb(m2, 1))
    max_turns = round(log(1 - confidence_threshold) / log(1 - P))
    return max_turns

@njit(cache=True)
def predict_rhs_subsequence_nb(X_t1, X_delta, X_lam, coef_t1, coef_delta, coef_lambda, db_rhs, n_rhs, kind_int, param_linear, breaks, slopes, intercepts, S_pred_buf, is_only_motif=False):
    (t_2, delta_2, lambda_2) = multi_linear_predict_three_nb(X_t1, X_delta, X_lam, coef_t1, coef_delta, coef_lambda)
    t_2 = int(round(t_2))
    delta_2 = int(round(delta_2))
    if t_2 < 0 or delta_2 <= 0 or t_2 + delta_2 > n_rhs:
        return (0, 0, 0, 0.0)
    if delta_2 > 1:
        x_norm = np.arange(delta_2, dtype=np.float64) / (delta_2 - 1)
    else:
        x_norm = np.zeros(1, dtype=np.float64)
    base = db_rhs[t_2]
    if np.isnan(base) and t_2 >= 1:
        base = db_rhs[t_2 - 1]
    if not is_only_motif:
        if kind_int == 0:
            S_pred = linear_predict_nb(x_norm, param_linear) * lambda_2 + base
        else:
            S_pred = piecewise_predict2_nb_exact_match(x_norm, breaks, slopes, intercepts) * lambda_2 + base
    elif kind_int == 0:
        S_pred = linear_predict_nb(x_norm, param_linear) * lambda_2
    else:
        S_pred = piecewise_predict2_nb_exact_match(x_norm, breaks, slopes, intercepts) * lambda_2
    for i in range(len(S_pred)):
        S_pred_buf[i] = S_pred[i]
    return (1, t_2, delta_2, lambda_2)



@njit(cache=True)
def t1_col_fit_ab_from_X_nb(X_t1):
    (d, k) = X_t1.shape
    n_col_fit = max(0, k - 1)
    a_all = np.zeros(n_col_fit, dtype=np.float64)
    b_all = np.zeros(n_col_fit, dtype=np.float64)
    if n_col_fit <= 0:
        return (a_all, b_all)
    x_col0 = X_t1[:, 0]
    mean_x = x_col0.mean()
    var_x = ((x_col0 - mean_x) ** 2).mean()
    if var_x < 1e-10:
        for j in range(n_col_fit):
            b_all[j] = X_t1[:, j + 1].mean()
    else:
        centered_x = x_col0 - mean_x
        dot_xx = np.dot(centered_x, centered_x)
        for j in range(n_col_fit):
            y_col = X_t1[:, j + 1]
            mean_y = y_col.mean()
            a_all[j] = np.dot(centered_x, y_col - mean_y) / dot_xx
            b_all[j] = mean_y - a_all[j] * mean_x
    return (a_all, b_all)

def predict_rhs_from_lhs_combination(md, rule, lhs_combination, db_rhs, is_only_motif=False):
    n_lhs = len(rule.LHS)
    if len(lhs_combination) != n_lhs:
        return (False, 0, 0, 0.0, None)
    X_t1 = np.empty(n_lhs, dtype=np.float64)
    X_delta = np.empty(n_lhs, dtype=np.float64)
    X_lam = np.empty(n_lhs, dtype=np.float64)
    for i in range(n_lhs):
        (c_id, class_id) = (rule.LHS[i][0], rule.LHS[i][1])
        sub = md.G[c_id][class_id].SubSq_obj_list[lhs_combination[i]]
        X_t1[i] = sub.t1
        X_delta[i] = sub.delta
        X_lam[i] = sub.lambda_range[0]
    coef_t1 = np.asarray(rule.params['t1'][0], dtype=np.float64)
    coef_delta = np.asarray(rule.params['delta'][0], dtype=np.float64)
    coef_lambda = np.asarray(rule.params['lambda'][0], dtype=np.float64)
    predict_from_t1_only = rule.params.get('predict_from_t1_only', False)
    if predict_from_t1_only:
        (X_t1_in, X_delta_in, X_lam_in) = (X_t1, X_t1, X_t1)
    elif len(coef_t1) == 3 * n_lhs + 1:
        X_flat = np.empty(3 * n_lhs, dtype=np.float64)
        for i in range(n_lhs):
            X_flat[3 * i] = X_t1[i]
            X_flat[3 * i + 1] = X_delta[i]
            X_flat[3 * i + 2] = X_lam[i]
        X_t1_in = X_delta_in = X_lam_in = X_flat
    else:
        (X_t1_in, X_delta_in, X_lam_in) = (X_t1, X_delta, X_lam)
    (rhs_cid, rhs_class) = (rule.RHS[0], rule.RHS[1])
    (param, kind) = md.G[rhs_cid][rhs_class].f
    n_rhs = len(db_rhs)
    db_rhs = np.asarray(db_rhs, dtype=np.float64)
    if kind not in ('linear', 'piecewise'):
        return (False, 0, 0, 0.0, None)
    kind_int = 0 if kind == 'linear' else 6
    if kind == 'linear':
        param_linear = np.asarray(param, dtype=np.float64)
        breaks = np.zeros(1, dtype=np.float64)
        slopes = np.zeros(1, dtype=np.float64)
        intercepts = np.zeros(1, dtype=np.float64)
    else:
        param_linear = np.zeros(2, dtype=np.float64)
        breaks = np.asarray(param['breaks'], dtype=np.float64)
        slopes = np.asarray(param['slopes'], dtype=np.float64)
        intercepts = np.asarray(param['intercepts'], dtype=np.float64)
    S_pred_buf = np.empty(n_rhs, dtype=np.float64)
    (valid_flag, t_2, delta_2, lambda_2) = predict_rhs_subsequence_nb(X_t1_in, X_delta_in, X_lam_in, coef_t1, coef_delta, coef_lambda, db_rhs, n_rhs, kind_int, param_linear, breaks, slopes, intercepts, S_pred_buf, is_only_motif)
    if valid_flag == 0:
        return (False, 0, 0, 0.0, None)
    S_pred = S_pred_buf[:delta_2].copy()
    return (True, t_2, delta_2, lambda_2, S_pred)

def Calc_imputation_rmse(ori_instance, missing_instance, imp_instance):
    ori_instance = np.asarray(ori_instance, dtype=np.float64)
    missing_instance = np.asarray(missing_instance, dtype=np.float64)
    imp_instance = np.asarray(imp_instance, dtype=np.float64)

    mask = np.isnan(missing_instance)
    n_missing = np.sum(mask)
    if n_missing == 0:
        return 0.0
    err2 = (imp_instance[mask] - ori_instance[mask]) ** 2
    return np.sqrt(np.mean(err2))

def get_min_regression_samples(n_lhs, use_t1_only=False):
    if use_t1_only:
        return n_lhs + 1
    return 4 if n_lhs == 1 else 3 * n_lhs + 1

def fill_missing_with_arima(missing_db, order=(1, 0, 1), max_obs=500, refit_gap=50):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        from statsmodels.tsa.arima.model import ARIMA
        SARIMAX = None
    import numpy as np
    import warnings
    out = np.array(missing_db, dtype=np.float64, copy=True)
    (n_row, n_col) = out.shape
    min_required = max(3, order[0] + order[2] + 2)
    for col in range(n_col):
        y = out[:, col]
        if not np.any(np.isnan(y)):
            continue
        valid_mask = ~np.isnan(y)
        valid_idx = np.where(valid_mask)[0]
        nan_mask = ~valid_mask
        diff = np.diff(nan_mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if nan_mask[0]:
            starts = np.insert(starts, 0, 0)
        if nan_mask[-1]:
            ends = np.append(ends, n_row)
        segs = list(zip(starts, ends))
        res = None
        last_fit_end = -1
        for (start, end) in segs:
            hist_start = max(0, start - max_obs)
            hist_idx = valid_idx[(valid_idx < start) & (valid_idx >= hist_start)]
            if len(hist_idx) < min_required:
                fill_val = np.nanmean(y[valid_idx]) if len(valid_idx) > 0 else 0.0
                out[start:end, col] = fill_val
                y[start:end] = fill_val
                continue
            hist = y[hist_idx]
            need_refit = res is None or start - last_fit_end > refit_gap
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    if need_refit:
                        if SARIMAX is not None:
                            model = SARIMAX(hist, order=order, enforce_stationarity=False, enforce_invertibility=False, simple_differencing=True)
                            res = model.fit(disp=False, maxiter=20)
                        else:
                            model = ARIMA(hist, order=order)
                            res = model.fit()
                        last_fit_end = start
                    steps = end - start
                    fcast = res.forecast(steps=steps)
                out[start:end, col] = fcast
                y[start:end] = fcast
            except Exception:
                fill_val = np.nanmean(hist)
                out[start:end, col] = fill_val
                y[start:end] = fill_val
    return out

def Calc_cdd_rhs_score(cdd_imp, segment):
    cdd_range = (cdd_imp.t_2, cdd_imp.t_2 + cdd_imp.delta_2)
    int_range = (max(cdd_range[0], segment[0]), min(cdd_range[1], segment[1]))
    if int_range[1] <= int_range[0]:
        return 0
    l1 = segment[1] - segment[0]
    l2 = cdd_range[1] - cdd_range[0]
    l3 = int_range[1] - int_range[0]
    r = l3 / l1 if l1 > 0 else 0
    p = l3 / l2 if l2 > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    epsilon = cdd_imp.epsilon
    return f1 * 1 / (epsilon + 1e-07)

def data_reader_for_classification(data_path):
    db = pd.read_csv(data_path).values
    return db
