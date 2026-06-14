from time import time
from time import perf_counter
import os
import sys
import tempfile
from multiprocessing import Pool
os.environ['NUMBA_DISABLE_CACHE'] = '1'
try:
    os.environ['NUMBA_CACHE_DIR'] = ''
except:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import shutil
from utils import *
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    print('Warning: kneed library not found. Using fallback elbow detection.')
motiflets_main_path = os.path.join(current_dir, 'motiflets-main')
if motiflets_main_path not in sys.path:
    sys.path.insert(0, motiflets_main_path)
try:
    import numba
    try:
        from numba.core.caching import CacheImpl

        def disabled_save_overload(self, sig, data):
            try:
                if hasattr(self, '_save_overload_original'):
                    return self._save_overload_original(sig, data)
            except:
                pass
        if hasattr(CacheImpl, 'save_overload'):
            if not hasattr(CacheImpl, '_save_overload_original'):
                CacheImpl._save_overload_original = CacheImpl.save_overload
            CacheImpl.save_overload = disabled_save_overload
            print('[Numba] cache save disabled via monkey patch')
    except Exception as e:
        pass
    try:
        from numba.core.caching import CacheFile

        def disabled_cache_file_save(self, key, data):
            pass
        if hasattr(CacheFile, 'save'):
            CacheFile.save = disabled_cache_file_save
            print('[Numba] cache file save disabled')
    except:
        pass
except:
    pass
from motiflets.plotting import Motiflets
from motif_mining_function import mine_motifs

class SubSq:

    def __init__(self, t1, delta, lambda_range, class_id, c_id):
        self.lambda_range = lambda_range
        self.delta = delta
        self.t1 = t1
        self.class_id = class_id
        self.c_id = c_id

class SubSqSet:

    def __init__(self, f, c_id, epsilon, lambda_lower_bound, delta_range):
        self.f = f
        self.c_id = c_id
        self.class_id = None
        self.epsilon = epsilon
        self.lambda_lower_bound = lambda_lower_bound
        self.delta_range = delta_range
        self.gamma = None
        self.SubSq_obj_list = []
        self.SubSq_obj_dict = {}
        self.t1_co_occurrence = set()
        self.par_co_occurrence = set()

class MotifDiscovery:

    def __init__(self, dataname, epsilonM, delta_range, window_rate=0.08, A_id_set=None, error_percentage_tolerance=0.1, distance_strategy='local_alignment', minimum_interval=1, SSM=None, scale=1, motif_length=None, motif_length_range=None, r=15, local_align_window=2, skip_regions=[], is_save=True, error_type='max', gamma_penalty=0, min_count=3, is_plot=False, double_step=1):
        self.dataname = dataname
        self.data_source = 'data/'
        self.data_path = self.data_source + dataname + '.csv'
        self.db = None
        self.epsilonM = epsilonM
        self.delta_range = delta_range
        self.lambda_lower_bound = 2 * epsilonM
        self.G = {}
        self.A_id_set = A_id_set
        self.error_percentage_tolerance = error_percentage_tolerance
        self.distance_strategy = distance_strategy
        self.minimum_interval = minimum_interval
        self.window_rate = window_rate
        self.temp_window_rate = None
        self.forbidden_area = set()
        self.forbidden_area2 = []
        self.permitted_area = None
        self.scale = scale
        self.motif_length = motif_length
        self.motif_length_range = motif_length_range
        self.nearly_all_forbidden = False
        self.ending = False
        self.r = r
        self.local_align_window = local_align_window
        self.skip_regions = skip_regions
        self.SSM = SSM
        self.is_plot = is_plot
        self.is_save = is_save
        self.error_type = error_type
        self.gamma_penalty = gamma_penalty
        self.min_count = min_count
        self.double_step = max(1, int(double_step))

    def _error_aggregate(self, error):
        if error is None or len(error) == 0:
            return np.inf
        if self.error_type == 'mae':
            return float(np.mean(error))
        if self.error_type == 'max':
            return float(np.max(error))

    def prune_small_subsets_and_report(self):
        min_count = self.min_count
        for c_id in self.G:
            for class_id in range(len(self.G[c_id])):
                n = len(self.G[c_id][class_id].SubSq_obj_list)
                print(f'({c_id},{class_id}):{n}')
        for c_id in list(self.G.keys()):
            kept = [sss for sss in self.G[c_id] if len(sss.SubSq_obj_list) >= min_count]
            for (i, sss) in enumerate(kept):
                sss.class_id = i
            self.G[c_id] = kept

    def read_data(self):
        df = pd.read_csv(self.data_path)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        self.columns = df.columns
        self.db = df.to_numpy()
        if self.db.ndim == 1:
            self.db = self.db.reshape(-1, 1)
        self.id_col = {i: self.columns[i] for i in range(len(self.columns))}
        self.col_id = {self.columns[i]: i for i in range(len(self.columns))}
        (self.n, self.m) = self.db.shape
        self.G = {c_id: [] for c_id in self.id_col.keys()}
        if self.A_id_set is None:
            self.A_id_set = list(self.id_col.keys())

    def apply_univariate_series(self, series_1d, column_name='v0'):
        s = np.asarray(series_1d, dtype=np.float64).ravel()
        s = s[~np.isnan(s)]
        if s.size == 0:
            raise ValueError('apply_univariate_series: empty series after dropping NaN')
        self.db = s.reshape(-1, 1)
        self.columns = pd.Index([column_name])
        (self.n, self.m) = self.db.shape
        self.id_col = {0: column_name}
        self.col_id = {column_name: 0}
        self.G = {0: []}
        self.A_id_set = [0]

    def call_motiflets(self, db, k_max=10, motif_length=None, motif_length_range=None, distance='znormed_ed', n_jobs=-1, excluded_ranges=None, is_plot=False):

        if isinstance(db, pd.Series):
            series = db.values
        elif isinstance(db, pd.DataFrame):
            series = db.iloc[:, 0].values
        elif isinstance(db, np.ndarray):
            series = db.flatten()
        elif isinstance(db, list):
            series = np.array(db)
        else:
            raise ValueError(f'Unsupported data type: {type(db)}')
        if series.ndim > 1:
            series = series.flatten()
        series = pd.to_numeric(pd.Series(series), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        if len(series) == 0:
            raise ValueError('Time series data is empty')
        (fd, temp_csv) = tempfile.mkstemp(prefix=f'temp_motiflets_data_{os.getpid()}_', suffix='.csv', dir=current_dir)
        os.close(fd)
        try:
            import numba
            try:
                numba.config.DISABLE_JIT = False
            except:
                pass
        except:
            pass

        def clear_numba_cache():
            try:
                import numba
                import glob
                import tempfile
                cache_dir = numba.config.CACHE_DIR
                print(f'[Numba] trying to clear cache directory: {cache_dir}')
                user_cache_dirs = []
                try:
                    import pathlib
                    user_cache = pathlib.Path.home() / '.numba_cache'
                    if user_cache.exists():
                        user_cache_dirs.append(str(user_cache))
                except:
                    pass
                all_cache_dirs = [cache_dir] + user_cache_dirs
                for cache_dir_path in all_cache_dirs:
                    if os.path.exists(cache_dir_path):
                        try:
                            deleted_count = 0
                            for pattern in ['*.nbc', '*.nbi']:
                                for file_path in glob.glob(os.path.join(cache_dir_path, '**', pattern), recursive=True):
                                    try:
                                        os.remove(file_path)
                                        deleted_count += 1
                                    except:
                                        pass
                            shutil.rmtree(cache_dir_path, ignore_errors=True)
                            if os.path.exists(cache_dir_path):
                                for (root, dirs, files) in os.walk(cache_dir_path):
                                    for file in files:
                                        try:
                                            os.remove(os.path.join(root, file))
                                            deleted_count += 1
                                        except:
                                            pass
                                    for dir in dirs:
                                        try:
                                            os.rmdir(os.path.join(root, dir))
                                        except:
                                            pass
                            if deleted_count > 0:
                                print(f'[Numba] already cleared cache directory: {cache_dir_path} (deleted {deleted_count} files)')
                            else:
                                print(f'[Numba] cache directory is already empty: {cache_dir_path}')
                        except Exception as e:
                            print(f'[Numba] warning when clearing cache: {e}')
                    else:
                        print(f'[Numba] cache directory does not exist: {cache_dir_path}')
                return True
            except Exception as e:
                print(f'[Numba] cannot clear cache: {e}')
                return False
        try:
            pd.DataFrame(series).to_csv(temp_csv, index=False, header=False)
            output_dir = os.path.join(current_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            print(f'[Motiflets] start identifying motif (k_max={k_max}, motif_length={motif_length})...')
            print('[Numba] pre-clear cache...')
            clear_numba_cache()
            max_retries = 3
            ml = None
            motif_sets = None
            motif_length = None
            for attempt in range(max_retries):
                try:
                    (ml, motif_sets, motif_length) = mine_motifs(csv_path=temp_csv, k_max=k_max, motif_length=motif_length, motif_length_range=motif_length_range, ds_name=self.dataname, column=0, distance=distance, n_jobs=n_jobs, output_dir=output_dir, db=None, forbidden_area=self.forbidden_area2, debug_context=getattr(self, '_debug_context', None))
                    break
                except (ReferenceError, KeyError, Exception) as e:
                    error_msg = str(e).lower()
                    error_type = type(e).__name__
                    is_forbidden_area_error = error_type == 'TypeError' and "'nonetype' object is not iterable" in error_msg
                    if is_forbidden_area_error:
                        print('Too much forbidden area! ')
                        self.nearly_all_forbidden = True
                        self.permitted_area = [x for x in range(self.n) if x not in self.forbidden_area]
                        (motif_sets, motif_length) = self.find_continuous_region()
                        break
                    is_cache_error = 'underlying object has vanished' in error_msg or 'keyerror' in error_msg or error_type == 'KeyError' or (error_type == 'ReferenceError') or ('cache' in error_msg) or ('numba' in error_msg)
                    if is_cache_error:
                        if attempt < max_retries - 1:
                            print(f'[Numba] Detected cache error ({error_type}), clearing cache and retrying... (attempt {attempt + 1}/{max_retries})')
                            clear_numba_cache()
                            import time
                            time.sleep(1.0)
                            continue
                        else:
                            print(f'[Numba] Maximum retry attempts reached, still failed.')
                            print(f'[Numba] Last error: {error_type}: {error_msg}')
                            print('\n' + '=' * 60)
                            print('Solutions:')
                            print('1. Manually clean Numba cache:')
                            print('   - Windows: Delete %USERPROFILE%\\.numba_cache directory')
                            print('   - Or run: python -c "import numba; import shutil; import os; cache_dir = numba.config.CACHE_DIR; shutil.rmtree(cache_dir, ignore_errors=True) if os.path.exists(cache_dir) else None; print(\'Cache cleaned\')"')
                            print('2. Modify motiflets library to disable cache:')
                            print('   - Find .py files in motiflets-main/motiflets/ directory')
                            print('   - Replace all @njit(cache=True) with @njit(cache=False)')
                            print('   - Or replace all cache=True with cache=False')
                            print('3. Re-run the program')
                            print('=' * 60 + '\n')
                            raise RuntimeError(f'Numba cache error cannot be resolved: {error_type}: {error_msg}\nPlease refer to the solutions above.')
                    else:
                        raise
            if motif_sets is None or motif_length is None:
                raise RuntimeError('mine_motifs execution failed: no valid results returned')
            motif_sets = filter_motif_sets(motif_sets, motif_length, self.minimum_interval)
            result = [(x + 1, int(x + motif_length + 1)) for x in motif_sets]
            clear_numba_cache()
            return (result, motif_length)
        finally:
            if os.path.exists(temp_csv):
                try:
                    os.remove(temp_csv)
                except:
                    pass



    def calc_f(self, motif_sets, c_id, longest_delta_rate=5):
        window_rate = self.window_rate
        t1 = perf_counter()
        Y = [[normalize_shift_scale(self.db[s[0]:s[1], c_id], self.scale)] for s in motif_sets]
        
        X = [list(range(s[1] - s[0])) for s in motif_sets]
        X = [[a / (len(i) - 1) for a in i] for i in X]
        Y_arrays = []
        for y_item in Y:
            if isinstance(y_item, list) and len(y_item) > 0:
                y_array = np.asarray(y_item[0]).flatten()
            else:
                y_array = np.asarray(y_item).flatten()
            Y_arrays.append(y_array)
        t2 = perf_counter()
        best_idx = 0
        if not self.nearly_all_forbidden:
            print('[calc_f] using the first motif_set subseq as representative (best_idx=0)')
        t3 = perf_counter()
        y_best = Y_arrays[best_idx]
        x_best = np.asarray(X[best_idx])
        (para, expr) = piecewise_regression4(x_best, y_best)
        t4 = perf_counter()
        (param, kind) = para
        if kind != 'piecewise':
            param = np.asarray(param, dtype=np.float64)
        elif kind == 'piecewise':
            breaks = np.asarray(param['breaks'], dtype=np.float64)
            slopes = np.asarray(param['slopes'], dtype=np.float64)
            intercepts = np.asarray(param['intercepts'], dtype=np.float64)
        predict_fn = select_predict_nb(kind)
        epsilon = -1
        lam_min = float('inf')
        delta = motif_sets[0][1] - motif_sets[0][0]
        n = len(motif_sets)
        S_list = [None] * n
        lam_list = np.empty(n, dtype=np.float64)
        for i in range(n):
            s_tmp = motif_sets[i]
            S_tmp = self.db[s_tmp[0]:s_tmp[1], c_id].copy()
            S_list[i] = S_tmp
            lam_list[i] = lam_of(S_tmp)
        s = motif_sets[best_idx]
        S = S_list[best_idx]
        lam = lam_list[best_idx]
        gamma_best = lam / delta
        window = int(window_rate * delta)
        x_list = [np.arange(len(S_list[i])) / (delta - 1) for i in range(n)]
        for s_id in range(n):
            s = motif_sets[s_id]
            S = S_list[s_id]
            lam = lam_list[s_id]
            gamma = lam / delta
            x = x_list[s_id]
            if kind != 'piecewise':
                S_pred = predict_fn(x, param) * lam + S[0]
            else:
                S_pred = predict_fn(x, breaks=breaks, slopes=slopes, intercepts=intercepts) * lam + S[0]
            if gamma * gamma_best != 0:
                ratio = gamma / gamma_best
                if ratio < 1:
                    ratio = 1.0 / ratio
                ratio = ratio ** self.gamma_penalty
                error = calc_error_local_alignment4(S, S_pred, window, self.gamma_penalty) * ratio
            else:
                error = calc_error_local_alignment4(S, S_pred, window, self.gamma_penalty)
            err_agg = self._error_aggregate(error)
            if epsilon < err_agg:
                epsilon = err_agg
            if lam < lam_min:
                lam_min = lam
        M = lam_min
        epsilon_lb = 0.5 * M + 1e-07 if self.epsilonM > 0.5 * M else self.epsilonM
        epsilon = max((epsilon_lb, epsilon))
        delta_range = (self.delta_range[0], min((longest_delta_rate * delta, self.delta_range[1])))
        epsilon += 1e-07
        lambda_lower_bound = min((2 * epsilon, lam_min - 1e-07))
        t5 = perf_counter()
        self.G[c_id].append(SubSqSet(para, c_id, epsilon, lambda_lower_bound, delta_range))
        self.G[c_id][-1].class_id = len(self.G[c_id]) - 1
        sss = self.G[c_id][-1]
        sss.gamma = gamma_best
        if self.nearly_all_forbidden:
            s = motif_sets[0]
            sss.SubSq_obj_list.append(SubSq(s[0], s[1] - s[0], [lam, lam], sss.class_id, c_id))

    def check_length(self, motif, c_id, threshold=4):
        S = self.db[motif[0]:motif[1], c_id]
        delta = len(S)
        if delta <= 1:
            self.temp_window_rate = 0
            return (0, delta)
        left_trim_max = 0
        cum_left = 0
        for i in range(1, delta):
            if S[i] > S[i - 1]:
                cum_left += 1
            elif S[i] < S[i - 1]:
                cum_left -= 1
            if np.abs(cum_left) >= threshold:
                left_trim_max = i - 1
                break
            left_trim_max = i
        right_trim_max = 0
        cum_right = 0
        for i in range(1, delta):
            if S[delta - i] > S[delta - i - 1]:
                cum_right += 1
            elif S[delta - i] < S[delta - i - 1]:
                cum_right -= 1
            if np.abs(cum_right) >= threshold:
                right_trim_max = i - 1
                break
            right_trim_max = i
        if left_trim_max + right_trim_max >= delta:
            err_full = lam_of(S - np.mean(S))
            if err_full < self.epsilonM:
                self.temp_window_rate = 0
                return (0, delta)
        left_trim = 0
        for i in range(1, min(left_trim_max + 1, delta)):
            seg = S[:i]
            if lam_of(seg - np.mean(seg)) < self.epsilonM:
                left_trim = i
            else:
                break
        right_trim = 0
        for i in range(1, min(right_trim_max + 1, delta)):
            seg = S[delta - i:]
            if lam_of(seg - np.mean(seg)) < self.epsilonM:
                right_trim = i
            else:
                break
        if left_trim + right_trim > delta:
            return (0, delta)
        elif delta - (right_trim + left_trim) < self.delta_range[0]:
            min_len = self.delta_range[0]
            effective_len = delta - (left_trim + right_trim)
            if effective_len < min_len:
                toggle = False
                while effective_len < min_len:
                    if not toggle:
                        if left_trim > 0:
                            left_trim -= 1
                        elif right_trim > 0:
                            right_trim -= 1
                        else:
                            break
                    elif right_trim > 0:
                        right_trim -= 1
                    elif left_trim > 0:
                        left_trim -= 1
                    else:
                        break
                    toggle = not toggle
                    effective_len = delta - (left_trim + right_trim)
        region = (max((left_trim - 1, 0)), min((delta - right_trim + 1, delta)))
        return region

    def double_matching(self, c_id, sampling_num=5):
        from time import perf_counter
        sub_sq_set = self.G[c_id][-1]
        (class_id, c_id) = (sub_sq_set.class_id, sub_sq_set.c_id)
        (para, kind) = sub_sq_set.f
        gamma = sub_sq_set.gamma
        data = self.db[:, c_id]
        n = len(data)
        delta_range = sub_sq_set.delta_range
        lambda_lower_bound = sub_sq_set.lambda_lower_bound
        epsilonM = sub_sq_set.epsilon
        window_rate = self.window_rate
        if kind != 'piecewise':
            para = np.asarray(para, dtype=np.float64)
        else:
            breaks = np.asarray(para['breaks'], dtype=np.float64)
            slopes = np.asarray(para['slopes'], dtype=np.float64)
            intercepts = np.asarray(para['intercepts'], dtype=np.float64)
        predict_fn = select_predict_nb(kind)
        delta_list = list(range(delta_range[1], delta_range[0] - 1, -self.double_step))
        x_norm_cache = {d: np.linspace(0, 1, d, dtype=np.float64) for d in delta_list}
        f_cache = {}
        for d in delta_list:
            x_norm = x_norm_cache[d]
            if kind != 'piecewise':
                f_cache[d] = predict_fn(x_norm, para)
            else:
                f_cache[d] = predict_fn(x_norm, breaks=breaks, slopes=slopes, intercepts=intercepts)
        window_cache = {d: int(window_rate * d) for d in delta_list}
        time0 = perf_counter()
        id = 0
        for t1 in range(n):
            id += 1
            if id % 200 == 0:
                print('id=', id, ' total:', n)
                print('Double matching time:', round(perf_counter() - time0, 3), 's')
            if t1 in self.forbidden_area:
                continue
            base = data[t1]
            for delta in delta_list:
                if t1 + delta > n:
                    continue
                S = data[t1:t1 + delta] - base
                if len(S) < delta_range[0]:
                    continue
                lam = S.max() - S.min()
                if lam < lambda_lower_bound:
                    continue
                S_pred = f_cache[delta] * lam
                error = calc_error_local_alignment4(S, S_pred, window_cache[delta], self.gamma_penalty)
                err_agg = self._error_aggregate(error)
                temp = err_agg
                if lam * gamma != 0:
                    ratio = lam / delta / gamma
                    temp = err_agg * max(ratio, 1.0 / ratio) ** self.gamma_penalty
                if temp <= epsilonM:
                    sub_sq_set.SubSq_obj_list.append(SubSq(t1, delta, [lam, lam], class_id, c_id))
                    forbid_end = t1 + int(delta * self.minimum_interval) + 1
                    self.forbidden_area.update(range(t1, forbid_end))
                    self.forbidden_area2.append((t1, forbid_end))
                    if self.nearly_all_forbidden:
                        forbidden = self.forbidden_area
                        self.permitted_area = [x for x in self.permitted_area if x not in forbidden]
                    if self.SSM is not None:
                        n_ssm = self.SSM.shape[0]
                        mask_start = max(0, min(t1, n_ssm))
                        mask_end = max(0, min(forbid_end, n_ssm))
                        if mask_start < mask_end:
                            self.SSM[mask_start:mask_end, :] = -100.0
                            self.SSM[:, mask_start:mask_end] = -100.0
                    if self.is_plot:
                        print('Matched region:', t1, 'to', t1 + delta)
                        data_subsequence = data[t1:t1 + delta]
                        plot_matched_sequences(S_pred + data_subsequence[0], data_subsequence, t1, delta, [lam, lam], window_rate=window_rate, is_save=self.is_save, gamma_penalty=self.gamma_penalty)
                    break

    def _merge_intervals(self, intervals):
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        (cur_start, cur_end) = sorted_intervals[0]
        for (start, end) in sorted_intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                (cur_start, cur_end) = (start, end)
        merged.append((cur_start, cur_end))
        return merged

    def find_continuous_region(self):
        if not self.permitted_area:
            raise RuntimeError('all areas are forbidden, cannot find motif')
        sorted_permitted = sorted(self.permitted_area)
        longest_start = sorted_permitted[0]
        longest_end = sorted_permitted[0] + 1
        longest_length = 1
        current_start = sorted_permitted[0]
        current_end = sorted_permitted[0] + 1
        current_length = 1
        for i in range(1, len(sorted_permitted)):
            if sorted_permitted[i] == sorted_permitted[i - 1] + 1:
                current_end = sorted_permitted[i] + 1
                current_length += 1
            else:
                if current_length > longest_length:
                    longest_start = current_start
                    longest_end = current_end
                    longest_length = current_length
                current_start = sorted_permitted[i]
                current_end = sorted_permitted[i] + 1
                current_length = 1
        if current_length > longest_length:
            longest_start = current_start
            longest_end = current_end
            longest_length = current_length
        motif_sets = [longest_start]
        motif_length = longest_length
        if longest_length < self.delta_range[0]:
            self.ending = True
        longest_segment = set(range(longest_start, longest_end))
        self.permitted_area = [x for x in self.permitted_area if x not in longest_segment]
        self.forbidden_area.update(longest_segment)
        return (motif_sets, motif_length)

    def post_matching(self, c_id):
        pass

def plot_matched_sequences(S_pred, data_subsequence, t1=0, delta=0, lambda_range=[0, 0], window_rate=0.1, is_save=False, is_show=False, gamma_penalty=0):
    (fig, (ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(10, 11))
    time_axis = np.arange(len(data_subsequence))
    time_axis_pred = np.arange(len(S_pred))
    L = min(len(data_subsequence), len(S_pred))
    abs_diff = calc_error_local_alignment4(data_subsequence, S_pred[:L], int(window_rate * L), gamma_penalty)
    time_axis_diff = np.arange(L)
    ax1.plot(time_axis, data_subsequence, 'b-', linewidth=2, label='original sequence', marker='o', markersize=4)
    ax1.set_xlabel('time', fontsize=11)
    ax1.set_ylabel('value', fontsize=11)
    ax1.set_title(f'original sequence (t1={t1}, delta={delta})', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.plot(time_axis_pred, S_pred, 'r-', linewidth=2, label='predicted sequence', marker='s', markersize=4)
    ax2.set_xlabel('time', fontsize=11)
    ax2.set_ylabel('value', fontsize=11)
    ax2.set_title(f'predicted sequence (lambda range: [{lambda_range[0]:.4f}, {lambda_range[1]:.4f}])', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax3.plot(time_axis_diff, abs_diff, 'g-', linewidth=2, label='absolute difference', marker='^', markersize=4)
    ax3.set_xlabel('time', fontsize=11)
    ax3.set_ylabel('|error|', fontsize=11)
    ax3.set_title('absolute difference between original and predicted', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if is_save:
        results_dir = os.path.join(current_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        image_path = os.path.join(results_dir, f'matched_t1_{t1}_delta_{delta}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f'figure saved to: {image_path}')
        plt.close(fig)

def plot_sequence_with_forbidden_area(md, c_id, iteration=None):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    if md.db is None:
        print('Warning: Data not loaded')
        return
    plt.ioff()
    (fig, ax) = plt.subplots(figsize=(14, 6))
    time_series = md.db[:, c_id]
    time_points = np.arange(len(time_series))
    ax.plot(time_points, time_series, 'k-', linewidth=0.8, alpha=0.7, label='Time series')
    if md.forbidden_area:
        forbidden_list = sorted(list(md.forbidden_area))
        if forbidden_list:
            ranges = []
            start = forbidden_list[0]
            end = forbidden_list[0]
            for i in range(1, len(forbidden_list)):
                if forbidden_list[i] == end + 1:
                    end = forbidden_list[i]
                else:
                    ranges.append((start, end))
                    start = forbidden_list[i]
                    end = forbidden_list[i]
            ranges.append((start, end))
            start_label_added = False
            end_label_added = False
            for (idx, (start_idx, end_idx)) in enumerate(ranges):
                if start_idx < len(time_series) and end_idx < len(time_series):
                    if idx == 0:
                        ax.axvspan(start_idx, end_idx + 1, alpha=0.3, color='lightcoral', label='Forbidden area')
                    else:
                        ax.axvspan(start_idx, end_idx + 1, alpha=0.3, color='lightcoral')
                    if not start_label_added:
                        ax.axvline(x=start_idx, color='red', linewidth=2, linestyle='--', alpha=0.8, label='Start (Red)')
                        start_label_added = True
                    else:
                        ax.axvline(x=start_idx, color='red', linewidth=2, linestyle='--', alpha=0.8)
                    if not end_label_added:
                        ax.axvline(x=end_idx, color='blue', linewidth=2, linestyle='--', alpha=0.8, label='End (Blue)')
                        end_label_added = True
                    else:
                        ax.axvline(x=end_idx, color='blue', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    title = f'{md.dataname} - Attribute {c_id}'
    if iteration is not None:
        title += f' (Iteration {iteration})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    if md.is_save:
        results_dir = os.path.join(current_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        filename = f'{md.dataname}_attr{c_id}'
        if iteration is not None:
            filename += f'_iter{iteration}'
        filename += '_forbidden_areas.png'
        image_path = os.path.join(results_dir, filename)
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {image_path}')
    plt.close(fig)

def _motif_worker_one_attr(args):
    (dataset, error_type, is_plot, c_id, epsilonM, delta_range, A_id_set, error_percentage_tolerance, r, motif_length_range, window_rate, gamma_penalty, double_step, calc_range) = args
    md = MotifDiscovery(dataname=dataset, epsilonM=epsilonM, delta_range=delta_range, window_rate=window_rate, A_id_set=A_id_set, error_percentage_tolerance=error_percentage_tolerance, r=r, is_save=True, error_type=error_type, gamma_penalty=gamma_penalty, is_plot=is_plot, double_step=double_step)
    md.read_data()
    _apply_calc_range(md, calc_range)
    md.nearly_all_forbidden = False
    md.ending = False
    md.forbidden_area = set()
    md.forbidden_area2 = []
    md.permitted_area = None
    md.SSM = None
    md.skip_regions = []
    iteration = 0
    while len(md.forbidden_area) < md.n:
        t1 = perf_counter()
        print('=' * 30, (c_id, iteration), '=' * 30)
        md._debug_context = f'({c_id},{iteration})'
        md.temp_window_rate = None
        iteration += 1
        if not md.nearly_all_forbidden:
            (motif_sets, motif_length) = md.call_motiflets(md.db[:, c_id], motif_length_range=motif_length_range, is_plot=is_plot)
            if md.is_save:
                save_data(motif_sets, filename=f'results/temp/motif_sets_attr{c_id}.json')
        else:
            (motif_sets, motif_length) = md.find_continuous_region()
            motif_sets = [(motif_sets[0], motif_sets[0] + motif_length)]

        if md.ending:
            break
        t2 = perf_counter()
        print('<<<<<< Calling existing motif discovery >>>>>>:', round(t2 - t1, 3), 's')
        (start_idx, end_idx) = md.check_length(motif_sets[0], c_id)
        motif_sets = [(x[0] + start_idx, x[0] + end_idx) for x in motif_sets]
        motif_length = end_idx - start_idx
        md.calc_f(motif_sets, c_id)
        t3 = perf_counter()
        print('<<<<<< Group regression >>>>>>:', round(t3 - t2, 3), 's')
        md.double_matching(c_id, sampling_num=0)
        t4 = perf_counter()
        print('<<<<<< Double matching >>>>>>:', round(t4 - t2, 3), 's')
        if is_plot:
            plot_sequence_with_forbidden_area(md, c_id, iteration=iteration)
            print('-' * 60)

    md.prune_small_subsets_and_report()
    return (c_id, md.G.get(c_id, []))

def _label_token_for_filename(lab):
    if isinstance(lab, (np.integer, int)):
        return str(int(lab))
    if isinstance(lab, (np.floating, float)):
        x = float(lab)
        if np.isfinite(x) and x == int(x):
            return str(int(x))
    s = str(lab).strip()
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, '_')
    return s or 'label'

def _classification_train_sampling_tasks(csv_path, sample_num, rng):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f'Unsupported data type: {csv_path}')
    vals = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    from collections import defaultdict
    by_class = defaultdict(list)
    for i in range(len(df)):
        by_class[df.iloc[i, 0]].append(i)
    tasks = []
    for (lab, idxs) in by_class.items():
        arr = np.asarray(idxs, dtype=int)
        k = min(int(sample_num), arr.size)
        if k <= 0:
            continue
        if k >= arr.size:
            pick = arr
        else:
            pick = rng.choice(arr, size=k, replace=False)
        for sid in np.atleast_1d(pick).tolist():
            ser = vals[int(sid), :].ravel()
            ser = ser[~np.isnan(ser)]
            tasks.append((int(sid), lab, ser))
    return tasks

def _motif_worker_classification_sample(args):
    (sample_id, label, series_1d, dataset, save_path, error_type, is_plot, epsilonM, delta_range, error_percentage_tolerance, r, motif_length_range, window_rate, gamma_penalty) = args
    c_id = 0
    md = MotifDiscovery(dataname=dataset, epsilonM=epsilonM, delta_range=delta_range, window_rate=window_rate, A_id_set=[c_id], error_percentage_tolerance=error_percentage_tolerance, r=r, is_save=True, error_type=error_type, gamma_penalty=gamma_penalty, is_plot=is_plot)
    md.apply_univariate_series(series_1d)
    md.nearly_all_forbidden = False
    md.ending = False
    md.forbidden_area = set()
    md.forbidden_area2 = []
    md.permitted_area = None
    md.SSM = None
    md.skip_regions = []
    iteration = 0
    while len(md.forbidden_area) < md.n:
        t1 = perf_counter()
        print('=' * 30, (dataset, sample_id, label, c_id, iteration), '=' * 30)
        md._debug_context = f'(cls_{sample_id}_{label},{c_id},{iteration})'
        md.temp_window_rate = None
        iteration += 1
        if not md.nearly_all_forbidden:
            (motif_sets, motif_length) = md.call_motiflets(md.db[:, c_id], motif_length_range=motif_length_range, is_plot=is_plot)
            if md.is_save:
                save_data(motif_sets, filename=f'results/temp/motif_sets_cls_{dataset}_{sample_id}_{os.getpid()}.json')
        else:
            (motif_sets, motif_length) = md.find_continuous_region()
            motif_sets = [(motif_sets[0], motif_sets[0] + motif_length)]

        if md.ending:
            break
        t2 = perf_counter()
        print('<<<<<< Calling existing motif discovery >>>>>>:', round(t2 - t1, 3), 's')
        (start_idx, end_idx) = md.check_length(motif_sets[0], c_id)
        motif_sets = [(x[0] + start_idx, x[0] + end_idx) for x in motif_sets]
        motif_length = end_idx - start_idx
        md.calc_f(motif_sets, c_id)
        t3 = perf_counter()
        print('<<<<<< Group regression >>>>>>:', round(t3 - t2, 3), 's')
        md.double_matching(c_id, sampling_num=0)
        t4 = perf_counter()
        print('<<<<<< Double matching >>>>>>:', round(t4 - t2, 3), 's')
        if is_plot:
            plot_sequence_with_forbidden_area(md, c_id, iteration=iteration)
            print('-' * 60)
            
    md.prune_small_subsets_and_report()
    save_instance(md, save_path)
    return (sample_id, label, save_path)

def _apply_calc_range(md, calc_range):
    if calc_range in (None, 'All'):
        return
    if not isinstance(calc_range, tuple) or len(calc_range) != 2:
        raise ValueError("calc_range must be 'All'/None or a tuple (start, end)")
    (start, end) = (int(calc_range[0]), int(calc_range[1]))
    md.db = md.db[start:end, :]
    (md.n, md.m) = md.db.shape

def _resolve_selected_cols(selected_cols, total_cols):
    if selected_cols in (None, 'All'):
        return list(range(total_cols))
    if not isinstance(selected_cols, tuple) or len(selected_cols) != 2:
        raise ValueError("selected_cols must be 'All'/None or a tuple (start, end)")
    (start, end) = (int(selected_cols[0]), int(selected_cols[1]))
    return list(range(total_cols))[start:end]

def motif_discovery2(dataset, error_type, is_plot=False, n_workers=4, is_classification=False, window_rate=None, double_step=1, selected_cols='All', calc_range='All'):
    
    if dataset in {'glucose_T1','glucose_T1_3'}:
        epsilonM = 0.1
        delta_range = [5, 150]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = window_rate if window_rate is not None else 0.05
        gamma_penalty = -0.5

    elif dataset in {'GPS', 'GPS_missing_raw'}:
        epsilonM = 1
        delta_range = [10, 150]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = -0.5
    elif dataset == 'IMU':
        epsilonM = 1
        delta_range = [70, 700]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
        double_step = 40
        selected_cols = (0, 40)
        calc_range = (0, 7000)
    elif dataset == 'MBA':
        epsilonM = 1
        delta_range = [10, 200]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
        double_step = 10
        selected_cols = 'All'
        calc_range = 'All'
    elif dataset in {'SKAB_VALVE2'}:
        epsilonM = 0.1
        delta_range = [100, 400]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
        double_step = 10
        selected_cols = 'All'
        calc_range = 'All'
    elif dataset in {'SMAP'}:
        epsilonM = 3
        delta_range = [10, 150]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
        double_step = 10
        selected_cols = 'All'
        calc_range = 'All'
    elif dataset in {'wave'}:
        epsilonM = 3
        delta_range = [500, 800]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
        double_step = 10
        selected_cols = 'All'
        calc_range = 'All'

    elif dataset == 'exchange_rate':
        epsilonM = 0.5
        delta_range = [100, 300]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
    else:
        epsilonM = 0.5
        delta_range = [100, 300]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 10))
        window_rate = 0.05
        gamma_penalty = 0
    t0 = perf_counter()
    md = MotifDiscovery(dataname=dataset, epsilonM=epsilonM, delta_range=delta_range, window_rate=window_rate, A_id_set=A_id_set, error_percentage_tolerance=error_percentage_tolerance, r=r, is_save=True, error_type=error_type, gamma_penalty=gamma_penalty, is_plot=is_plot, double_step=double_step)
    md.read_data()
    _apply_calc_range(md, calc_range)
    t1 = perf_counter()
    print('[Creating MotifDiscovery (for A_id_set)]:', round(t1 - t0, 3), 's')
    if md.A_id_set is None:
        attr_ids = list(range(md.db.shape[1]))
    else:
        attr_ids = list(md.A_id_set)
    selected_set = set(_resolve_selected_cols(selected_cols, md.db.shape[1]))
    attr_ids = [c for c in attr_ids if c in selected_set]
    worker_args = [(dataset, error_type, is_plot, c_id, epsilonM, delta_range, [c_id], error_percentage_tolerance, r, motif_length_range, window_rate, gamma_penalty, double_step, calc_range) for c_id in attr_ids]
    results = {}
    with Pool(processes=n_workers) as pool:
        for (idx, (c_id, G_cid)) in enumerate(pool.imap_unordered(_motif_worker_one_attr, worker_args), 1):
            results[c_id] = G_cid
            print(f'Processed attribute {c_id} ({idx}/{len(worker_args)})')
    for c_id in results:
        md.G[c_id] = results[c_id]
    md.prune_small_subsets_and_report()
    save_instance(md, f'results/obj/{dataset}.pkl')
    return md

def motif_discovery3(dataset, error_type, is_plot=False, n_workers=4, is_classification=True, sample_num=1, random_seed=0, project_root=None):
    if dataset == 'Coffee':
        epsilonM = 0.1
        delta_range = [5, 30]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 5))
        window_rate = 0.05
        gamma_penalty = -0.5
    elif dataset == 'Lighting7':
        epsilonM = 0.1
        delta_range = [5, 30]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 5))
        window_rate = 0.05
        gamma_penalty = -0.5

    elif dataset == 'Trace':
        epsilonM = 0.1
        delta_range = [10, 40]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 5))
        window_rate = 0.05
        gamma_penalty = -0.5
    else:
        epsilonM = 0.1
        delta_range = [10, 150]
        A_id_set = None
        error_percentage_tolerance = 0.15
        r = 8
        motif_length_range = list(range(delta_range[0], delta_range[1], 5))
        window_rate = 0.05
        gamma_penalty = -0.5

    if is_classification:
        root = project_root if project_root is not None else current_dir
        csv_path = os.path.join(root, 'data', dataset, f'{dataset}_TRAIN.csv')

        rng = np.random.default_rng(random_seed)
        tasks = _classification_train_sampling_tasks(csv_path, sample_num, rng)
        out_dir = os.path.join(root, 'results', 'classification', dataset, 'motif_res')
        if os.path.isdir(out_dir):
            try:
                shutil.rmtree(out_dir)
            except Exception:
                for fn in os.listdir(out_dir):
                    fp = os.path.join(out_dir, fn)
                    try:
                        if os.path.isfile(fp) or os.path.islink(fp):
                            os.remove(fp)
                        elif os.path.isdir(fp):
                            shutil.rmtree(fp)
                    except Exception:
                        pass
        os.makedirs(out_dir, exist_ok=True)
        worker_args = []
        for (sample_id, label, ser) in tasks:
            lt = _label_token_for_filename(label)
            save_path = os.path.join(out_dir, f'{sample_id}_{lt}.pkl')
            worker_args.append((sample_id, label, ser, dataset, save_path, error_type, is_plot, epsilonM, delta_range, error_percentage_tolerance, r, motif_length_range, window_rate, gamma_penalty))
        if not worker_args:
            print('[motif_discovery3] there are no tasks to process (check CSV is empty or sample_num)')
            return {'output_dir': out_dir, 'saved': [], 'n_tasks': 0}
        saved = []
        with Pool(processes=n_workers) as pool:
            for (idx, ret) in enumerate(pool.imap_unordered(_motif_worker_classification_sample, worker_args), 1):
                saved.append(ret)
                print(f'[motif_discovery3 classification] {idx}/{len(worker_args)} -> {ret[2]}')
        return {'output_dir': out_dir, 'saved': saved, 'n_tasks': len(worker_args)}
    t0 = perf_counter()
    md = MotifDiscovery(dataname=dataset, epsilonM=epsilonM, delta_range=delta_range, window_rate=window_rate, A_id_set=A_id_set, error_percentage_tolerance=error_percentage_tolerance, r=r, is_save=True, error_type=error_type, gamma_penalty=gamma_penalty, is_plot=is_plot)
    md.read_data()
    t1 = perf_counter()
    print('[Creating MotifDiscovery (for A_id_set)]:', round(t1 - t0, 3), 's')
    if md.A_id_set is None:
        attr_ids = list(range(md.db.shape[1]))
    else:
        attr_ids = list(md.A_id_set)
    worker_args = [(dataset, error_type, is_plot, c_id, epsilonM, delta_range, [c_id], error_percentage_tolerance, r, motif_length_range, window_rate) for c_id in attr_ids]
    results = {}
    with Pool(processes=n_workers) as pool:
        for (idx, (c_id, G_cid)) in enumerate(pool.imap_unordered(_motif_worker_one_attr, worker_args), 1):
            results[c_id] = G_cid
            print(f'Processed attribute {c_id} ({idx}/{len(worker_args)})')
    for c_id in results:
        md.G[c_id] = results[c_id]
    md.prune_small_subsets_and_report()
    save_instance(md, f'results/obj/{dataset}.pkl')
    return md
if __name__ == '__main__':
    task = 1
    # task 1: motif discovery
    if task == 1:
        dataset = 'exchange_rate'
        error_type = 'max'
        t1 = perf_counter()
        motif_discovery2(dataset, error_type, is_plot=False, n_workers=8)
        t2 = perf_counter()
        print('Time:', round(t2 - t1, 6))
    # task 2: motif discovery for classification
    elif task == 2:
        dataset = 'Trace'
        error_type = 'max'
        t1 = perf_counter()
        motif_discovery3(dataset, error_type, is_plot=False, n_workers=8, is_classification=True, sample_num=20, random_seed=0, project_root=None)
        t2 = perf_counter()
        print('Time:', round(t2 - t1, 6))
