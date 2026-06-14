# -*- coding: utf-8 -*-
"""Motif mining helper used by motif_discovery.call_motiflets."""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from motiflets.plotting import Motiflets

warnings.simplefilter("ignore")


def load_time_series_from_csv(csv_path, column=None, index_col=None):
    """Load a single numeric column from a CSV file as a pandas Series."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file not found: {csv_path}")

    try:
        if index_col is not None:
            df = pd.read_csv(csv_path, index_col=index_col)
        else:
            df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"cannot read CSV: {e}") from e

    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        print(f"Using sole column: {df.columns[0]}")
    else:
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("no numeric columns in CSV")
            column = numeric_cols[0]
            print(f"Auto-selected column: {column}")

        if isinstance(column, int):
            if column >= df.shape[1]:
                raise ValueError(f"column index {column} out of range (n_cols={df.shape[1]})")
            series = df.iloc[:, column]
        elif isinstance(column, str):
            if column not in df.columns:
                raise ValueError(f"column '{column}' not found; available: {list(df.columns)}")
            series = df[column]
        else:
            raise ValueError(f"invalid column argument: {column}")

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if not isinstance(series, pd.Series):
        series = pd.Series(series.values.flatten())

    series = series.dropna()
    if len(series) == 0:
        raise ValueError("empty time series (all missing?)")

    print(f"Loaded {len(series)} points")
    return series


def mine_motifs(
    csv_path,
    k_max=20,
    motif_length=None,
    motif_length_range=None,
    ds_name=None,
    column=None,
    distance="znormed_ed",
    n_jobs=-1,
    output_dir="./motif_results",
    db=None,
    forbidden_area=None,
    debug_context=None,
    save_plots=False,
):
    """Run Motiflets on a CSV path; return (ml, motif_positions, motif_length)."""
    print(f"\nLoading data: {csv_path}")
    series = load_time_series_from_csv(csv_path, column=column)

    if ds_name is None:
        ds_name = os.path.splitext(os.path.basename(csv_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInit Motiflets (dataset={ds_name}, distance={distance})")
    ml = Motiflets(ds_name=ds_name, series=series, distance=distance, n_jobs=n_jobs)

    if motif_length is None:
        if motif_length_range is None:
            min_len = max(10, len(series) // 50)
            max_len = min(len(series) // 4, 500)
            motif_length_range = np.arange(min_len, max_len + 1, max(1, (max_len - min_len) // 10))
            print(f"Auto motif length range: {motif_length_range}")

        print(f"\nLearning best motif length (k_max={k_max})...")
        motif_length = ml.fit_motif_length(
            k_max=k_max,
            motif_length_range=motif_length_range,
            plot=False,
        )
        print(f"Best motif length: {motif_length}")

    print(f"\nMining motifs (k_max={k_max}, motif_length={motif_length})...")
    if forbidden_area is not None:
        ctx = f" [{debug_context}]" if debug_context is not None else ""
        print(f"Forbidden area{ctx}: {forbidden_area}")

    dists, motif_sets, elbow_points = ml.fit_k_elbow(
        k_max=k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifs_as_grid=False,
        forbidden_area=forbidden_area,
    )

    print("\nMining done.")
    print(f"Elbow points: {elbow_points}")
    elbow_points = [elbow_points[0]]

    print(f"\nSaving results to: {output_dir}")
    ml.motiflets = motif_sets
    ml.dists = dists
    ml.elbow_points = elbow_points

    for i, k in enumerate(elbow_points):
        if not save_plots:
            continue
        try:
            from motiflets.plotting import plot_motifset

            motif_set = motif_sets[k]
            if motif_set is None:
                print(f"Warning: k={k} motif set is None, skip plot")
                continue

            if isinstance(motif_set, np.ndarray):
                if motif_set.ndim == 0:
                    motif_set = np.array([motif_set.item()])
                if motif_set.ndim > 1:
                    motif_set = motif_set.flatten()
            elif not isinstance(motif_set, (list, np.ndarray)):
                motif_set = np.array([motif_set])

            fig, ax = plot_motifset(
                ds_name,
                series,
                motifsets=[motif_set],
                dist=dists[k],
                motif_length=motif_length,
                show=False,
            )
            plt.savefig(os.path.join(output_dir, f"{ds_name}_motif_k{k}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved plot for k={k}")
        except Exception as e:
            print(f"Warning: cannot plot k={k}: {e}")

    summary_path = os.path.join(output_dir, f"{ds_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Motif mining summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"dataset: {ds_name}\n")
        f.write(f"length: {len(series)}\n")
        f.write(f"motif_length: {motif_length}\n")
        f.write(f"k_max: {k_max}\n")
        f.write(f"distance: {distance}\n")
        f.write(f"elbow_points: {elbow_points}\n\n")
        for k in elbow_points:
            motif_set = sorted(motif_sets[k])
            dist = dists[k]
            f.write(f"k={k}:\n")
            f.write(f"  extent: {dist:.6f}\n")
            f.write(f"  positions: {motif_set}\n")
            f.write(f"  count: {len(motif_set)}\n\n")

    print(f"Summary saved to: {summary_path}")
    return ml, motif_sets[k], motif_length


def main():
    parser = argparse.ArgumentParser(description="Motif mining CLI wrapper")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file path")
    parser.add_argument("--column", type=str, default=None, help="Column name or index")
    parser.add_argument("--k_max", type=int, default=20, help="Max motif set size")
    parser.add_argument("--motif_length", type=int, default=None, help="Fixed motif length")
    parser.add_argument("--motif_length_range", type=int, nargs="+", default=None)
    parser.add_argument("--ds_name", type=str, default=None)
    parser.add_argument("--distance", type=str, default="znormed_ed",
                        choices=["znormed_ed", "euclidean", "cosine", "CID"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="./motif_results")
    args = parser.parse_args()

    column = args.column
    if column is not None:
        try:
            column = int(column)
        except ValueError:
            pass

    try:
        mine_motifs(
            csv_path=args.csv_path,
            k_max=args.k_max,
            motif_length=args.motif_length,
            motif_length_range=args.motif_length_range,
            ds_name=args.ds_name,
            column=column,
            distance=args.distance,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
        )
        print("Done.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Pass CLI args or call mine_motifs() directly.")
    else:
        main()
