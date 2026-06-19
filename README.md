
## SubD: Discovering Subsequence Dependency Rules for Time Series Data

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the project directory:**

```bash
cd <project_directory>
```

2. **Create a virtual environment (recommended):**

```bash
# On Windows
python -m venv subd_env
subd_env\Scripts\activate

# On macOS/Linux
python3 -m venv subd_env
source subd_env/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Important Notes

> **Motiflets:** The Motiflets library is bundled under `motiflets-main/` and loaded via `sys.path` in `motif_discovery.py`. No separate pip install is required.
>
> **Working directory:** Run all scripts from the project root so that `data/` and `results/` paths resolve correctly.
>


## File Structure

- `motif_discovery.py`: motif mining (task 1 / task 2)
- `rule_discovery.py`: pairwise motif matching and valid-rule mining (task 1 / task 2 / task 3)
- `utils.py`: utilities (error metrics, I/O, regression helpers)
- `motiflets-main/`: bundled Motiflets source and `motif_mining_function.py`
- `data/`: input datasets
- `results/`: generated artifacts (created at runtime)
- `exp/`: downstream experiment scripts (classification, detection, imputation)
- `Appendix.pdf`: supplementary document (additional details, proofs, and extended experimental results)
- `requirements.txt`: Python package dependencies

## Data Preparation

### General multivariate series (`motif_discovery.py` task 1, `rule_discovery.py` task 1)

Place a CSV at `data/<dataset_name>.csv`. Each column is one attribute (numeric). Example: `data/glucose_T1_3.csv`, `data/exchange_rate.csv`.

### Classification (`motif_discovery.py` task 2, `rule_discovery.py` task 2 & 3)

Example layout for dataset `Trace`:

```
data/Trace/
  Trace_TRAIN.csv
  Trace_TEST.csv
```

- `Trace_TRAIN.csv` / `Trace_TEST.csv`: first column is the class label.


## Script Running

Configure the `task` variable and dataset parameters at the bottom of each script, then run from the project root.

### Motif discovery (`motif_discovery.py`)

| Task | Description | Entry function |
|------|-------------|----------------|
| 1 | Motif mining on multivariate CSV | `motif_discovery2(dataset, ...)` |
| 2 | Per-sample motif mining for classification | `motif_discovery3(dataset, is_classification=True, ...)` |

```bash
python motif_discovery.py
```

### Rule mining (`rule_discovery.py`)

| Task | Description | Entry function |
|------|-------------|----------------|
| 1 | Full pipeline: motif → pairwise match → valid rules | `run_full_rule_mining_pipeline(...)` |
| 2 | Classification: pairwise motif matching | `run_classification_motif_pairs(...)` |
| 3 | Classification: valid-rule mining | `run_classification_valid_rules(...)` |

```bash
python rule_discovery.py
```

Before running, set `dataset_name`, `supp_threshold`, `epsilon_rate`, `n_workers`, and related options in the `if __name__ == "__main__":` block of the corresponding script.

## Expected Output

Only newly written files are listed below (under `results/` unless noted).

### `motif_discovery.py` 

- `results/obj/<dataset>.pkl` — `MotifDiscovery` object with mined subsequence sets
- `results/temp/motif_sets_attr<c_id>.json` — intermediate Motiflets motif sets per attribute (when `is_save=True`)
- `results/<dataset>_summary.txt` — Motiflets mining summary (from bundled miner)
- `results/matched_t1_<t1>_delta_<delta>.png` — optional match plots during `double_matching` (when `is_plot=True` and `is_save=True`)

### `rule_discovery.py`

Runs motif discovery (task 1 outputs above), then:

- `results/obj_motif_pair/<dataset>.pkl` — pairwise motif match results
- `results/valid_rules/<dataset>.pkl` — valid rules keyed by RHS `(c_id, class_id)`
- `results/temp_valid_rules/*.png` — optional pairwise-match diagnostic figures when plot/save conditions are met

## Datasets

Download links to be added.

- **glucose_T1**: [UC_HT_T1DM](https://github.com/fisiologiacuantitativauc/UC_HT_T1DM)
- **exchange_rate**: [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)
- **GPS, GPS_missing_raw**: [CDI25](https://github.com/CDI25/CDI25)
- **IMU**: [mems-calib](https://github.com/dusan-nemec/mems-calib/tree/master)
- **MBA**: [market-basket-analysis](https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis)
- **SMAP**: [SMAP_MSL (TranAD)](https://github.com/imperial-qore/TranAD/tree/main/data/SMAP_MSL)
- **VALVE**: [skoltech-anomaly-benchmark-skab](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab)
- **wave**: [east-atlantic-swan-wave-periods](https://www.kaggle.com/datasets/thedevastator/east-atlantic-swan-wave-periods)
- **Trace, Coffee, Lighting7**: [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/)


Place files according to the layouts in **Data Preparation** after download.

## Appendix

The file [`Appendix.pdf`](Appendix.pdf) provides supplementary material for SubD, including additional technical details, proofs, and extended experimental results referenced in the paper.

## Tools

- **Motiflets**: Motif discovery backend used inside `motif_discovery.py`.
  - Website: [Motiflets](https://github.com/patrickzib/motiflets)
  - This repository includes a trimmed copy under `motiflets-main/`.
