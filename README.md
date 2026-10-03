# Separability-Ceiling Selection (SCS)

Code and result tables for the paper

> B. Purnama, E. A. Winanto, Sharipuddin, D. Sandra, F. T. Ramadhanti,
> "A separability-ceiling criterion for training-free feature selection in
> lightweight intrusion detection for vehicular, medical, and consumer IoT
> networks" (submitted).

The separability ceiling of a feature subset is the best test score any
classifier can reach on that subset. It is obtained by counting the class
composition of the discrete patterns the subset induces. SCS is a forward
selection that adds, at each step, the feature with the highest held-out
macro-F1 ceiling. No model is trained during selection, and the tables built
along the way form a back-off lookup detector.

## Contents

| Path | What it is |
|---|---|
| `code/scs.py` | SCS selection and the back-off pattern-table detector |
| `code/light_runs.py` | Separability ceiling on CICIoV2024 and the feature-subset transfer experiment |
| `code/analysis_csv.py` | Per-class audit, ranking agreement, and filter-score diagnostics (reads stored tables only) |
| `code/out/` | Every table reported in the paper, as CSV |
| `code/figs/` | Figures of the paper |
| `results_CICIoV/`, `results_CICIoMT/`, `results_CICIoT/` | Feature scores, selected subsets, and per-class reports of the GWO wrapper baseline |

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Data

The datasets are not redistributed here. Download them from the Canadian
Institute for Cybersecurity and either place them in `data/` under the names
below or point the environment variables to your copies.

| Dataset | Default path | Environment variable | Label column |
|---|---|---|---|
| CICIoV2024 (binary encoding, all classes merged) | `data/CICIoV2024_binary.csv` | `SCS_IOV_CSV` | `specific_class` |
| CICIoMT2024 (merged, shuffled) | `data/CICIoMT2024.csv` | `SCS_IOMT_CSV` | `Label` |
| CICIoT2023 (merged) | `data/CICIoT2023.csv` | `SCS_IOT_CSV` | `Label` |

## Reproducing the results

Run the commands from the `code/` directory.

```bash
python analysis_csv.py
```

needs no dataset and regenerates the audit tables and figures in a few seconds.

```bash
python light_runs.py e1
```

computes the separability ceilings on CICIoV2024 (about six minutes on a laptop).

```bash
python light_runs.py e2
```

runs the feature-subset transfer experiment on 100,000-row subsamples.

```bash
python scs.py iov
```

runs SCS on CICIoV2024 (about 80 seconds).

```bash
python scs.py flow
```

runs SCS on CICIoMT2024 and CICIoT2023.

`analysis_csv.py` must be run before `light_runs.py e2`, because the transfer
experiment reads `out/consensus_ranking.csv`.

## Notes

- All experiments use an 80/20 stratified split with seed 42; the flow
  experiments are repeated over seeds 42, 43, and 44.
- CICIoV2024 contains 3,588 distinct feature vectors in 1,408,219 frames, and
  99.8% of the test frames of a random split also occur in training. Results on
  this dataset describe separability of the recorded frames.
- The GWO wrapper baseline itself is not included; its outputs are provided so
  that every comparison in the paper can be recomputed.
