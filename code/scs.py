"""
Paper 2 -- Separability-Ceiling Selection (SCS) and the back-off pattern-table detector.

SCS is a training-free forward selection. Features are discretised (binary
features as they are, continuous ones into B quantile bins). A subset maps
every sample to a discrete pattern; the detector is a table pattern -> class.
At each step the feature is added whose table gives the highest macro-F1 on a
held-out part of the training data. A pattern never seen while building the
table falls back to the prediction of the shorter subset (back-off).

Writes : out/scs_iov.csv, out/scs_flow_runs.csv, out/scs_flow_summary.csv,
         out/scs_selected.csv, figs/fig_scs_curve.pdf
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import light_runs as L

OUT, FIGS, SEED = L.OUT, L.FIGS, L.SEED


def macro_f1(y, p, C):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    used = den > 0
    return float((2 * tp[used] / den[used]).mean())


def discretise(X_ref, X, n_bins):
    """Integer codes per column; bin edges are quantiles of X_ref (training rows only)."""
    codes = np.empty(X.shape, dtype=np.int64)
    card = np.empty(X.shape[1], dtype=np.int64)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X_ref[:, j], qs))
        codes[:, j] = np.searchsorted(edges, X[:, j], side="right")
        card[j] = len(edges) + 1
    return codes, card


def table_step(base, n_base, code_j, card_j, y, build, C, pred_prev):
    """Extend patterns `base` with one feature; return (ids, n_ids, predictions for all rows, n table entries)."""
    ids = base * card_j + code_j
    n_ids = n_base * card_j
    cnt = np.bincount(ids[build] * C + y[build], minlength=n_ids * C).reshape(n_ids, C)
    seen = cnt.sum(axis=1) > 0
    pred = np.where(seen[ids], cnt.argmax(axis=1)[ids], pred_prev)
    return ids, n_ids, pred, int(seen.sum())


def scs_select(codes, card, y, fit, val, C, k_max):
    """Greedy forward selection on held-out macro-F1 of the back-off table."""
    n = len(y)
    base, n_base = np.zeros(n, dtype=np.int64), 1
    pred = np.full(n, np.bincount(y[fit], minlength=C).argmax(), dtype=np.int64)
    chosen, remaining = [], list(range(codes.shape[1]))
    for _ in range(k_max):
        best = None
        for j in remaining:
            if card[j] < 2:
                continue
            ids, n_ids, p, _ = table_step(base, n_base, codes[:, j], card[j], y, fit, C, pred)
            score = (macro_f1(y[val], p[val], C), (p[val] == y[val]).mean())
            if best is None or score > best[0]:
                best = (score, j, ids, p)
        _, j, ids, pred = best
        chosen.append(j); remaining.remove(j)
        base = np.unique(ids, return_inverse=True)[1].ravel()      # compact pattern ids
        n_base = int(base.max()) + 1
    return chosen


def table_eval(codes, card, y, build, test, C, order):
    """Build the back-off table on `build` rows along `order`; test metrics after each feature."""
    n = len(y)
    base, n_base = np.zeros(n, dtype=np.int64), 1
    pred = np.full(n, np.bincount(y[build], minlength=C).argmax(), dtype=np.int64)
    rows, entries_total = [], 0
    for k, j in enumerate(order, 1):
        ids, n_ids, pred, entries = table_step(base, n_base, codes[:, j], card[j], y, build, C, pred)
        base = np.unique(ids, return_inverse=True)[1].ravel()
        n_base = int(base.max()) + 1
        entries_total += entries
        rows.append({"k": k, "table_entries": entries, "table_entries_with_backoff": entries_total,
                     "table_acc": (pred[test] == y[test]).mean() * 100,
                     "table_f1_macro": macro_f1(y[test], pred[test], C) * 100})
    return rows


# ====================================================================== CICIoV2024
def run_iov(k_max=12):
    class_map = {0: "BENIGN", 1: "DoS", 2: "Spoofing-GAS", 3: "Spoofing-RPM",
                 4: "Spoofing-SPEED", 5: "Spoofing-STEERING"}
    scores = pd.read_csv(L.res("IoV", "feature_scores.csv"))
    feats = scores["feature"].tolist()
    wrapper = pd.read_csv(L.res("IoV", "selected_features.csv"))["selected_feature"].tolist()
    df = pd.read_csv(L.IOV_PATH, usecols=feats + ["specific_class"], dtype={f: "int8" for f in feats})
    y = LabelEncoder().fit_transform(df["specific_class"].map(class_map).astype(str)).astype(np.int64)
    codes = df[feats].to_numpy(dtype=np.int64); del df
    card = np.full(len(feats), 2, dtype=np.int64)
    C = int(y.max()) + 1
    tr, te = train_test_split(np.arange(len(y)), test_size=0.20, random_state=SEED, stratify=y)
    fit, val = train_test_split(tr, test_size=0.25, random_state=SEED, stratify=y[tr])

    t0 = time.time()
    order = scs_select(codes, card, y, fit, val, C, k_max)
    sel_s = time.time() - t0
    rows = []
    for r in table_eval(codes, card, y, tr, te, C, order):
        rows.append({"method": "SCS", **r, "features": ";".join(feats[j] for j in order[:r["k"]]),
                     "selection_s": sel_s})
    w = table_eval(codes, card, y, tr, te, C, [feats.index(f) for f in wrapper])[-1]
    rows.append({"method": "GWO wrapper subset", **w, "features": ";".join(wrapper), "selection_s": np.nan})
    for name, colname in (("Top-k composite", "final_score"), ("Top-k IG", "ig_score"), ("Top-k RF Gini", "rf_score")):
        top = scores.sort_values(colname, ascending=False)["feature"].tolist()[:k_max]
        for r in table_eval(codes, card, y, tr, te, C, [feats.index(f) for f in top]):
            if r["k"] in (6, 8, 12):
                rows.append({"method": name, **r, "features": ";".join(top[:r["k"]]), "selection_s": np.nan})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(OUT, "scs_iov.csv"), index=False)
    print(out.drop(columns="features").round(4).to_string(index=False))
    print("SCS order:", [feats[j] for j in order], f"selection {sel_s:.1f}s")


# ====================================================================== CICIoMT2024 / CICIoT2023
def run_flow(n_sub=100_000, seeds=(42, 43, 44), bins=(4, 8, 16), main_bins=8, k_max=10):
    import xgboost as xgb
    sel = {d: pd.read_csv(L.res(d, "selected_features.csv"))["selected_feature"].tolist() for d in ("IoMT", "IoT")}

    def load(path, nrows):
        df = pd.read_csv(path, nrows=nrows).replace([np.inf, -np.inf], np.nan)
        y = df.pop("Label").astype(str).str.upper()
        df = df.select_dtypes(include=[np.number])
        return df.fillna(df.median(numeric_only=True)), y

    data = {"IoMT": load(L.IOMT_PATH, 600_000), "IoT": load(L.IOT_PATH, None)}
    runs, picked = [], []
    for ds in ("IoMT", "IoT"):
        Xf, yf = data[ds]
        feats = Xf.columns.tolist()
        k_w = len(sel[ds])
        for seed in seeds:
            rng = np.random.RandomState(seed)                       # same subsample and split as light_runs.run_e2
            pick = rng.choice(len(yf), size=min(n_sub, len(yf)), replace=False)
            X, ys = Xf.iloc[pick], yf.iloc[pick]
            keep = ys.map(ys.value_counts()) >= 5
            X, ys = X[keep].reset_index(drop=True), ys[keep]
            y = LabelEncoder().fit_transform(ys).astype(np.int64)
            C = int(y.max()) + 1
            tr, te = train_test_split(np.arange(len(y)), test_size=0.20, random_state=seed, stratify=y)
            fit, val = train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
            Xv = X.to_numpy(dtype=float)

            def xgb_eval(cols):
                clf = xgb.XGBClassifier(n_estimators=120, max_depth=8, learning_rate=0.1, subsample=0.8,
                                        colsample_bytree=0.8, tree_method="hist", random_state=seed,
                                        n_jobs=-1, verbosity=0)
                clf.fit(X.iloc[tr][cols], y[tr])
                p = clf.predict(X.iloc[te][cols])
                return accuracy_score(y[te], p) * 100, f1_score(y[te], p, average="macro", zero_division=0) * 100

            for B in bins:
                codes, card = discretise(Xv[tr], Xv, B)
                t0 = time.time()
                order = scs_select(codes, card, y, fit, val, C, k_max)
                sel_s = time.time() - t0
                curve = table_eval(codes, card, y, tr, te, C, order)
                w_tab = table_eval(codes, card, y, tr, te, C, [feats.index(f) for f in sel[ds]])[-1]
                for k in sorted({k_w, k_max}):
                    cols = [feats[j] for j in order[:k]]
                    r = curve[k - 1]
                    row = {"target": ds, "seed": seed, "bins": B, "method": f"SCS (k={k})", "n_features": k,
                           "table_acc": r["table_acc"], "table_f1_macro": r["table_f1_macro"],
                           "table_entries": r["table_entries_with_backoff"], "selection_s": sel_s,
                           "xgb_acc": np.nan, "xgb_f1_macro": np.nan}
                    if B == main_bins:
                        row["xgb_acc"], row["xgb_f1_macro"] = xgb_eval(cols)
                        picked.append({"target": ds, "seed": seed, "k": k, "features": ";".join(cols)})
                    runs.append(row)
                runs.append({"target": ds, "seed": seed, "bins": B, "method": "GWO wrapper subset", "n_features": k_w,
                             "table_acc": w_tab["table_acc"], "table_f1_macro": w_tab["table_f1_macro"],
                             "table_entries": w_tab["table_entries_with_backoff"], "selection_s": np.nan,
                             "xgb_acc": np.nan, "xgb_f1_macro": np.nan})
                if B == main_bins:
                    for r in curve:
                        runs.append({"target": ds, "seed": seed, "bins": B, "method": "SCS curve", "n_features": r["k"],
                                     "table_acc": r["table_acc"], "table_f1_macro": r["table_f1_macro"],
                                     "table_entries": r["table_entries_with_backoff"], "selection_s": sel_s,
                                     "xgb_acc": np.nan, "xgb_f1_macro": np.nan})
                print(f"[SCS] {ds} seed={seed} B={B} select={sel_s:.1f}s  "
                      f"table@{k_w}: acc={curve[k_w-1]['table_acc']:.2f} f1={curve[k_w-1]['table_f1_macro']:.2f}  "
                      f"table@{k_max}: acc={curve[-1]['table_acc']:.2f} f1={curve[-1]['table_f1_macro']:.2f}  "
                      f"wrapper table: acc={w_tab['table_acc']:.2f} f1={w_tab['table_f1_macro']:.2f}", flush=True)
            pd.DataFrame(runs).to_csv(os.path.join(OUT, "scs_flow_runs.csv"), index=False)
    runs = pd.DataFrame(runs)
    pd.DataFrame(picked).to_csv(os.path.join(OUT, "scs_selected.csv"), index=False)
    agg = (runs.groupby(["target", "bins", "method", "n_features"], sort=False)
               .agg(table_acc=("table_acc", "mean"), table_acc_sd=("table_acc", "std"),
                    table_f1=("table_f1_macro", "mean"), table_f1_sd=("table_f1_macro", "std"),
                    table_entries=("table_entries", "mean"), selection_s=("selection_s", "mean"),
                    xgb_acc=("xgb_acc", "mean"), xgb_acc_sd=("xgb_acc", "std"),
                    xgb_f1=("xgb_f1_macro", "mean"), xgb_f1_sd=("xgb_f1_macro", "std")).reset_index())
    agg.round(3).to_csv(os.path.join(OUT, "scs_flow_summary.csv"), index=False)
    print(agg[agg["method"] != "SCS curve"].round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, ds in zip(axes, ("IoMT", "IoT")):
        c = agg[(agg["target"] == ds) & (agg["method"] == "SCS curve")]
        ax.plot(c["n_features"], c["table_f1"], color=L.BLUE, linewidth=1.6, marker="o", markersize=3, label="Macro-F1")
        ax.plot(c["n_features"], c["table_acc"], color=L.ORANGE, linewidth=1.6, marker="o", markersize=3, label="Accuracy")
        ax.set_title({"IoMT": "CICIoMT2024", "IoT": "CICIoT2023"}[ds], fontsize=8, color=L.INK)
        ax.set_xlabel("Number of selected features $k$"); ax.grid(axis="y"); ax.set_ylim(0, 100)
        ax.set_xticks(range(1, k_max + 1))
    axes[0].set_ylabel("Pattern-table test score (%)"); axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_scs_curve.pdf")); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("iov", "all"):
        run_iov()
    if which in ("flow", "all"):
        run_flow()
