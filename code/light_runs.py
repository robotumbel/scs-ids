"""
Paper 2 -- two light experiments (minutes on a laptop, no metaheuristic search).

E1  CICIoV2024 : separability ceiling of binary feature subsets.
                 A k-bit subset induces at most 2^k distinct inputs, so any
                 classifier is a lookup table. We count, on the same 80/20
                 split as the original run, the best accuracy such a table
                 can reach for (a) the GWO wrapper subset, (b) top-k by each ranking,
                 (c) a greedy forward search.
E2  CICIoMT2024 <-> CICIoT2023 : feature-subset transfer on a 100K subsample.

Writes : out/e1_*.csv, out/e2_*.csv, figs/fig_e1_ceiling.pdf
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
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(HERE, "out");  os.makedirs(OUT,  exist_ok=True)
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)
SEED = 42

IOV_PATH  = os.environ.get("SCS_IOV_CSV",  os.path.join(ROOT, "data", "CICIoV2024_binary.csv"))
IOMT_PATH = os.environ.get("SCS_IOMT_CSV", os.path.join(ROOT, "data", "CICIoMT2024.csv"))
IOT_PATH  = os.environ.get("SCS_IOT_CSV",  os.path.join(ROOT, "data", "CICIoT2023.csv"))

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def res(name, fname):
    return os.path.join(ROOT, f"results_CIC{name}", fname)


# ====================================================================== E1
def dup_stats(X, y, idx_tr, idx_te):
    """How many distinct full feature vectors exist, and how much of the test set was seen in training."""
    inv = np.unique(np.packbits(X, axis=1), axis=0, return_inverse=True)[1].ravel()
    n_vec = inv.max() + 1
    in_train = np.zeros(n_vec, dtype=bool); in_train[inv[idx_tr]] = True
    per_cls = pd.Series(inv).groupby(y).nunique()
    out = pd.DataFrame([{
        "n_rows": len(y), "distinct_vectors": int(n_vec),
        "distinct_in_train": int(in_train.sum()),
        "test_rows_seen_in_train_pct": in_train[inv[idx_te]].mean() * 100,
        **{f"distinct_class_{c}": int(v) for c, v in per_cls.items()},
    }])
    out.to_csv(os.path.join(OUT, "e1_duplication.csv"), index=False)
    print(out.T.to_string(header=False))


def plot_ceiling(e1):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for name, colr, lab in [("top-k composite", BLUE, "Top-$k$ composite"), ("top-k IG", ORANGE, "Top-$k$ IG"),
                            ("top-k RF Gini", AQUA, "Top-$k$ RF Gini"), ("top-k PCA", "#4a3aa7", "Top-$k$ PCA"),
                            ]:
        g = e1[e1["subset"] == name]
        ax.plot(g["k"], g["ceiling_f1_macro"], color=colr, linewidth=1.6, label=lab)
    scs_path = os.path.join(OUT, "scs_iov.csv")
    if os.path.exists(scs_path):
        g = pd.read_csv(scs_path); g = g[g["method"] == "SCS"]
        ax.plot(g["k"], g["table_f1_macro"], color=YELLOW, linewidth=1.6, label="SCS (proposed)")
    a = e1[e1["subset"] == "GWO wrapper subset"].iloc[0]
    ax.scatter([a["k"]], [a["ceiling_f1_macro"]], s=30, color=INK, edgecolor="white",
               linewidth=0.8, zorder=5, label="Wrapper subset ($k$=6)")
    ax.set_xlabel("Number of binary features $k$"); ax.set_ylabel("Macro-F1 ceiling (%)")
    ax.set_ylim(0, 103); ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.savefig(os.path.join(FIGS, "fig_e1_ceiling.pdf")); plt.close(fig)


def run_e1():
    t0 = time.time()
    class_map = {0: "BENIGN", 1: "DoS", 2: "Spoofing-GAS", 3: "Spoofing-RPM",
                 4: "Spoofing-SPEED", 5: "Spoofing-STEERING"}
    scores = pd.read_csv(res("IoV", "feature_scores.csv"))
    feats = scores["feature"].tolist()                       # the 75 non-constant features
    selected = pd.read_csv(res("IoV", "selected_features.csv"))["selected_feature"].tolist()

    df = pd.read_csv(IOV_PATH, usecols=feats + ["specific_class"],
                     dtype={f: "int8" for f in feats})
    y = LabelEncoder().fit_transform(df["specific_class"].map(class_map).astype(str))
    names = sorted(class_map.values())
    n_cls = len(names)
    X = df[feats].to_numpy(dtype=np.int8)
    del df
    assert set(np.unique(X)) <= {0, 1}, "features are expected to be binary"

    # identical split to the original run: indices depend only on y, seed and test_size
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.20,
                                      random_state=SEED, stratify=y)
    col = {f: i for i, f in enumerate(feats)}
    prior = np.bincount(y[idx_tr], minlength=n_cls).astype(float)
    print(f"[E1] loaded {X.shape} in {time.time() - t0:.1f}s, train={len(idx_tr):,} test={len(idx_te):,}")

    def keys_of(subset):
        if len(subset) > 62:                                  # would overflow an int64 key
            cols = [col[f] for f in subset]
            return np.unique(np.packbits(X[:, cols], axis=1), axis=0, return_inverse=True)[1].ravel()
        k = np.zeros(len(y), dtype=np.int64)
        for f in subset:
            k = (k << 1) | X[:, col[f]]
        return np.unique(k, return_inverse=True)[1]

    def table(subset):
        inv = keys_of(subset)
        n_pat = inv.max() + 1
        cnt = np.bincount(inv[idx_tr] * n_cls + y[idx_tr], minlength=n_pat * n_cls)
        return inv, cnt.reshape(n_pat, n_cls).astype(float)

    def evaluate(subset, tag):
        inv, cnt = table(subset)
        seen = cnt.sum(axis=1) > 0
        maj = int(np.argmax(prior))
        pred_map = np.where(seen, cnt.argmax(axis=1), maj)                 # frequency-optimal table
        pred_bal = np.where(seen, (cnt / prior).argmax(axis=1), maj)       # class-balanced table
        yt = y[idx_te]
        pm, pb = pred_map[inv[idx_te]], pred_bal[inv[idx_te]]
        mixed = (cnt > 0).sum(axis=1) > 1
        return {
            "subset": tag, "k": len(subset),
            "patterns_observed": int(seen.sum()), "patterns_possible": 2 ** len(subset),
            "patterns_mixed_class": int(mixed.sum()),
            "train_share_in_mixed_pct": cnt[mixed].sum() / cnt.sum() * 100,
            "ceiling_acc": accuracy_score(yt, pm) * 100,
            "ceiling_f1_macro": f1_score(yt, pm, average="macro", zero_division=0) * 100,
            "balanced_table_acc": accuracy_score(yt, pb) * 100,
            "balanced_table_f1_macro": f1_score(yt, pb, average="macro", zero_division=0) * 100,
            "balanced_table_bal_acc": balanced_accuracy_score(yt, pb) * 100,
        }, (inv, cnt)

    def train_acc(subset):
        _, cnt = table(subset)
        return cnt.max(axis=1).sum() / cnt.sum()

    dup_stats(X, y, idx_tr, idx_te)

    rows = []
    r, (inv, cnt) = evaluate(selected, "GWO wrapper subset")
    rows.append(r)

    # class composition of every pattern of the GWO wrapper subset
    pat = pd.DataFrame(cnt.astype(int), columns=names)
    pat = pat[pat.sum(axis=1) > 0]
    pat["n_classes"] = (pat[names] > 0).sum(axis=1)
    pat.sort_values(names[0], ascending=False).to_csv(os.path.join(OUT, "e1_wrapper_patterns.csv"), index=False)

    rankings = {
        "Composite": scores.sort_values("final_score", ascending=False)["feature"].tolist(),
        "IG":         scores.sort_values("ig_score",    ascending=False)["feature"].tolist(),
        "RF Gini":    scores.sort_values("rf_score",    ascending=False)["feature"].tolist(),
        "PCA":        scores.sort_values("pca_score",   ascending=False)["feature"].tolist(),
    }
    K_MAX = 36
    for name, order in rankings.items():
        for k in range(1, K_MAX + 1):
            rows.append(evaluate(order[:k], f"top-k {name}")[0])
        print(f"[E1] ranking {name} done ({time.time() - t0:.0f}s)")

    greedy, remaining = [], list(feats)
    for k in range(1, 13):
        best = max(remaining, key=lambda f: train_acc(greedy + [f]))
        greedy.append(best); remaining.remove(best)
        r = evaluate(greedy, "greedy forward")[0]
        r["features"] = ";".join(greedy)
        rows.append(r)
        print(f"[E1] greedy k={k:2d} +{best:9s} ceiling={r['ceiling_acc']:.4f}%  macroF1={r['ceiling_f1_macro']:.4f}%")
        if r["ceiling_acc"] >= 99.9999:
            break

    full = evaluate(feats, "all 75 features")[0]
    rows.append(full)
    e1 = pd.DataFrame(rows)
    e1.round(4).to_csv(os.path.join(OUT, "e1_ceiling.csv"), index=False)
    print(e1[e1["subset"].isin(["GWO wrapper subset", "all 75 features", "greedy forward"])].round(4).to_string(index=False))

    plot_ceiling(e1)
    print(f"[E1] done in {time.time() - t0:.0f}s")


# ====================================================================== E2
def run_e2(n_sub=100_000, seeds=(42, 43, 44)):
    import xgboost as xgb
    t0 = time.time()
    sel = {d: pd.read_csv(res(d, "selected_features.csv"))["selected_feature"].tolist()
           for d in ("IoMT", "IoT")}
    cons = pd.read_csv(os.path.join(OUT, "consensus_ranking.csv"))["feature"].tolist()

    def load(path, nrows):
        df = pd.read_csv(path, nrows=nrows)
        df = df.replace([np.inf, -np.inf], np.nan)
        y = df.pop("Label").astype(str).str.upper()
        df = df.select_dtypes(include=[np.number])
        return df.fillna(df.median(numeric_only=True)), y

    data = {"IoMT": load(IOMT_PATH, 600_000), "IoT": load(IOT_PATH, None)}
    all_feats = data["IoMT"][0].columns.tolist()
    rows = []
    for ds, other in (("IoMT", "IoT"), ("IoT", "IoMT")):
        Xf, yf = data[ds]
        k = len(sel[ds])
        for seed in seeds:
            rng = np.random.RandomState(seed)
            pick = rng.choice(len(yf), size=min(n_sub, len(yf)), replace=False)
            X, ys = Xf.iloc[pick], yf.iloc[pick]
            keep = ys.map(ys.value_counts()) >= 5            # stratified split needs a few samples
            X, ys = X[keep], ys[keep]
            y = LabelEncoder().fit_transform(ys)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
            sets = {
                "In-domain GWO wrapper subset":         sel[ds],
                "Cross-domain GWO wrapper subset":      sel[other],
                f"Consensus top-k (k={k})":      cons[:k],
                "Union of both GWO wrapper subsets":    sorted(set(sel["IoMT"]) | set(sel["IoT"])),
                f"Random k (k={k})":             list(rng.choice(all_feats, size=k, replace=False)),
                "All 39 features":               all_feats,
            }
            for tag, feats in sets.items():
                t1 = time.time()
                clf = xgb.XGBClassifier(n_estimators=120, max_depth=8, learning_rate=0.1,
                                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                        random_state=seed, n_jobs=-1, verbosity=0)
                clf.fit(Xtr[feats], ytr)
                p = clf.predict(Xte[feats])
                rows.append({"target": ds, "seed": seed, "feature_set": tag, "n_features": len(feats),
                             "n_classes": len(np.unique(y)), "n_train": len(ytr), "n_test": len(yte),
                             "accuracy": accuracy_score(yte, p) * 100,
                             "f1_macro": f1_score(yte, p, average="macro", zero_division=0) * 100,
                             "balanced_acc": balanced_accuracy_score(yte, p) * 100,
                             "fit_s": time.time() - t1})
                print(f"[E2] {ds} seed={seed} {tag:28s} acc={rows[-1]['accuracy']:.2f} "
                      f"f1={rows[-1]['f1_macro']:.2f} ({rows[-1]['fit_s']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "e2_transfer_runs.csv"), index=False)
    runs = pd.DataFrame(rows)
    agg = (runs.groupby(["target", "feature_set"], sort=False)
               .agg(n_features=("n_features", "first"), n_classes=("n_classes", "first"),
                    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                    f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                    bal_mean=("balanced_acc", "mean")).reset_index())
    agg.round(3).to_csv(os.path.join(OUT, "e2_transfer_summary.csv"), index=False)
    print(agg.round(3).to_string(index=False))
    print(f"[E2] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e1", "all"):
        run_e1()
    if which in ("e2", "all"):
        run_e2()
    if which == "plot":                       # redraw from out/e1_ceiling.csv
        plot_ceiling(pd.read_csv(os.path.join(OUT, "e1_ceiling.csv")))
