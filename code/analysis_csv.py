"""
Paper 2 -- analysis from existing result CSVs only (no model training).

Reads  : ../results_CICIoV, ../results_CICIoMT, ../results_CICIoT
Writes : out/*.csv, figs/*.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(HERE, "out");  os.makedirs(OUT,  exist_ok=True)
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)

DATASETS = {"IoV": "CICIoV2024", "IoMT": "CICIoMT2024", "IoT": "CICIoT2023"}
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


def family(ds, cls):
    c = cls.upper()
    if c == "BENIGN":
        return "Benign"
    if ds == "IoV":
        return "DoS" if c == "DOS" else "Spoofing"
    if ds == "IoMT":
        if c.startswith("TCP_IP-DDOS"): return "DDoS"
        if c.startswith("TCP_IP-DOS"):  return "DoS"
        if c.startswith("MQTT"):        return "MQTT"
        if c.startswith("RECON"):       return "Recon"
        return "Spoofing"
    if c.startswith("DDOS"):  return "DDoS"
    if c.startswith("DOS"):   return "DoS"
    if c.startswith("MIRAI"): return "Mirai"
    if c.startswith("RECON") or c == "VULNERABILITYSCAN": return "Recon"
    if c in ("DNS_SPOOFING", "MITM-ARPSPOOFING"):         return "Spoofing"
    if c == "DICTIONARYBRUTEFORCE":                       return "BruteForce"
    return "Web"


# ------------------------------------------------------------------ RQ1
per_class, summary = {}, []
for ds in DATASETS:
    df = pd.read_csv(res(ds, "per_class_report.csv"), index_col=0)
    acc = df.loc["accuracy", "f1-score"]
    mac, wtd = df.loc["macro avg", "f1-score"], df.loc["weighted avg", "f1-score"]
    df = df.drop(index=["accuracy", "macro avg", "weighted avg"])
    df["family"] = [family(ds, c) for c in df.index]
    df["share"] = df["support"] / df["support"].sum()
    per_class[ds] = df
    rho, p = spearmanr(np.log10(df["support"]), df["f1-score"])
    weak = df[df["f1-score"] < 0.5]
    summary.append({
        "dataset": DATASETS[ds], "classes": len(df),
        "accuracy": acc * 100, "f1_macro": mac * 100, "f1_weighted": wtd * 100,
        "gap_acc_minus_macro": (acc - mac) * 100,
        "n_f1_below_0.5": len(weak), "n_f1_below_0.1": int((df["f1-score"] < 0.1).sum()),
        "test_share_f1_below_0.5_pct": weak["share"].sum() * 100,
        "imbalance_ratio": df["support"].max() / df["support"].min(),
        "spearman_logsupport_f1": rho, "p_value": p,
    })
    df.to_csv(os.path.join(OUT, f"perclass_{ds}.csv"))
pd.DataFrame(summary).round(4).to_csv(os.path.join(OUT, "perclass_summary.csv"), index=False)

fam_rows = []
for ds, df in per_class.items():
    for fam, g in df.groupby("family"):
        fam_rows.append({
            "dataset": DATASETS[ds], "family": fam, "n_classes": len(g),
            "support": int(g["support"].sum()), "share_pct": g["share"].sum() * 100,
            "f1_mean": g["f1-score"].mean(), "f1_min": g["f1-score"].min(),
            "f1_max": g["f1-score"].max(),
            "precision_mean": g["precision"].mean(), "recall_mean": g["recall"].mean(),
        })
pd.DataFrame(fam_rows).round(4).to_csv(os.path.join(OUT, "perclass_family.csv"), index=False)

# Figure 1: per-class F1 against test support (one panel per dataset)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
LABELS = {   # class -> (dx, dy, ha) label offset in points
    "IoV":  {"Spoofing-SPEED": (5, 3, "left"), "BENIGN": (-5, -9, "right")},
    "IoMT": {"Recon-OS_Scan": (5, 3, "left"), "Recon-VulScan": (5, 3, "left"),
             "TCP_IP-DoS-ICMP": (-5, -9, "right"), "Benign": (5, -9, "left")},
    "IoT":  {"UPLOADING_ATTACK": (4, 7, "left"), "DDOS-SYN_FLOOD": (-5, 4, "right"),
             "DOS-TCP_FLOOD": (-5, -9, "right"), "RECON-OSSCAN": (5, 3, "left")},
}
for ax, (ds, df) in zip(axes, per_class.items()):
    ax.scatter(df["support"], df["f1-score"], s=18, color=BLUE,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_ylim(-0.04, 1.06); ax.grid(axis="y", zorder=0)
    ax.set_title(f"{DATASETS[ds]} ({len(df)} classes)", fontsize=8, color=INK)
    ax.set_xlabel("Test support (log scale)")
    for name, (dx, dy, ha) in LABELS[ds].items():
        r = df.loc[name]
        ax.annotate(name, (r["support"], r["f1-score"]), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=5.5, color=MUTED)
axes[0].set_ylabel("Per-class F1")
fig.savefig(os.path.join(FIGS, "fig_perclass_support.pdf")); plt.close(fig)

# Figure 2: family-level F1 range (min .. max, dot = mean)
fam = pd.DataFrame(fam_rows)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True,
                         gridspec_kw={"width_ratios": [1, 1, 1]})
for ax, ds in zip(axes, DATASETS):
    g = fam[fam["dataset"] == DATASETS[ds]].sort_values("f1_mean")
    y = np.arange(len(g))
    ax.hlines(y, g["f1_min"], g["f1_max"], color=BLUE, linewidth=2, zorder=2)
    ax.scatter(g["f1_mean"], y, s=22, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{f} ({n})" for f, n in zip(g["family"], g["n_classes"])])
    ax.set_xlim(-0.03, 1.03); ax.grid(axis="x", zorder=0)
    ax.set_title(DATASETS[ds], fontsize=8, color=INK); ax.set_xlabel("F1 (min, mean, max)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_family_f1.pdf")); plt.close(fig)

# ------------------------------------------------------------------ RQ2 / RQ3
scores = {ds: pd.read_csv(res(ds, "feature_scores.csv")).set_index("feature") for ds in DATASETS}
selected = {ds: pd.read_csv(res(ds, "selected_features.csv"))["selected_feature"].tolist()
            for ds in DATASETS}
COLS = ["ig_score", "pca_score", "rf_score", "final_score"]
NAMES = {"ig_score": "IG", "pca_score": "PCA", "rf_score": "RF Gini", "final_score": "Composite"}

m, t = scores["IoMT"], scores["IoT"]
shared = m.index.intersection(t.index)
rows = []
for c in COLS:
    rho, p = spearmanr(m.loc[shared, c], t.loc[shared, c])
    tau, pt = kendalltau(m.loc[shared, c], t.loc[shared, c])
    rows.append({"scorer": NAMES[c], "n_features": len(shared), "spearman": rho,
                 "spearman_p": p, "kendall": tau, "kendall_p": pt})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "crossdomain_rank_agreement.csv"), index=False)

ks = list(range(1, 21))
jac = {}
for c in COLS:
    om = m[c].sort_values(ascending=False).index
    ot = t[c].sort_values(ascending=False).index
    jac[NAMES[c]] = [len(set(om[:k]) & set(ot[:k])) / len(set(om[:k]) | set(ot[:k])) for k in ks]
jac = pd.DataFrame(jac, index=ks); jac.index.name = "k"
# expected Jaccard of two random k-subsets of d features (hypergeometric mean overlap k^2/d)
d = len(shared)
jac["Random (expected)"] = [(k * k / d) / (2 * k - k * k / d) for k in ks]
jac.round(4).to_csv(os.path.join(OUT, "crossdomain_jaccard_at_k.csv"))

fig, ax = plt.subplots(figsize=(3.5, 2.5))
for name, col, ls in [("Composite", BLUE, "-"), ("IG", ORANGE, "-"),
                      ("PCA", AQUA, "-"), ("RF Gini", YELLOW, "-")]:
    ax.plot(ks, jac[name], color=col, linewidth=1.6, linestyle=ls, label=name)
ax.plot(ks, jac["Random (expected)"], color=MUTED, linewidth=1, linestyle=":", label="Random (expected)")
ax.set_xlabel("Top-$k$ features"); ax.set_ylabel("Jaccard, CICIoMT2024 vs CICIoT2023")
ax.set_ylim(0, 1.03); ax.set_xticks([1, 5, 10, 15, 20]); ax.grid(axis="y")
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.savefig(os.path.join(FIGS, "fig_jaccard_topk.pdf")); plt.close(fig)

# rank-rank scatter, IoMT vs IoT
rm = m["final_score"].rank(ascending=False).loc[shared]
rt = t["final_score"].rank(ascending=False).loc[shared]
fig, ax = plt.subplots(figsize=(3.5, 3.3))
ax.plot([1, d], [1, d], color="#e4e3df", linewidth=1, zorder=1)
ax.scatter(rm, rt, s=16, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
OFF = {"Header_Length": (7, -2, "left"), "syn_count": (-6, 3, "right"), "ack_count": (-6, 3, "right"),
       "psh_flag_number": (-6, 4, "right"), "Tot sum": (7, -7, "left"), "rst_count": (-6, 3, "right"),
       "Tot size": (-6, 5, "right"), "AVG": (4, 6, "left"), "Number": (7, -4, "left"),
       "Rate": (-6, 4, "right")}
for f in sorted(set(selected["IoMT"]) | set(selected["IoT"])):
    dx, dy, ha = OFF[f]
    ax.annotate(f, (rm[f], rt[f]), xytext=(dx, dy), ha=ha, textcoords="offset points",
                fontsize=5.5, color=MUTED)
    ax.scatter([rm[f]], [rt[f]], s=34, facecolor="none", edgecolor=ORANGE, linewidth=1.1, zorder=4)
ax.set_xlabel("Composite rank on CICIoMT2024"); ax.set_ylabel("Composite rank on CICIoT2023")
ax.set_xlim(0, d + 1); ax.set_ylim(0, d + 1); ax.set_aspect("equal")
fig.savefig(os.path.join(FIGS, "fig_rank_scatter.pdf")); plt.close(fig)

# consensus ranking (mean of the two Composite ranks)
cons = pd.DataFrame({"rank_IoMT": rm, "rank_IoT": rt})
cons["mean_rank"] = cons.mean(axis=1)
cons["sel_IoMT"] = cons.index.isin(selected["IoMT"])
cons["sel_IoT"] = cons.index.isin(selected["IoT"])
cons.sort_values("mean_rank").to_csv(os.path.join(OUT, "consensus_ranking.csv"))

# scorer agreement inside each dataset + scale / rank-preservation diagnostics
rows = []
for ds, s in scores.items():
    w = pd.read_json(res(ds, "summary.json"), typ="series")["composite_weights"]
    contrib = pd.DataFrame({"IG": w["alpha"] * s["ig_score"], "PCA": w["beta"] * s["pca_score"],
                            "RF": w["gamma"] * s["rf_score"]})
    share = contrib.sum() / contrib.sum().sum() * 100
    top = s.sort_values("final_score", ascending=False).index
    k = len(selected[ds])
    rows.append({
        "dataset": DATASETS[ds], "n_features": len(s),
        "tau_IG_PCA": kendalltau(s.ig_score, s.pca_score)[0],
        "tau_IG_RF": kendalltau(s.ig_score, s.rf_score)[0],
        "tau_PCA_RF": kendalltau(s.pca_score, s.rf_score)[0],
        "w_IG": w["alpha"], "w_PCA": w["beta"], "w_RF": w["gamma"],
        "max_IG": s.ig_score.max(), "max_PCA": s.pca_score.max(), "max_RF": s.rf_score.max(),
        "share_IG_pct": share["IG"], "share_PCA_pct": share["PCA"], "share_RF_pct": share["RF"],
        "spearman_composite_vs_attention": spearmanr(s.composite_score, s.att_score)[0],
        "spearman_composite_vs_final": spearmanr(s.composite_score, s.final_score)[0],
        "k_selected": k,
        "selected_in_topk": len(set(selected[ds]) & set(top[:k])),
        "selected_in_top2k": len(set(selected[ds]) & set(top[:2 * k])),
        "selected_median_rank": float(np.median([list(top).index(f) + 1 for f in selected[ds]])),
        "selected_ranks": ";".join(f"{f}={list(top).index(f) + 1}" for f in selected[ds]),
    })
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "scorer_diagnostics.csv"), index=False)

for f in sorted(os.listdir(OUT)):
    if f.endswith(".csv") and not f.startswith("perclass_Io"):
        print(f"\n=== {f}")
        print(pd.read_csv(os.path.join(OUT, f)).to_string(index=False))
