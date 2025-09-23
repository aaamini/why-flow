
from __future__ import annotations
from typing import Any, Sequence, List, Dict, Tuple
import math, os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from new_metrics import pr_knn, pr_knn_conditioned
from methods import MethodSpec, make_evf_generators
from data import load_dataset

# Config
DATASET = "2circles"
N_TRAIN = 50
N_REAL  = 2000
K       = 3

P_GEN  = 0.95
P_REAL = 0.50

T_VALUES     = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
STEP_VALUES  = [2, 4, 6, 8, 10, 12, 16]
N_EVAL       = N_REAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_train, Y_real = load_dataset(DATASET, N_TRAIN, N_REAL, device=device)

gens = make_evf_generators(Y_train)
methods: List[MethodSpec] = [
    MethodSpec(name="Exact x_t",    params=T_VALUES,    generator=gens.exact_xt,         n_samples=N_EVAL),
    MethodSpec(name="Euler-1",      params=T_VALUES,    generator=gens.euler_one_step,   n_samples=N_EVAL),
    MethodSpec(name="D-ODE (rk2)",  params=STEP_VALUES, generator=lambda s,n: gens.dode(int(s), n, t1=1.0, method="rk2"), n_samples=N_EVAL),
]

VARIANTS = [
    ("vanilla",           0.0,   0.0),
    ("gen_novel_95",      P_GEN, 0.0),
    ("real_novel_50",     0.0,   P_REAL),
    ("both_gen95_real50", P_GEN, P_REAL),
]

def eval_methods(methods: List[MethodSpec], Y_train: torch.Tensor, Y_real: torch.Tensor, k: int) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # base = pr_knn(Y_real, Y_train, k=k)
    # records.append({"method": "Train", "param": math.nan, "variant": "vanilla",
    #                 "precision": base["precision"], "recall": base["recall"],
    #                 "kept_real_frac": 1.0, "kept_gen_frac": 1.0})
    # for vname, pg, prc in VARIANTS[1:]:
    for vname, pg, prc in VARIANTS:
        out = pr_knn_conditioned(Y_real, Y_train, Y_train, p_gen=pg, p_real=prc, k=k)
        records.append({"method": "Train", "param": math.nan, "variant": vname,
                        "precision": out["precision"], "recall": out["recall"],
                        "kept_real_frac": out["kept_real_frac"], "kept_gen_frac": out["kept_gen_frac"]})

    for m in methods:
        for p in m.params:
            X = m.generator(p, m.n_samples)
            # base = pr_knn(Y_real, X, k=k)
            # records.append({"method": m.name, "param": float(p), "variant": "vanilla",
            #                 "precision": base["precision"], "recall": base["recall"],
            #                 "kept_real_frac": 1.0, "kept_gen_frac": 1.0})
            # for vname, pg, prc in VARIANTS[1:]:
            for vname, pg, prc in VARIANTS:
                out = pr_knn_conditioned(Y_real, X, Y_train, p_gen=pg, p_real=prc, k=k)
                records.append({"method": m.name, "param": float(p), "variant": vname,
                                "precision": out["precision"], "recall": out["recall"],
                                "kept_real_frac": out["kept_real_frac"], "kept_gen_frac": out["kept_gen_frac"]})

    return pd.DataFrame.from_records(records)

df = eval_methods(methods, Y_train, Y_real, K)

os.makedirs("./pr_outputs", exist_ok=True)
csv_path = "./pr_outputs/pr_sweep_results.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

COLORS = {
    "Exact x_t": "#1f77b4",
    "Euler-1": "#2ca02c",
    "D-ODE (rk2)": "#d62728",
    "Train": "#7f7f7f",
}
def plot_variant(df: pd.DataFrame, variant: str):
    sub = df[df["variant"] == variant].copy()
    plt.figure(figsize=(5.5, 5.5))
    plt.title(f"PR — {variant}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
    for m, g in sub.groupby("method"):
        color = COLORS.get(m, None)
        gg = g.sort_values("param")
        if m != "Train" and len(gg) > 1:
            plt.plot(gg["recall"], gg["precision"], "-", lw=1.5, color=color, alpha=0.8)
        plt.scatter(gg["recall"], gg["precision"], s=45, color=color, label=m, alpha=0.9)
        for _, row in gg.iterrows():
            p = row["param"]
            if not pd.isna(p):
                plt.annotate(f"{p:g}", (row["recall"], row["precision"]),
                             textcoords="offset points", xytext=(5,3), fontsize=8, color=color)
    plt.legend(loc="upper left", frameon=False); plt.grid(alpha=0.25)
    out = f"./pr_outputs/pr_curve_{variant}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

for vname, _, _ in VARIANTS:
    plot_variant(df, vname)
plt.show()
