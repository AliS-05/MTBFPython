"""
Model 1 vs Model 3 per-cell MTBF comparison (answers Dr. Han's "run Model 1, see if
we get the same result as Model 3").

Model 1 = common factor effects only.  Model 3 = common + (subsystem x ftyp)-specific
interactions.  Both write a per-cell point MTBF to plots_mcmc/mtbf_model{1,3}.csv in the
`mtbf_map` column (Model 1 = posterior mean, Model 3 = Laplace MAP — each model's own R
convention, per Rcode.r).  This script only reads those two CSVs; it does no fitting, so
it's instant and fully reproducible.

    python compare_models_1_3.py

Reports: agreement at the center (median ratio + IQR), how many cells agree within 2x vs
diverge, the divergent-cell list (what Dr. Han asked for), and a breakdown of the
divergent cells by failure type.  Writes model1_vs_model3.csv for the record / to attach.
"""
import os
import numpy as np
import pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
MDIR  = os.path.join(HERE, "plots_mcmc")
DIVERGE = 2.0            # a cell "diverges" if the two MTBFs differ by more than this factor

def load(model):
    path = os.path.join(MDIR, f"mtbf_model{model}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run `python mcmc_test.py --model {model}` first")
    return pd.read_csv(path).set_index("cell")

m1, m3 = load(1), load(3)

# align on the cells both models kept (should be the same 77 finite-contractor cells)
cells = m1.index.intersection(m3.index)
df = pd.DataFrame({
    "subsystem": m1.loc[cells, "subsystem"].astype(int),
    "ftype":     m1.loc[cells, "ftype"].astype(int),
    "mtbf_model1": m1.loc[cells, "mtbf_map"],
    "mtbf_model3": m3.loc[cells, "mtbf_map"],
})
df["ratio_3over1"] = df.mtbf_model3 / df.mtbf_model1
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_3over1"])

r = df.ratio_3over1
q25, q50, q75 = r.quantile([.25, .5, .75])
within = ((r >= 1/DIVERGE) & (r <= DIVERGE)).sum()
n = len(r)

print(f"\n=== Model 1 vs Model 3 — {n} cells ===")
print(f"ratio = Model3 / Model1 (point MTBF; M1=posterior mean, M3=MAP)")
print(f"  median 3/1 = {q50:.3f}   IQR [{q25:.3f}, {q75:.3f}]")
print(f"  median 1/3 = {1/q50:.3f}   IQR [{1/q75:.3f}, {1/q25:.3f}]")
print(f"  agree within {DIVERGE:g}x: {within}/{n} ({100*within/n:.0f}%)")
print(f"  diverge >{DIVERGE:g}x:    {n-within}/{n} ({100*(n-within)/n:.0f}%)")

div = df[(r < 1/DIVERGE) | (r > DIVERGE)].sort_values("ratio_3over1")
print(f"\n--- divergent cells (>{DIVERGE:g}x) ---")
with pd.option_context("display.float_format", lambda x: f"{x:.1f}"):
    print(div[["subsystem", "ftype", "mtbf_model1", "mtbf_model3", "ratio_3over1"]]
          .to_string())

print("\n--- divergent cells by failure type ---")
print(f"  all cells      : " + ", ".join(f"t{t}={c}" for t, c in df.ftype.value_counts().sort_index().items()))
print(f"  divergent cells: " + ", ".join(f"t{t}={c}" for t, c in div.ftype.value_counts().sort_index().items()))

out = os.path.join(MDIR, "model1_vs_model3.csv")
df.sort_values("ratio_3over1").to_csv(out)
print(f"\nwrote {out}")
