"""
mtbf_graphs.py — consolidated graphing/analysis utilities for the MCMC/GLM MTBF
pipeline. One file with mode flags, instead of near-duplicate scripts.

Folds in:
  - compareGLM_Gamma.py + validate_mtbf_test.py — these were 99% byte-identical
    (same load_maintenance/load_theta0/build_cells/closed_form_trajectories/
    newton_map; validate_mtbf_test.py only added the EXTEND_WARMUP_LEFT carry-back
    in its main()). That duplication is gone — one copy of each function below.
  - compare_models_1_3.py — independent of the above (reads pre-existing
    plots_mcmc/mtbf_model{1,3}.csv, no data pipeline needed), now a mode instead
    of its own script.

Two roles:
  1. IMPORT-ONLY module — evolve_all_models_test.py does `import mtbf_graphs as V`
     for build_cells / closed_form_trajectories / load_maintenance / newton_map
     (single source of truth for the data pipeline + gamma math). No mode needed.
  2. STANDALONE CLI — two independent report/plot modes:
       python mtbf_graphs.py --mode validate --model 2   # gamma vs evolving Model N overlay
                                                            # plots (--model 1/2/3, default 2),
                                                            # reads plots_evolution_full/mtbf_m{N}.csv
                                                            # (evolve_all_models_test.py's per-model output)
       python mtbf_graphs.py --mode compare13             # static Model1-vs-Model3 ratio report,
                                                            # reads plots_mcmc/mtbf_model{1,3}.csv
                                                            # (mcmc_test.py --model 1/3 output)
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ───────────────────────── CONFIG (matches mcmc_test.py) ─────────────────────────
DATASET_FILEPATH = "/mnt/c/Users/sefra/Downloads/dataset.xlsx"
MAINT_SHEET, MTBF_SHEET = "Failure Data", "Initial MTBF"

NSUB, FTYP    = 29, [1, 2, 6]
NTYP, NFAC    = len(FTYP), 7
NUM_PAIRS     = 13
TH0MX         = 1e8
SIG_SCALAR    = np.pi / np.sqrt(6)            # Gumbel sd, prior on intercepts

# --mode validate — reads evolve_all_models_test.py's per-model CSVs (mtbf_m{n}.csv),
# each with columns cutoff/subsystem/ftype/n_fail_cum/gamma/m{n}/m{n}_lo95/m{n}_hi95[/m{n}_rhat]
OUT_DIR             = "plots_validate"
EVOLUTION_DIR       = "plots_evolution_full"
def evolution_csv(model): return os.path.join(EVOLUTION_DIR, f"mtbf_m{model}.csv")
MCMC_COLOR          = "#009E73"               # Okabe-Ito bluish green

# --mode compare13
MDIR    = "plots_mcmc"                        # mcmc_test.py --model 1/3 output
DIVERGE = 2.0                                 # a cell "diverges" if MTBFs differ by more than this factor


# ═══════════════════ SHARED: data + gamma pipeline (used by both modes AND ═══════════════════
# ═══════════════════ by evolve_all_models_test.py's `import mtbf_graphs as V`) ═══════════════

def load_maintenance():
    raw = pd.read_excel(DATASET_FILEPATH, sheet_name=MAINT_SHEET, header=0)
    if raw.shape[1] < 49:
        raise ValueError(f"maintenance sheet has {raw.shape[1]} columns, expected >= 49")
    def num(c): return pd.to_numeric(raw.iloc[:, c], errors="coerce").values
    tau = num(1)
    sub_pos = [3 + 3*p for p in range(NUM_PAIRS)]
    subs  = np.column_stack([num(c)   for c in sub_pos])
    types = np.column_stack([num(c+1) for c in sub_pos])
    factors = np.column_stack([num(42 + f) for f in range(NFAC)])
    dcol = raw.iloc[:, 0]
    if np.issubdtype(dcol.dtype, np.datetime64):
        dates = pd.DatetimeIndex(dcol)
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(dcol, format="%y/%m/%d",
                                                errors="coerce"))
        if dates.isna().mean() > 0.5:      # sheet stores some other date format
            dates = pd.DatetimeIndex(pd.to_datetime(dcol, errors="coerce"))
    return tau, subs, types, factors, dates


def load_theta0():
    raw = pd.read_excel(DATASET_FILEPATH, sheet_name=MTBF_SHEET, header=0)
    t = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")
    th = np.full((NSUB, NTYP), np.nan); v = t.values
    r, c = min(NSUB, v.shape[0]), min(NTYP, v.shape[1]); th[:r, :c] = v[:r, :c]
    return th


def build_cells():
    """Per-cell full-data datasets + the raw pieces the expanding windows need."""
    tau_all, subs, types, factors, dates = load_maintenance()
    nt = subs.shape[0]

    counts = np.zeros((nt, NSUB * NTYP))
    for j in range(1, NSUB + 1):
        for ti, k in enumerate(FTYP):
            counts[:, (j-1)*NTYP + ti] = ((subs == j) & (types == k)).sum(axis=1)

    # Factor3 main effect: full level dummies (R no-intercept model.matrix rule)
    f3 = factors[:, 2]; levels = np.unique(f3[np.isfinite(f3)])
    lev_idx = np.full(len(f3), -1, dtype=int)
    for i, v in enumerate(levels):
        lev_idx[f3 == v] = i
    f3_dummy = np.zeros((len(f3), len(levels)))
    ok = lev_idx >= 0
    f3_dummy[np.arange(len(f3))[ok], lev_idx[ok]] = 1.0

    FM = np.column_stack([factors[:, [0, 1]], f3_dummy, factors[:, 3:7]])

    # na.omit: drop flights with non-finite tau or factors (GLM side only —
    # the closed form keeps every flight, as phase 1 does)
    keep = np.isfinite(tau_all) & (tau_all > 0) & np.all(np.isfinite(factors), axis=1)
    if (~keep).sum():
        print(f"na.omit: dropping {(~keep).sum()} of {nt} flights (GLM side)")
    counts_all, tau_raw, dates_all = counts, tau_all, dates
    counts, FM, tau_all, dates = counts[keep], FM[keep], tau_all[keep], dates[keep]

    theta0 = load_theta0()
    th0 = theta0.reshape(-1).copy(); th0[~np.isfinite(th0)] = TH0MX
    mu_hyp = -np.euler_gamma - np.log(th0)        # digamma(1) - log(theta0)

    # Full-data design, centered (identifiability fix — see module docstring)
    X = np.column_stack([np.ones(keep.sum()), FM - FM.mean(axis=0)])
    off = np.log(tau_all)
    cells = {}
    for c in range(NSUB * NTYP):
        j, k = c // NTYP + 1, FTYP[c % NTYP]
        mu_c = np.concatenate([[mu_hyp[c]], np.zeros(FM.shape[1])])
        cells[(j, k)] = (X, counts[:, c].astype(float), off, mu_c)
    extras = {"FM_raw": FM, "counts": counts, "off": off, "dates": dates,
              "mu_hyp": mu_hyp, "th0": th0,
              "counts_all": counts_all, "tau_raw": tau_raw, "dates_all": dates_all}
    return cells, theta0.reshape(-1), extras


def closed_form_trajectories(counts, tau, dates, th0):
    """Phase-1 gamma-posterior recursion (same math as graph.py), computed from
    dataset.xlsx so both overlay lines share one data source. Per cell with a
    finite contractor theta0: after each flight, n* = 1 + cumulative failures,
    theta = (theta0 + T)/n*, chi-squared 95% bounds. Uses every flight (no
    na.omit), matching phase 1."""
    import scipy.stats as sps
    T = np.nancumsum(tau)
    out = {}
    for c in range(NSUB * NTYP):
        if not np.isfinite(th0[c]):
            continue
        j, k = c // NTYP + 1, FTYP[c % NTYP]
        n = 1 + np.cumsum(counts[:, c])
        theta = (th0[c] + T) / n
        upper = 2*(th0[c] + T) / sps.chi2.ppf(0.025, df=2*n)
        lower = 2*(th0[c] + T) / sps.chi2.ppf(0.975, df=2*n)
        out[(j, k)] = (dates, theta, upper, lower)
    return out


def newton_map(X, y, offset, prior_mean, prior_sd, beta_init=None,
               maxit=100, tol=1e-8):
    """Posterior mode of the Poisson GLM with Normal priors (quiet).
    Returns (beta_MAP, H); posterior ~= N(beta_MAP, H^-1) (Laplace)."""
    m0 = np.asarray(prior_mean, float); prec0 = 1.0 / np.asarray(prior_sd, float) ** 2
    beta = (m0 if beta_init is None else np.asarray(beta_init, float)).copy()
    def pen_ll(b):
        eta = np.clip(X @ b + offset, -50, 50)
        return np.sum(y*eta - np.exp(eta)) - 0.5*np.sum(prec0*(b-m0)**2)
    ll = pen_ll(beta)
    for _ in range(maxit):
        eta = np.clip(X @ beta + offset, -50, 50); mu = np.exp(eta)
        grad = X.T @ (y - mu) - prec0*(beta-m0)
        H = X.T @ (X * mu[:, None]) + np.diag(prec0)
        delta = np.linalg.solve(H, grad)
        step = 1.0
        while step > 1e-8 and pen_ll(beta + step*delta) < ll:
            step *= 0.5
        beta = beta + step*delta
        new = pen_ll(beta)
        if abs(new - ll) < tol*(abs(ll)+tol):
            break
        ll = new
    eta = np.clip(X @ beta + offset, -50, 50); mu = np.exp(eta)
    H = X.T @ (X * mu[:, None]) + np.diag(prec0)
    return beta, H


# ═══════════════════ MODE: validate — gamma vs evolving-GLM overlay per cell ═══════════════════

def run_validate(model=2):
    """Gamma vs evolving Model `model` (1, 2, or 3), monthly, one plot per cell.
    Reads mtbf_m{model}.csv (written by evolve_all_models_test.py alongside the
    combined evolution CSV — one file per model, see that script's docstring)."""
    mcol, locol, hicol = f"m{model}", f"m{model}_lo95", f"m{model}_hi95"
    ev_path = evolution_csv(model)
    os.makedirs(OUT_DIR, exist_ok=True)
    cells, th0_raw, extras = build_cells()
    all_cells  = [(c // NTYP + 1, FTYP[c % NTYP]) for c in range(NSUB * NTYP)]
    cell_index = {(j, k): c for c, (j, k) in enumerate(all_cells)}
    plot_cells = [(j, k) for c, (j, k) in enumerate(all_cells)
                  if np.isfinite(th0_raw[c])]

    if not os.path.exists(ev_path):
        raise FileNotFoundError(
            f"{ev_path} not found — run evolve_all_models_test.py first; it fits "
            "the expanding-window monthly GLMs and writes the per-model evolving "
            "MTBF estimates this mode graphs against the gamma posterior.")
    ev = pd.read_csv(ev_path, parse_dates=["cutoff"])

    # closed-form gamma trajectory (per-flight recursion over every xlsx flight)
    cf = closed_form_trajectories(extras["counts_all"], extras["tau_raw"],
                                  extras["dates_all"], th0_raw)

    n_drawn = 0
    for (j, k) in plot_cells:
        if (j, k) not in cf:
            continue
        cdx, theta, upper, lower = cf[(j, k)]
        cmask = ~cdx.isna()
        if cmask.sum() < 2:
            continue
        order = np.argsort(cdx[cmask].asi8, kind="stable")
        cdates = cdx[cmask][order]
        theta, upper, lower = theta[cmask][order], upper[cmask][order], lower[cmask][order]

        # Resample the gamma to month-end: each month's last flight is the
        # cumulative state as of month-end == evolve's expanding-window cutoff,
        # so gamma and GLM land on the same monthly x-points.
        gdf = pd.DataFrame({"date": pd.DatetimeIndex(cdates), "theta": theta,
                            "upper": upper, "lower": lower})
        gdf = gdf.groupby(gdf["date"].dt.to_period("M")).tail(1)
        mdates = pd.DatetimeIndex(gdf["date"])
        mtheta, mupper, mlower = gdf["theta"].values, gdf["upper"].values, gdf["lower"].values

        # evolving Model `model` for this cell, monthly (from evolve_all_models_test.py's
        # per-model CSV)
        g = ev[(ev.subsystem == j) & (ev.ftype == k)].sort_values("cutoff")
        if g.empty:
            continue

        # trim gamma to start at the first MCMC cutoff — no pre-history stretch
        start = g.cutoff.iloc[0]
        keep_m = mdates >= start
        mdates, mtheta, mupper, mlower = mdates[keep_m], mtheta[keep_m], mupper[keep_m], mlower[keep_m]

        th0 = th0_raw[cell_index[(j, k)]]
        plt.figure(figsize=(20, 10))
        plt.axhline(th0, color="blue", linestyle=":", label="Contractor MTBF")

        # closed-form gamma posterior, monthly (red)
        plt.plot(mdates, mtheta, "r-o", ms=4, label="Bayes Estimate (closed form, monthly)")
        plt.plot(mdates, mupper, "r--", label="95% CI (closed form)")
        plt.plot(mdates, mlower, "r--")

        # evolving Bayesian GLM, monthly (green)
        plt.plot(g.cutoff, g[mcol], color=MCMC_COLOR, marker="o", lw=2, ms=5,
                 label=f"Model {model} (MAP), evolving")
        plt.plot(g.cutoff, g[hicol], color=MCMC_COLOR, linestyle=":", lw=1.2,
                 label=f"95% CI (Model {model})")
        plt.plot(g.cutoff, g[locol], color=MCMC_COLOR, linestyle=":", lw=1.2)

        # ylim from the two point estimates + contractor (wide early CIs clip
        # rather than swallow the axis — phase-1 convention)
        yv = [v for v in (list(mtheta) + list(mlower) + [th0] + list(g[mcol]))
              if np.isfinite(v)]
        ylo, yhi = min(yv), max(yv)
        pad = (yhi - ylo) * 0.1 or 0.1 * max(yhi, 1.0)
        plt.ylim(ylo - pad, yhi + pad)

        plt.title(f"Subsystem {j} Type {k} — closed form vs evolving Model {model} (monthly)")
        plt.xlabel("Date"); plt.ylabel("MTBF (hrs)")
        plt.legend(); plt.grid(alpha=0.3); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{j}_{k}.png"), dpi=100)
        plt.close()
        n_drawn += 1

    print(f"{n_drawn} overlay plots (monthly gamma vs evolving Model {model}) -> {OUT_DIR}/")


# ═══════════════════ MODE: compare13 — static Model1 vs Model3 ratio report ═══════════════════

def run_compare13():
    """Answers Dr. Han's "run Model 1, see if we get the same result as Model 3".
    Reads the two per-cell point-MTBF CSVs mcmc_test.py --model {1,3} already wrote
    (plots_mcmc/mtbf_model{1,3}.csv, `mtbf_map` column — Model 1 = posterior mean,
    Model 3 = Laplace MAP, each model's own R convention). No fitting; instant,
    fully reproducible. Writes plots_mcmc/model1_vs_model3.csv."""
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
    print(f"\n--- divergent cells (>{DIVERGE:g}x), for Dr. Han ---")
    with pd.option_context("display.float_format", lambda x: f"{x:.1f}"):
        print(div[["subsystem", "ftype", "mtbf_model1", "mtbf_model3", "ratio_3over1"]]
              .to_string())

    print("\n--- divergent cells by failure type ---")
    print(f"  all cells      : " + ", ".join(f"t{t}={c}" for t, c in df.ftype.value_counts().sort_index().items()))
    print(f"  divergent cells: " + ", ".join(f"t{t}={c}" for t, c in div.ftype.value_counts().sort_index().items()))

    out = os.path.join(MDIR, "model1_vs_model3.csv")
    df.sort_values("ratio_3over1").to_csv(out)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "compare13"], required=True)
    ap.add_argument("--model", type=int, choices=[1, 2, 3], default=2,
                    help="validate mode only: which model to overlay against gamma (default 2)")
    args = ap.parse_args()
    if args.mode == "validate":
        run_validate(args.model)
    else:
        run_compare13()


if __name__ == "__main__":
    main()
