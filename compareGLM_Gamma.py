"""
Phase-3 graphing: overlay the evolving Bayesian Poisson GLM against the
closed-form gamma-posterior MTBF, month by month, one plot per cell.

Division of labour (as of 2026-07-11):
  - evolve_mtbf.py is the model — it refits the per-cell Poisson GLM on an
    expanding cumulative monthly window and writes each window's MTBF estimate
    to plots_evolution/mtbf_evolution.csv.
  - THIS script does no fitting. It reads that CSV for the evolving GLM line,
    recomputes the closed-form gamma trajectory from dataset.xlsx, resamples the
    gamma to month-end so both series share the GLM's monthly cutoffs, and draws
    them together (contractor line + gamma + GLM, all monthly).

Run on the machine with dataset.xlsx, after evolve_mtbf.py has produced its CSV:
    python validate_mtbf.py
Outputs land in plots_validate/ (one {subsystem}_{ftype}.png per cell).

Still exports build_cells / closed_form_trajectories / newton_map — evolve_mtbf.py
imports the data pipeline + model helpers from here (single source of truth).
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ───────────────────────── CONFIG (matches mcmc.py) ─────────────────────────
DATASET_FILEPATH = "/mnt/c/Users/sefra/Downloads/dataset.xlsx"
MAINT_SHEET, MTBF_SHEET = "Failure Data", "Initial MTBF"

NSUB, FTYP    = 29, [1, 2, 6]
NTYP, NFAC    = len(FTYP), 7
NUM_PAIRS     = 13
TH0MX         = 1e8
SIG_SCALAR    = np.pi / np.sqrt(6)            # Gumbel sd, prior on intercepts

OUT_DIR = "plots_validate"
os.makedirs(OUT_DIR, exist_ok=True)
EVOLUTION_CSV = os.path.join("plots_evolution", "mtbf_evolution.csv")  # written by evolve_mtbf.py

# GLM overlay color (Okabe-Ito bluish green — distinct from the phase-1
# red trajectory and blue contractor line under deuteranopia)
MCMC_COLOR = "#009E73"

# ───────────────────── DATA LOADING (positional, per R) ─────────────────────
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

# ─────────────── CLOSED-FORM TRAJECTORIES (phase 1, from the xlsx) ───────────────
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

# ───────────────────────── MAP (penalized Newton) ─────────────────────────
def newton_map(X, y, offset, prior_mean, prior_sd, beta_init=None,
               maxit=100, tol=1e-8):
    """Posterior mode of the Poisson GLM with Normal priors (quiet).
    Returns (beta_MAP, H); posterior ~= N(beta_MAP, H^-1) (Laplace).
    Kept here because evolve_mtbf.py imports it for the evolving MAP."""
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

# ──────────────────────────────── MAIN (graphing only) ────────────────────────────────
def main():
    cells, th0_raw, extras = build_cells()
    all_cells  = [(c // NTYP + 1, FTYP[c % NTYP]) for c in range(NSUB * NTYP)]
    cell_index = {(j, k): c for c, (j, k) in enumerate(all_cells)}
    plot_cells = [(j, k) for c, (j, k) in enumerate(all_cells)
                  if np.isfinite(th0_raw[c])]

    if not os.path.exists(EVOLUTION_CSV):
        raise FileNotFoundError(
            f"{EVOLUTION_CSV} not found — run evolve_mtbf.py first; it fits the "
            "expanding-window monthly GLM and writes the evolving MTBF estimates "
            "this script graphs against the gamma posterior.")
    ev = pd.read_csv(EVOLUTION_CSV, parse_dates=["cutoff"])

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

        # evolving GLM for this cell, monthly (from evolve_mtbf.py's CSV)
        g = ev[(ev.subsystem == j) & (ev.ftype == k)].sort_values("cutoff")
        if g.empty:
            continue

        th0 = th0_raw[cell_index[(j, k)]]
        plt.figure(figsize=(20, 10))
        plt.axhline(th0, color="blue", linestyle=":", label="Contractor MTBF")

        # closed-form gamma posterior, monthly (red)
        plt.plot(mdates, mtheta, "r-o", ms=4, label="Bayes Estimate (closed form, monthly)")
        plt.plot(mdates, mupper, "r--", label="95% CI (closed form)")
        plt.plot(mdates, mlower, "r--")

        # evolving Bayesian GLM, monthly (green)
        plt.plot(g.cutoff, g.mtbf_map, color=MCMC_COLOR, marker="o", lw=2, ms=5,
                 label="Bayesian GLM (MAP), evolving")
        plt.plot(g.cutoff, g.mtbf_hi95, color=MCMC_COLOR, linestyle=":", lw=1.2,
                 label="95% credible interval (GLM)")
        plt.plot(g.cutoff, g.mtbf_lo95, color=MCMC_COLOR, linestyle=":", lw=1.2)

        # ylim from the two point estimates + contractor (wide early CIs clip
        # rather than swallow the axis — phase-1 convention)
        yv = [v for v in (list(mtheta) + list(mlower) + [th0] + list(g.mtbf_map))
              if np.isfinite(v)]
        ylo, yhi = min(yv), max(yv)
        pad = (yhi - ylo) * 0.1 or 0.1 * max(yhi, 1.0)
        plt.ylim(ylo - pad, yhi + pad)

        plt.title(f"Subsystem {j} Type {k} — closed form vs evolving Bayesian GLM (monthly)")
        plt.xlabel("Date"); plt.ylabel("MTBF (hrs)")
        plt.legend(); plt.grid(alpha=0.3); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{j}_{k}.png"), dpi=100)
        plt.close()
        n_drawn += 1

    print(f"{n_drawn} overlay plots (monthly gamma vs evolving GLM) -> {OUT_DIR}/")

if __name__ == "__main__":
    main()
