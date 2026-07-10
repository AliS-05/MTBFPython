"""
Phase-3 (evolution): does the Bayesian Poisson GLM's covariate-adjusted MTBF
evolve the same way the closed-form gamma-posterior does as flights accumulate?

Dr. Han (2026-07-10 meeting): phase 3 so far fit the GLM ONCE on the full data
and drew it as a static reference line/band. Instead, refit the per-cell Poisson
GLM on an EXPANDING cumulative window and drop each window's MTBF estimate onto
the plot, so the GLM trace evolves alongside the closed-form trajectory.

Windowing:
  - warm-up chunk = all kept flights up to and including 2019-08-31,
  - then step month-end by month-end through 2021-10 (last flight 2021-10-12),
  - each step refits on ALL prior flights (cumulative — never a single month).

The model is UNCHANGED from mcmc.py stage 3 / validate_mtbf.py: same centered
per-flight design, same log(flight-hours) offset, same Normal priors, same NUTS
settings. This script only feeds it prefixes of the data. exp(-beta0) is the
covariate-adjusted MTBF; the 95% credible interval is the percentile interval of
the exp(-beta0) draws; the point estimate is the penalized-Newton MAP
(== arm::bayesglm), computed on the same window.

Centering choice: each window re-centers its factor columns on the WINDOW's own
mean (exactly what sub_dataset does with whatever rows it is handed). So the
estimate at cutoff T uses ONLY data up to T — no look-ahead — which is what makes
it a fair "alongside the gamma" comparison (the gamma recursion is also cumulative
up to T). Flip CENTER_ON_FULL_DATA = True to instead hold the reference at the
full-program average conditions (fixed goalposts across time).

This is the expensive one: ~27 month-ends x ~77 cells x (4 chains x 4000 iters)
NUTS fits — expect many hours. It is RESUMABLE: every (cutoff, cell) result is
appended to plots_evolution/mtbf_evolution.csv the moment it is computed, and a
restart skips whatever is already in that file. Run --smoke FIRST to sanity-check
the whole pipeline before committing to the full run.

    python evolve_mtbf.py --smoke       # 3 cells x 3 cutoffs, ~minutes
    python evolve_mtbf.py               # full run, resumable
    python evolve_mtbf.py --plot-only   # redraw from the CSV (e.g. mid-run)

Outputs land in plots_evolution/.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az

# Reuse the EXACT data pipeline + model helpers from the validated scripts so the
# model spec stays identical (single source of truth — nothing here redefines it).
from validate_mtbf import (
    build_cells, closed_form_trajectories, newton_map,
    NSUB, FTYP, NTYP, SIG_SCALAR, MCMC_COLOR,
)

# ───────────────────────────── CONFIG ─────────────────────────────
WARMUP_CUTOFF = pd.Timestamp("2019-08-31")   # warm-up chunk: all flights <= this
END_CUTOFF    = pd.Timestamp("2021-10-31")   # month-end covering last flight 2021-10-12

CENTER_ON_FULL_DATA = False                  # see module docstring (default: per-window)
EXTEND_WARMUP_LEFT  = True                   # carry the warm-up MTBF flat (faded) to the left edge

SUB_DRAWS, SUB_TUNE = 2000, 2000             # faithful to mcmc.py stage 3
TARGET_ACCEPT       = 0.99
CHAINS              = 4

OUT_DIR       = "plots_evolution"
CSV_BASENAME  = "mtbf_evolution"
CSV_COLS      = ["cutoff", "subsystem", "ftype", "n_fail_cum",
                 "mtbf_map", "mtbf_mean", "mtbf_median",
                 "mtbf_lo95", "mtbf_lo50", "mtbf_hi50", "mtbf_hi95",
                 "max_rhat", "divergences"]


# ───────────────────────────── HELPERS ─────────────────────────────
def month_end_cutoffs(warmup, end):
    """Warm-up month-end, then every month-end through `end` (inclusive)."""
    try:
        ce = pd.date_range(warmup, end, freq="ME")   # pandas >= 2.2
    except ValueError:
        ce = pd.date_range(warmup, end, freq="M")    # older pandas
    return list(ce)


def compile_window_model(Xw, off_w, pdim, sub_sd):
    """One compiled Poisson GLM for a whole window; swap y & prior-mean per cell
    (Xw and the offset are shared across all cells in a window because the design
    is per-FLIGHT). Falls back to per-cell pm.sample if nutpie is unavailable —
    identical model either way. Returns (fit_fn, backend_name)."""
    try:
        import nutpie
        with pm.Model() as model:
            Xd   = pm.Data("Xd",   Xw)
            offd = pm.Data("offd", off_w)
            yd   = pm.Data("yd",   np.zeros(Xw.shape[0]))
            mud  = pm.Data("mud",  np.zeros(pdim))
            beta = pm.Normal("beta", mu=mud, sigma=sub_sd, shape=pdim)
            pm.Poisson("obs", mu=pm.math.exp(pm.math.dot(Xd, beta) + offd), observed=yd)
        compiled = nutpie.compile_pymc_model(model)

        def fit(y, mu_c, seed):
            cm = compiled.with_data(yd=y.astype(float), mud=mu_c)
            return nutpie.sample(cm, draws=SUB_DRAWS, tune=SUB_TUNE, chains=CHAINS,
                                 target_accept=TARGET_ACCEPT, seed=seed,
                                 progress_bar=False)
        return fit, "nutpie"
    except Exception as e:
        print(f"  [nutpie unavailable ({e}); per-cell pm.sample fallback]")

        def fit(y, mu_c, seed):
            with pm.Model():
                b = pm.Normal("beta", mu=mu_c, sigma=sub_sd, shape=pdim)
                pm.Poisson("obs", mu=pm.math.exp(pm.math.dot(Xw, b) + off_w),
                           observed=y.astype(float))
                return pm.sample(draws=SUB_DRAWS, tune=SUB_TUNE, chains=CHAINS,
                                 cores=1, target_accept=TARGET_ACCEPT,
                                 random_seed=seed, progressbar=False)
        return fit, "pymc"


def load_done_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    prev = pd.read_csv(csv_path)
    return {(str(r.cutoff), int(r.subsystem), int(r.ftype)) for r in prev.itertuples()}


def make_plots(th0_raw, extras, plot_cells, cell_index, csv_path):
    if not os.path.exists(csv_path):
        print("no evolution CSV to plot yet"); return
    ev = pd.read_csv(csv_path, parse_dates=["cutoff"])
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
        order  = np.argsort(cdx[cmask].asi8, kind="stable")
        cdates = cdx[cmask][order]
        theta  = theta[cmask][order]
        upper  = upper[cmask][order]
        lower  = lower[cmask][order]

        g = ev[(ev.subsystem == j) & (ev.ftype == k)].sort_values("cutoff")
        if g.empty:
            continue

        th0 = th0_raw[cell_index[(j, k)]]
        plt.figure(figsize=(20, 10))
        plt.axhline(th0, color="blue", linestyle=":", label="Contractor MTBF")
        plt.plot(cdates, theta, "r-",  label="Bayes Estimate (closed form)")
        plt.plot(cdates, upper, "r--", label="95% CI (closed form)")
        plt.plot(cdates, lower, "r--")

        # continue the warm-up estimate flat to the left edge as a SOLID line: every
        # pre-warm-up flight is already folded into that first cumulative fit, and the
        # 2017–mid-2019 era is essentially failure-free so the prior-dominated value is
        # genuinely flat there. No markers — the markers on the live trace are what
        # distinguish the month-by-month fits from this carried-back segment. Set
        # EXTEND_WARMUP_LEFT=False to drop it, or add alpha=/linestyle="--" to fade/dash.
        if EXTEND_WARMUP_LEFT and cdates.min() < g.cutoff.iloc[0]:
            g0, x0 = g.iloc[0], cdates.min()
            plt.plot([x0, g0.cutoff], [g0.mtbf_map,  g0.mtbf_map],  color=MCMC_COLOR, lw=2)
            plt.plot([x0, g0.cutoff], [g0.mtbf_hi95, g0.mtbf_hi95], color=MCMC_COLOR,
                     lw=1.2, linestyle=":")
            plt.plot([x0, g0.cutoff], [g0.mtbf_lo95, g0.mtbf_lo95], color=MCMC_COLOR,
                     lw=1.2, linestyle=":")

        # evolving GLM: solid MAP line with monthly markers + dotted 95% CI lines,
        # mirroring the closed-form line/dashed-CI style and the phase-3 overlay theme
        plt.plot(g.cutoff, g.mtbf_map, color=MCMC_COLOR, marker="o", lw=2, ms=5,
                 label="Bayesian GLM (MAP), evolving")
        plt.plot(g.cutoff, g.mtbf_hi95, color=MCMC_COLOR, linestyle=":", lw=1.2,
                 label="95% credible interval (GLM)")
        plt.plot(g.cutoff, g.mtbf_lo95, color=MCMC_COLOR, linestyle=":", lw=1.2)

        # ylim from the closed form + GLM MAP points only — wide early GLM CIs are
        # allowed to clip rather than swallow the axis (phase-1 convention).
        yv = [v for v in (list(theta) + list(lower) + [th0] + list(g.mtbf_map))
              if np.isfinite(v)]
        ylo, yhi = min(yv), max(yv)
        pad = (yhi - ylo) * 0.1 or 0.1 * max(yhi, 1.0)
        plt.ylim(ylo - pad, yhi + pad)

        plt.title(f"Subsystem {j} Type {k} — closed form vs evolving Bayesian GLM")
        plt.xlabel("Date"); plt.ylabel("MTBF (hrs)")
        plt.legend(); plt.grid(alpha=0.3); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{j}_{k}.png"), dpi=100); plt.close()
        n_drawn += 1
    print(f"{n_drawn} overlay plots -> {OUT_DIR}/")


# ────────────────────────────── MAIN ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3 cells x 3 cutoffs to validate the pipeline (separate CSV)")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip fitting; redraw plots from the existing CSV")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR,
                            f"{CSV_BASENAME}{'_smoke' if args.smoke else ''}.csv")

    cells, th0_raw, extras = build_cells()
    dates   = pd.DatetimeIndex(extras["dates"])      # kept-flight dates (na.omit'd)
    FM_raw  = extras["FM_raw"]                        # per-flight factor design (kept)
    counts  = extras["counts"]                        # per-cell counts (kept flights)
    off_all = extras["off"]                           # log(flight hours), kept flights
    mu_hyp  = extras["mu_hyp"]
    pdim    = 1 + FM_raw.shape[1]
    sub_sd  = np.concatenate([[SIG_SCALAR], np.full(pdim - 1, 2.5)])
    fm_full_mean = FM_raw.mean(axis=0)               # fixed reference (optional mode)

    all_cells  = [(c // NTYP + 1, FTYP[c % NTYP]) for c in range(NSUB * NTYP)]
    cell_index = {(j, k): c for c, (j, k) in enumerate(all_cells)}
    plot_cells = [(j, k) for c, (j, k) in enumerate(all_cells)
                  if np.isfinite(th0_raw[c])]        # only cells that get overlaid

    if args.plot_only:
        make_plots(th0_raw, extras, plot_cells, cell_index, csv_path)
        return

    cutoffs = month_end_cutoffs(WARMUP_CUTOFF, END_CUTOFF)
    if args.smoke:
        cutoffs, plot_cells = cutoffs[:3], plot_cells[:3]
        print(f"[SMOKE] {len(plot_cells)} cells x {len(cutoffs)} cutoffs -> {csv_path}")

    done  = load_done_keys(csv_path)
    total = len(cutoffs) * len(plot_cells)
    print(f"{len(cutoffs)} cutoffs x {len(plot_cells)} cells = {total} fits "
          f"({len(done)} already done, resuming)")

    new_file = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    w = csv.writer(fcsv)
    if new_file:
        w.writerow(CSV_COLS); fcsv.flush()

    for ci, cutoff in enumerate(cutoffs):
        cstr  = cutoff.strftime("%Y-%m-%d")
        wmask = np.asarray(dates <= cutoff)
        nflt  = int(wmask.sum())
        if nflt < 2:
            print(f"[{cstr}] only {nflt} flights — skipping window"); continue

        todo = [(j, k) for (j, k) in plot_cells if (cstr, j, k) not in done]
        if not todo:
            print(f"[{cstr}] all {len(plot_cells)} cells already done"); continue

        Fm_w   = FM_raw[wmask]
        center = fm_full_mean if CENTER_ON_FULL_DATA else Fm_w.mean(axis=0)
        Xw     = np.column_stack([np.ones(nflt), Fm_w - center])
        off_w  = off_all[wmask]
        fit, backend = compile_window_model(Xw, off_w, pdim, sub_sd)
        print(f"[{cstr}] {nflt} flights, {len(todo)} cells to fit "
              f"({backend}) [{ci+1}/{len(cutoffs)}]")

        for (j, k) in todo:
            c    = cell_index[(j, k)]
            y_w  = counts[wmask, c].astype(float)
            mu_c = np.concatenate([[mu_hyp[c]], np.zeros(pdim - 1)])
            seed = 1000 + ci * 1000 + c

            ids   = fit(y_w, mu_c, seed)
            beta0 = ids.posterior["beta"].values.reshape(-1, pdim)[:, 0]
            d     = np.exp(-beta0)
            lo95, lo50, med, hi50, hi95 = np.percentile(d, [2.5, 25, 50, 75, 97.5])
            b_map, _ = newton_map(Xw, y_w, off_w, mu_c, sub_sd)
            try:
                rhat = float(az.rhat(ids)["beta"].max())
                ndiv = int(np.asarray(ids.sample_stats["diverging"]).sum())
            except Exception:
                rhat, ndiv = float("nan"), -1

            w.writerow([cstr, j, k, int(y_w.sum()),
                        np.exp(-b_map[0]), d.mean(), med,
                        lo95, lo50, hi50, hi95, rhat, ndiv])
            fcsv.flush()
            done.add((cstr, j, k))
            print(f"    s{j}t{k}: MAP={np.exp(-b_map[0]):,.0f} "
                  f"[{lo95:,.0f}, {hi95:,.0f}]  n={int(y_w.sum())}  rhat={rhat:.3f}")

    fcsv.close()
    print(f"\nEvolution table -> {csv_path}")
    make_plots(th0_raw, extras, plot_cells, cell_index, csv_path)


if __name__ == "__main__":
    main()
