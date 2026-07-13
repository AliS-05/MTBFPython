"""
evolve_all_models.py — Dr. Han (2026-07-12): three MTBF evolutions per cell in one
graph, over the same expanding cumulative monthly windows the closed form uses:
  1. closed-form gamma posterior (phase 1, cheap recursion)
  2. Model 1  — common factor effects            (NUTS,  == mcmc_test run_model_1)
  3. Model 3  — common + specific interactions    (Laplace/bayesglm, == run_model_3)

Reuses the VALIDATED code as-is: designs + priors + fit functions from mcmc_test.py,
data/dates + closed form from validate_mtbf.py. This file only adds the window loop
and the 3-line plot — no model spec is redefined here.

    python evolve_all_models.py --smoke   # 2 cutoffs, sanity-check the pipeline FIRST
    python evolve_all_models.py           # full run, resumable — leave it overnight
    python evolve_all_models.py --plot-only

One Model-1 NUTS fit + one Model-3 Laplace fit per cutoff (~27 cutoffs). Resumable:
each cutoff's rows flush to CSV immediately; a restart skips cutoffs already there.
Outputs -> plots_evolution_all/.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mcmc_test as M          # build_design, mcmc_poisson_glm, bayesglm_laplace, bayesglm_prior_scales, constants
import compareGLM_Gamma as V      # load_maintenance (dates), build_cells, closed_form_trajectories

NSUB, NTYP, FTYP = M.NSUB, M.NTYP, M.FTYP
NCELL = NSUB * NTYP

WARMUP_CUTOFF = pd.Timestamp("2019-08-31")   # first window: all flights <= this
END_CUTOFF    = pd.Timestamp("2021-10-31")   # month-end covering the last flight
OUT_DIR  = "plots_evolution_all"
CSV_PATH = os.path.join(OUT_DIR, "mtbf_evolution_all.csv")
CSV_COLS = ["cutoff", "subsystem", "ftype", "n_fail_cum", "gamma",
            "m1", "m1_lo95", "m1_hi95", "m3", "max_rhat"]

CF_COLOR = "r"          # closed-form gamma (matches the phase-1 plots)
M1_COLOR = "#009E73"    # Okabe-Ito green
M3_COLOR = "#CC79A7"    # Okabe-Ito magenta


def month_end_cutoffs(a, b):
    try:
        return list(pd.date_range(a, b, freq="ME"))   # pandas >= 2.2
    except ValueError:
        return list(pd.date_range(a, b, freq="M"))


def marginal_mtbf(X, off, indc, beta):
    """Per-cell exposure-weighted fitted MTBF at the window's ACTUAL flights:
    MTBF_c = sum(tau_i) / sum(exp(off_i + X_i.beta)) over cell c's rows. Robust —
    same estimator mcmc_test's _save_marginal_mtbf uses. Returns length-NCELL array."""
    eta = np.clip(off + X @ beta, -50, 50)
    mu  = np.exp(eta)
    T   = np.exp(off)
    out = np.full(NCELL, np.nan)
    for c in range(NCELL):
        m = (indc == c)
        if m.any():
            out[c] = T[m].sum() / mu[m].sum()
    return out


def marginal_mtbf_ci(X, off, indc, betas):
    """Per-cell marginal MTBF point (at posterior-mean beta) + 95% CI from the
    per-draw marginal MTBF distribution. betas: (ndraws, p). Model 1 only —
    Model 3's Laplace CI is unusable, so it stays a point estimate."""
    bmean = betas.mean(axis=0)
    pt = np.full(NCELL, np.nan)
    lo = np.full(NCELL, np.nan)
    hi = np.full(NCELL, np.nan)
    for c in range(NCELL):
        m = (indc == c)
        if not m.any():
            continue
        Xc = X[m]; offc = off[m]; Tc = np.exp(offc).sum()
        eta = np.clip(offc[:, None] + Xc @ betas.T, -50, 50)      # (n_c, ndraws)
        d = Tc / np.exp(eta).sum(axis=0)                          # (ndraws,)
        lo[c], hi[c] = np.percentile(d, [2.5, 97.5])
        pt[c] = Tc / np.exp(np.clip(offc + Xc @ bmean, -50, 50)).sum()
    return pt, lo, hi


def gamma_at(cf, key, cutoff):
    """Closed-form gamma MTBF as of `cutoff` = the trajectory value at the latest
    flight on/before the cutoff (dates sorted, matching validate_mtbf's plotting)."""
    if key not in cf:
        return np.nan
    d, theta, _, _ = cf[key]
    keep = ~pd.isna(d)
    d2, th2 = d[keep], theta[keep]
    order = np.argsort(d2.asi8, kind="stable")
    d2, th2 = d2[order], th2[order]
    sel = d2 <= cutoff
    return float(th2[np.flatnonzero(sel)[-1]]) if sel.any() else np.nan


def load_done_cutoffs(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["cutoff"].astype(str).unique())


def run(smoke):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── full design + priors (all surviving flights), from the validated pipeline ──
    D = M.build_design()

    # per-row flight dates aligned with D's rows (SAME na.omit build_design uses) ──
    tau, subs, types, factors, dates = V.load_maintenance()
    flight_ok = np.isfinite(tau) & (tau > 0) & np.all(np.isfinite(factors), axis=1)
    row_keep  = np.repeat(flight_ok, NCELL)
    row_dates = pd.DatetimeIndex(np.repeat(dates.values, NCELL)[row_keep])
    if len(row_dates) != D.INDc.shape[0]:
        raise SystemExit(f"date/row misalignment: {len(row_dates)} dates vs {D.INDc.shape[0]} design rows")

    # ── Model 3 full interaction design (== run_model_3), + its scaled prior base ──
    Nc  = D.INDc.shape[0]
    INT = np.zeros((Nc, NCELL * D.nfm_int)); rr = np.arange(Nc)
    for m in range(D.nfm_int):
        INT[rr, D.indc * D.nfm_int + m] = D.FMIc[:, m]
    X_full  = np.column_stack([D.INDc, D.FMc, INT])
    fm_full = np.concatenate([D.mu_hyp, np.zeros(X_full.shape[1] - NCELL)])
    fs_base = np.concatenate([np.full(NCELL, M.SIG_SCALAR),
                              np.full(X_full.shape[1] - NCELL, 2.5)])

    # ── closed-form gamma trajectories (all flights), for the gamma column ──
    _, th0_raw, extras = V.build_cells()
    cf = V.closed_form_trajectories(extras["counts_all"], extras["tau_raw"],
                                    extras["dates_all"], th0_raw)

    cutoffs = month_end_cutoffs(WARMUP_CUTOFF, END_CUTOFF)
    if smoke:
        cutoffs = cutoffs[:2]
    draws = 300 if smoke else M.COMMON_DRAWS
    tune  = 300 if smoke else M.COMMON_TUNE

    done = load_done_cutoffs(CSV_PATH)
    new  = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(CSV_COLS); f.flush()

    print(f"{len(cutoffs)} cutoffs; {len(done)} already done. draws={draws}")
    for ci, cutoff in enumerate(cutoffs):
        cstr = cutoff.strftime("%Y-%m-%d")
        if cstr in done:
            print(f"[{cstr}] done, skipping"); continue
        rmask = np.asarray(row_dates <= cutoff)
        if rmask.sum() < NCELL * 2:
            print(f"[{cstr}] too few rows ({rmask.sum()}), skipping"); continue

        Xc  = D.X_common[rmask]
        Xf  = X_full[rmask]
        y   = D.yc[rmask]
        off = D.offc[rmask]
        ind = D.indc[rmask]
        print(f"[{cstr}] {int(rmask.sum()//NCELL)} flights  [{ci+1}/{len(cutoffs)}]")

        # Model 1 — common factor effects, NUTS (point = posterior mean, per R)
        _, idc = M.mcmc_poisson_glm(Xc, y, off, D.prior_mean, D.prior_sd,
                                    draws, tune, seed=1000 + ci)
        post = idc.posterior["beta"].values.reshape(-1, Xc.shape[1])
        m1, m1_lo, m1_hi = marginal_mtbf_ci(Xc, off, ind, post)
        try:
            import arviz as az
            rhat = float(az.rhat(idc)["beta"].max())
        except Exception:
            rhat = float("nan")

        # Model 3 — common + specific, Laplace/bayesglm (point = MAP, per R)
        fs = M.bayesglm_prior_scales(Xf, fs_base)
        b_map, _ = M.bayesglm_laplace(Xf, y, off, fm_full, fs)
        m3 = marginal_mtbf(Xf, off, ind, b_map)

        for c in range(NCELL):
            if not D.cell_keep[c]:
                continue
            j, k = c // NTYP + 1, FTYP[c % NTYP]
            nfail = int(y[ind == c].sum())
            w.writerow([cstr, j, k, nfail, gamma_at(cf, (j, k), cutoff),
                        m1[c], m1_lo[c], m1_hi[c], m3[c], rhat])
        f.flush()
        print(f"    rhat={rhat:.3f}  m1 med={np.nanmedian(m1):,.0f}  m3 med={np.nanmedian(m3):,.0f}")
    f.close()
    print(f"\ntable -> {CSV_PATH}")
    make_plots()


def make_plots():
    if not os.path.exists(CSV_PATH):
        print("no CSV to plot"); return
    ev = pd.read_csv(CSV_PATH, parse_dates=["cutoff"])
    has_m1_ci = "m1_lo95" in ev.columns
    _, th0_raw, extras = V.build_cells()
    cf = V.closed_form_trajectories(extras["counts_all"], extras["tau_raw"],
                                    extras["dates_all"], th0_raw)
    n = 0
    for (j, k), g in ev.groupby(["subsystem", "ftype"]):
        g = g.sort_values("cutoff")
        c = (j - 1) * NTYP + FTYP.index(k)
        th0 = th0_raw[c]
        plt.figure(figsize=(20, 10))
        plt.axhline(th0, color="blue", linestyle=":", label="Contractor MTBF")

        ref = [th0]
        if (j, k) in cf:
            cdx, theta, upper, lower = cf[(j, k)]
            cm = ~pd.isna(cdx)
            order = np.argsort(cdx[cm].asi8, kind="stable")
            cd = cdx[cm][order]
            th, up, lo = theta[cm][order], upper[cm][order], lower[cm][order]
            plt.plot(cd, th, color=CF_COLOR, lw=2, label="Closed-form gamma")
            plt.plot(cd, up, color=CF_COLOR, lw=1, linestyle="--", label="Closed-form 95% CI")
            plt.plot(cd, lo, color=CF_COLOR, lw=1, linestyle="--")
            ref += [v for v in list(th) + list(lo) if np.isfinite(v)]

        plt.plot(g.cutoff, g.m1, color=M1_COLOR, marker="o", lw=2, ms=5,
                 label="Model 1 (common)")
        if has_m1_ci:
            plt.plot(g.cutoff, g.m1_lo95, color=M1_COLOR, lw=1, linestyle=":",
                     label="Model 1 95% CI")
            plt.plot(g.cutoff, g.m1_hi95, color=M1_COLOR, lw=1, linestyle=":")
        plt.plot(g.cutoff, g.m3, color=M3_COLOR, marker="s", lw=2, ms=5,
                 label="Model 3 (common + specific)")

        # LINEAR axis in real hours, framed on the stable reference (closed-form
        # point + its lower CI + contractor). The prior-dominated early GLM spikes
        # and the closed form's exploding early upper CI clip off the top instead
        # of blowing out the hour scale.
        if len(ref) > 1:
            ylo, yhi = min(ref), max(ref)
            pad = (yhi - ylo) * 0.1 or 0.1 * max(yhi, 1.0)
            plt.ylim(ylo - pad, yhi + pad)

        plt.title(f"Subsystem {j} Type {k} — closed form vs Model 1 vs Model 3")
        plt.xlabel("Date"); plt.ylabel("MTBF (hrs)")
        plt.legend(); plt.grid(alpha=0.3); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{j}_{k}.png"), dpi=100); plt.close()
        n += 1
    print(f"{n} plots -> {OUT_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        make_plots()
    else:
        run(args.smoke)


if __name__ == "__main__":
    main()
