"""
evolve_all_models_test.py — extends evolve_all_models.py with Model 2, so ONE run
produces all four MTBF traces per cell over the same expanding cumulative monthly
windows the closed form uses:
  1. closed-form gamma posterior (phase 1, cheap recursion)
  2. Model 1  — common factor effects              (NUTS,     == mcmc_test run_model_1)
  3. Model 2  — subsystem x ftyp-specific only      (NUTS x87, == mcmc_test run_model_2)
  4. Model 3  — common + specific interactions      (Laplace,  == mcmc_test run_model_3)

Reuses the VALIDATED code as-is: designs + priors + fit functions from mcmc_test.py,
data/dates + closed form from mtbf_graphs.py. This file only adds the window
loop, the Model 2 per-cell fit, and the 4-line plot — no model spec is redefined.

IMPORTANT — Model 2 uses RAW (uncentered) factor columns, matching the 2026-07-12 fix
to mcmc_test.py's run_model_2 (Rcode.r stage 3 fits raw `Fm`, not `Fm - Fm.mean()`).
The OLD evolve_mtbf.py / cummulativeMcmc.py still centers per-window — that bug was
never carried into mcmc_test.py's fix and is NOT reused here. Do not copy that file's
design construction back in.

    python evolve_all_models_test.py --smoke   # 2 cutoffs, Model 2 on 5 cells only
    python evolve_all_models_test.py           # full run, resumable — leave it overnight
    python evolve_all_models_test.py --plot-only

Cost: Model 1 and Model 3 are one fit per cutoff (shared design across all cells, as
in evolve_all_models.py). Model 2 cannot share a design across cells the same way —
it needs one NUTS fit per (cutoff, cell), ~77 cells x ~27 cutoffs ~= 2000 fits. Kept
cheap the same way evolve_mtbf.py did it: ONE nutpie model compiled per window (X and
the exposure offset are identical across cells within a window — factors/exposure are
per-FLIGHT, not per-cell), only y and the prior mean swap per cell via with_data().
Still expect this to run for hours; run --smoke first.

Resumability is per-cutoff (same as evolve_all_models.py): a cutoff's rows are only
flushed to CSV after ALL cells (Model 1 + all 77 Model 2 fits + Model 3) for that
cutoff are done. A restart skips cutoffs already fully present in the CSV.

Outputs -> plots_evolution_full/ (a NEW directory/CSV — the schema gained m2 columns,
so this does not touch or resume from plots_evolution_all/mtbf_evolution_all.csv).
Writes the combined mtbf_evolution_full.csv (all 4 series, one row per cutoff x cell —
what make_plots() reads) AND one CSV per model (mtbf_m1.csv / mtbf_m2.csv / mtbf_m3.csv,
each with gamma + n_fail_cum alongside that model's columns so it stands alone).
mtbf_graphs.py's --mode validate reads these per-model files.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse, csv
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mcmc_R_port as M              # build_design, mcmc_poisson_glm, bayesglm_laplace, bayesglm_prior_scales, constants
import mtbf_graphs as V            # load_maintenance (dates), build_cells, closed_form_trajectories

NSUB, NTYP, FTYP = M.NSUB, M.NTYP, M.FTYP
NCELL = NSUB * NTYP

WARMUP_CUTOFF = pd.Timestamp("2019-08-31")   # first window: all flights <= this
END_CUTOFF    = pd.Timestamp("2021-10-31")   # month-end covering the last flight
OUT_DIR  = "plots_evolution_full"
CSV_PATH = os.path.join(OUT_DIR, "mtbf_evolution_full.csv")   # combined — what make_plots() reads
CSV_COLS = ["cutoff", "subsystem", "ftype", "n_fail_cum", "gamma",
            "m1", "m1_lo95", "m1_hi95", "m1_rhat",
            "m2", "m2_lo95", "m2_hi95", "m2_rhat",
            "m3", "m3_lo95", "m3_hi95"]

# Per-model CSVs too (each model gets its own file, gamma/n_fail_cum included in each
# for standalone use) — same rows as the combined CSV, just split by model. mtbf_graphs.py
# --mode validate reads these (path is MODEL_CSV_PATH(n) below).
MODEL_CSV_COLS = {
    1: ["cutoff", "subsystem", "ftype", "n_fail_cum", "gamma", "m1", "m1_lo95", "m1_hi95", "m1_rhat"],
    2: ["cutoff", "subsystem", "ftype", "n_fail_cum", "gamma", "m2", "m2_lo95", "m2_hi95", "m2_rhat"],
    3: ["cutoff", "subsystem", "ftype", "n_fail_cum", "gamma", "m3", "m3_lo95", "m3_hi95"],
}

def MODEL_CSV_PATH(n):
    return os.path.join(OUT_DIR, f"mtbf_m{n}.csv")

CF_COLOR = "r"          # closed-form gamma (matches the phase-1 plots)
M1_COLOR = "#009E73"    # Okabe-Ito green
M2_COLOR = "#0072B2"    # Okabe-Ito blue
M3_COLOR = "#CC79A7"    # Okabe-Ito magenta
Y_CAP    = 2.0          # axis ceiling = this x the max of the sane models (gamma + Model 1 +
                        # Model 2 + contractor); Model 3 and its Laplace CI clip off the top
SMOKE_M2_CELLS = 5      # --smoke fits Model 2 on only this many cells (Model 1/3 still run on all)


def month_end_cutoffs(a, b):
    try:
        return list(pd.date_range(a, b, freq="ME"))   # pandas >= 2.2
    except ValueError:
        return list(pd.date_range(a, b, freq="M"))


def marginal_mtbf_ci(X, off, indc, betas, point_beta):
    """Per-cell exposure-weighted MTBF at each cell's ACTUAL flights — VERBATIM math from
    mcmc_test._save_marginal_mtbf: MTBF_c = sum(exp(off)) / sum(exp(off + Xc.beta)) over
    cell c's rows. Used for Model 1 and Model 3 (shared design across cells)."""
    off = np.asarray(off, float)
    pt = np.full(NCELL, np.nan)
    lo = np.full(NCELL, np.nan)
    hi = np.full(NCELL, np.nan)
    for c in range(NCELL):
        m = (indc == c)
        if not m.any():
            continue
        Xc = X[m]; oc = off[m]; T = np.exp(oc).sum()
        mu = np.exp(np.clip(oc[:, None] + Xc @ betas.T, -50, 50))        # (n_c, n_draws)
        d = T / mu.sum(axis=0)
        lo[c], hi[c] = np.percentile(d, [2.5, 97.5])
        pt[c] = T / np.exp(np.clip(oc + Xc @ point_beta, -50, 50)).sum()
    return pt, lo, hi


def compile_window_model(Xw, off_w, pdim, sub_sd):
    """ONE compiled Poisson GLM for a whole window; swap y & prior-mean per cell. Xw and
    off_w are fixed at compile time (shared across all cells in a window because the
    factor design + exposure are per-FLIGHT, not per-cell — every cell sees the same
    flights in the long-format design, only its failure counts differ). Falls back to
    per-cell pm.sample if nutpie is unavailable — identical model either way.
    Ported from evolve_mtbf.py's compile_window_model, minus the centering bug (see
    module docstring) — Xw here must already be RAW (uncentered) factor columns."""
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
            return nutpie.sample(cm, draws=M.SUB_DRAWS, tune=M.SUB_TUNE, chains=4,
                                 target_accept=M.TARGET_ACCEPT, seed=seed,
                                 progress_bar=False)
        return fit, "nutpie"
    except Exception as e:
        print(f"    [nutpie unavailable ({e}); per-cell pm.sample fallback]")

        def fit(y, mu_c, seed):
            with pm.Model():
                b = pm.Normal("beta", mu=mu_c, sigma=sub_sd, shape=pdim)
                pm.Poisson("obs", mu=pm.math.exp(pm.math.dot(Xw, b) + off_w),
                           observed=y.astype(float))
                return pm.sample(draws=M.SUB_DRAWS, tune=M.SUB_TUNE, chains=4,
                                 cores=1, target_accept=M.TARGET_ACCEPT,
                                 random_seed=seed, progressbar=False)
        return fit, "pymc"


def gamma_at(cf, key, cutoff):
    """Closed-form gamma MTBF as of `cutoff` = the trajectory value at the latest
    flight on/before the cutoff (dates sorted, matching compareGLM_Gamma's plotting)."""
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

    # ── Model 2 dims (== run_model_2's sub_dataset: RAW Fm, no centering) ──
    pdim2   = 1 + D.nfm
    sub_sd2 = np.concatenate([[M.SIG_SCALAR], np.full(D.nfm, 2.5)])
    kept_cells = [c for c in range(NCELL) if D.cell_keep[c]]

    # ── closed-form gamma trajectories (all flights), for the gamma column ──
    _, th0_raw, extras = V.build_cells()
    cf = V.closed_form_trajectories(extras["counts_all"], extras["tau_raw"],
                                    extras["dates_all"], th0_raw)

    cutoffs = month_end_cutoffs(WARMUP_CUTOFF, END_CUTOFF)
    if smoke:
        cutoffs = cutoffs[:2]
    draws = 300 if smoke else M.COMMON_DRAWS
    tune  = 300 if smoke else M.COMMON_TUNE
    m2_cells_this_run = set(kept_cells[:SMOKE_M2_CELLS]) if smoke else set(kept_cells)

    done = load_done_cutoffs(CSV_PATH)
    new  = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(CSV_COLS); f.flush()

    model_files = {}
    for n in (1, 2, 3):
        path = MODEL_CSV_PATH(n)
        is_new = not os.path.exists(path)
        fh = open(path, "a", newline="")
        wr = csv.writer(fh)
        if is_new:
            wr.writerow(MODEL_CSV_COLS[n]); fh.flush()
        model_files[n] = (fh, wr)

    print(f"{len(cutoffs)} cutoffs; {len(done)} already done. draws={draws}; "
          f"Model 2 on {len(m2_cells_this_run)}/{len(kept_cells)} cells this run")
    for ci, cutoff in enumerate(cutoffs):
        cstr = cutoff.strftime("%Y-%m-%d")
        if cstr in done:
            print(f"[{cstr}] done, skipping"); continue
        rmask = np.asarray(row_dates <= cutoff)
        if rmask.sum() < NCELL * 2:
            print(f"[{cstr}] too few rows ({rmask.sum()}), skipping"); continue

        Xc  = D.X_common[rmask]
        Xf  = X_full[rmask]
        Fm  = D.FMc[rmask]           # RAW factor columns, per-flight — shared across cells
        y   = D.yc[rmask]
        off = D.offc[rmask]
        ind = D.indc[rmask]
        print(f"[{cstr}] {int(rmask.sum()//NCELL)} flights  [{ci+1}/{len(cutoffs)}]")

        # Model 1 — common factor effects, NUTS (point = posterior mean, per R)
        _, idc = M.mcmc_poisson_glm(Xc, y, off, D.prior_mean, D.prior_sd,
                                    draws, tune, seed=1000 + ci)
        post = idc.posterior["beta"].values.reshape(-1, Xc.shape[1])
        m1, m1_lo, m1_hi = marginal_mtbf_ci(Xc, off, ind, post, post.mean(axis=0))
        try:
            m1_rhat = float(az.rhat(idc)["beta"].max())
        except Exception:
            m1_rhat = float("nan")

        # Model 2 — 87 per-cell NUTS fits, RAW design compiled ONCE for this window
        # (see compile_window_model docstring: X/offset are cell-invariant per window).
        m2 = np.full(NCELL, np.nan); m2_lo = np.full(NCELL, np.nan); m2_hi = np.full(NCELL, np.nan)
        m2_rhat = np.full(NCELL, np.nan)
        any_cell_mask = ind == kept_cells[0]
        Xw2   = np.column_stack([np.ones(int(any_cell_mask.sum())), Fm[any_cell_mask]])
        offw2 = off[any_cell_mask]
        fit2, backend2 = compile_window_model(Xw2, offw2, pdim2, sub_sd2)
        for c in kept_cells:
            if c not in m2_cells_this_run:
                continue
            cmask = ind == c
            assert cmask.sum() == Xw2.shape[0], \
                f"cell {c} has {cmask.sum()} rows, window design has {Xw2.shape[0]} — flight/cell misalignment"
            y_c  = y[cmask]
            mu_c = np.concatenate([[D.mu_hyp[c]], np.zeros(D.nfm)])
            ids2 = fit2(y_c, mu_c, seed=2000 + ci * 1000 + c)
            p2 = ids2.posterior["beta"].values.reshape(-1, pdim2)
            Tc = np.exp(offw2).sum()
            mu2d = np.exp(np.clip(offw2[:, None] + Xw2 @ p2.T, -50, 50))
            d = Tc / mu2d.sum(axis=0)
            m2_lo[c], m2_hi[c] = np.percentile(d, [2.5, 97.5])
            m2[c] = Tc / np.exp(np.clip(offw2 + Xw2 @ p2.mean(0), -50, 50)).sum()
            try:
                m2_rhat[c] = float(az.rhat(ids2)["beta"].max())
            except Exception:
                pass

        # Model 3 — common + specific, Laplace/bayesglm (point = MAP, per R).
        # MTBF + CI via the MARGINAL method (== mcmc_test _save_marginal_mtbf), from the
        # sim()/Laplace draws (== Rcode.r's sim(n.sims=10000)). Marginal keeps M3 near M1;
        # the at-mean exp(-eta_bar) version extrapolated to the average profile and blew M3 up.
        fs = M.bayesglm_prior_scales(Xf, fs_base)
        b_map, H = M.bayesglm_laplace(Xf, y, off, fm_full, fs)
        m3d = M._laplace_draws(b_map, H, seed=1000 + ci).T          # (NSIM, p)
        m3, m3_lo, m3_hi = marginal_mtbf_ci(Xf, off, ind, m3d, b_map)

        for c in range(NCELL):
            if not D.cell_keep[c]:
                continue
            j, k = c // NTYP + 1, FTYP[c % NTYP]
            nfail = int(y[ind == c].sum())
            g_at_cutoff = gamma_at(cf, (j, k), cutoff)
            w.writerow([cstr, j, k, nfail, g_at_cutoff,
                        m1[c], m1_lo[c], m1_hi[c], m1_rhat,
                        m2[c], m2_lo[c], m2_hi[c], m2_rhat[c],
                        m3[c], m3_lo[c], m3_hi[c]])
            model_files[1][1].writerow([cstr, j, k, nfail, g_at_cutoff, m1[c], m1_lo[c], m1_hi[c], m1_rhat])
            model_files[2][1].writerow([cstr, j, k, nfail, g_at_cutoff, m2[c], m2_lo[c], m2_hi[c], m2_rhat[c]])
            model_files[3][1].writerow([cstr, j, k, nfail, g_at_cutoff, m3[c], m3_lo[c], m3_hi[c]])
        f.flush()
        for fh, _ in model_files.values():
            fh.flush()
        print(f"    m1_rhat={m1_rhat:.3f}  m1 med={np.nanmedian(m1):,.0f}  "
              f"m2 med={np.nanmedian(m2):,.0f}  m3 med={np.nanmedian(m3):,.0f}  "
              f"[{backend2}]")
    f.close()
    for fh, _ in model_files.values():
        fh.close()
    print(f"\ntable -> {CSV_PATH}  (+ per-model: {', '.join(MODEL_CSV_PATH(n) for n in (1,2,3))})")
    make_plots()


def make_plots():
    if not os.path.exists(CSV_PATH):
        print("no CSV to plot"); return
    ev = pd.read_csv(CSV_PATH, parse_dates=["cutoff"])
    has_m1_ci = "m1_lo95" in ev.columns
    has_m2    = "m2" in ev.columns and ev["m2"].notna().any()
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

        if (j, k) in cf:
            cdx, theta, upper, lower = cf[(j, k)]
            cm = ~pd.isna(cdx)
            order = np.argsort(cdx[cm].asi8, kind="stable")
            cd = cdx[cm][order]
            th, up, lo = theta[cm][order], upper[cm][order], lower[cm][order]
            plt.plot(cd, th, color=CF_COLOR, lw=2, label="Closed-form gamma")
            plt.plot(cd, up, color=CF_COLOR, lw=1, linestyle="--", label="Closed-form 95% CI")
            plt.plot(cd, lo, color=CF_COLOR, lw=1, linestyle="--")

        plt.plot(g.cutoff, g.m1, color=M1_COLOR, marker="o", lw=2, ms=5,
                 label="Model 1 (common)")
        if has_m1_ci:
            plt.plot(g.cutoff, g.m1_lo95, color=M1_COLOR, lw=1, linestyle=":",
                     label="Model 1 95% CI")
            plt.plot(g.cutoff, g.m1_hi95, color=M1_COLOR, lw=1, linestyle=":")

        gm2 = g[g.m2.notna()] if has_m2 else g.iloc[0:0]
        if len(gm2):
            plt.plot(gm2.cutoff, gm2.m2, color=M2_COLOR, marker="^", lw=2, ms=5,
                     label="Model 2 (subsystem-specific)")
            plt.plot(gm2.cutoff, gm2.m2_lo95, color=M2_COLOR, lw=1, linestyle=":",
                     label="Model 2 95% CI")
            plt.plot(gm2.cutoff, gm2.m2_hi95, color=M2_COLOR, lw=1, linestyle=":")

        plt.plot(g.cutoff, g.m3, color=M3_COLOR, marker="s", lw=2, ms=5,
                 label="Model 3 (common + specific)")
        if "m3_lo95" in ev.columns:
            plt.plot(g.cutoff, g.m3_lo95, color=M3_COLOR, lw=1, linestyle=":",
                     label="Model 3 95% CI")
            plt.plot(g.cutoff, g.m3_hi95, color=M3_COLOR, lw=1, linestyle=":")

        # Axis hard-capped at Y_CAP x the max of the sane models (gamma + contractor +
        # Model 1 + Model 2); Model 3 and its Laplace CI just clip off the top instead
        # of smushing the plot. n_fail_cum>0 drops the prior-dominated early windows.
        gd = g[g.n_fail_cum > 0]
        cap = [th0]
        if (j, k) in cf:
            cap += [v for v in th if np.isfinite(v) and v > 0]
        if "m1" in ev.columns:
            cap += [v for v in gd.m1 if np.isfinite(v) and v > 0]
        if has_m2:
            cap += [v for v in gd.m2 if np.isfinite(v) and v > 0]
        yhi = Y_CAP * max(cap)
        lows = [v for v in ([th0]
                + (list(th) + list(lo) if (j, k) in cf else [])
                + (list(gd.m1) if "m1" in ev.columns else [])
                + (list(gd.m2) if has_m2 else [])
                + (list(gd.m3) if "m3" in ev.columns else []))
                if np.isfinite(v) and 0 < v <= yhi]
        plt.ylim((min(lows) if lows else 0.0) * 0.85, yhi)

        plt.title(f"Subsystem {j} Type {k} — closed form vs Model 1 vs Model 2 vs Model 3")
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
