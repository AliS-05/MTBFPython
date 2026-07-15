"""
Python port of Rcode.r — Bayesian GLM factor analysis for MTBF failure counts.

Stages, matching the R script:
  1. Frequentist Poisson GLM                    == R glm()
  2. Common-factor Bayesian GLM via NUTS        == rstanarm::stan_glm (iter=2000*2, adapt_delta=0.99)
  3. 87 per-(subsystem x type) Bayesian GLMs    == stan_glm loop (iter=2000*2)
  4. Combined interaction model, Laplace + sim  == arm::bayesglm(prior.df=Inf) + sim()

Model selection (Dr. Han's numbering): Model 1 = stage 2 (common factor effects),
Model 2 = stage 3 (subsystem x ftyp-specific only), Model 3 = stage 4 (common + specific).
    python mcmc.py            # all three models + frequentist preamble
    python mcmc.py --model 1  # just Model 1 (common)
    python mcmc.py --model 3  # just Model 3 (combined)

Fidelity note on stage 4: arm::bayesglm defaults to scaled=TRUE, which divides each
coefficient's prior scale by that column's spread (max-min if the column is binary,
2*sd otherwise). Sparse interaction columns (nonzero on 1/87 of rows, small poly
contrast values) get their prior sd inflated from 2.5 to ~23, i.e. 95% intervals
~±46 — which is why R's Factor3 interaction plots use xlim ±45. This port replicates
that rescaling in stage 4 only; rstanarm's normal() prior does not autoscale, so
stages 2-3 use the literal (mu, sig) / (0, 2.5) priors.

Factor3 encoding (R model.matrix rule): Factor3 is the only factor-class variable
in the formulas, and every model drops the intercept (-1), so R codes its MAIN
effect with full indicators — one column per level, named Factor31..Factor34 — in
all three model stages. Inside interaction terms (s*t*:Factor3) it reverts to the
ordered factor's contr.poly contrasts (.L/.Q/.C). This asymmetry is also why R's
prior lengths are internally consistent: nfac+3 (=10) factor-main columns
(2 + 4 dummies + 4) vs nfac+2 (=9) interaction columns per cell (2 + 3 contrasts + 4).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from types import SimpleNamespace

# ───────────────────────── CONFIG ─────────────────────────
DATASET_FILEPATH = "/mnt/c/Users/sefra/Downloads/dataset.xlsx"
MAINT_SHEET, MTBF_SHEET = "Failure Data", "Initial MTBF"

NSUB, FTYP        = 29, [1, 2, 6]
NTYP, NFAC        = len(FTYP), 7
NUM_PAIRS         = 13
TH0MX             = 1e8
P_IN, P_OUT       = 0.50, 0.95
SIG_SCALAR        = np.pi / np.sqrt(6)          # Gumbel sd = 1.28255
NSIM              = 10000
MIN_PRIOR_SCALE   = 1e-12                       # arm::bayesglm min.prior.scale

CORES         = 1
NUTS_SAMPLER  = "nutpie"                        # "nutpie", "numpyro" or "pymc"
COMMON_DRAWS, COMMON_TUNE = 2000, 2000          # R: iter=2000*2, half warmup
SUB_DRAWS,    SUB_TUNE    = 2000, 2000          # R: iter=2000*2 (line 274 — same as common)
TARGET_ACCEPT = 0.99                            # R: adapt_delta=0.99

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_mcmc")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ───────────────────────── HELPERS ─────────────────────────
def save(name):
  plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, name), dpi=100); plt.close()

def intervals_data(samples_df, p_in=P_IN, p_out=P_OUT, point_est="mean"):
  """Replaces bayesplot::mcmc_intervals_data. Columns: par, ll, l, m, h, hh."""
  rows = []
  for col in samples_df.columns:
      s = samples_df[col].values
      m = s.mean() if point_est == "mean" else np.median(s)
      ll, l, h, hh = np.quantile(s, [(1-p_out)/2, (1-p_in)/2, (1+p_in)/2, (1+p_out)/2])
      rows.append({"par": col, "ll": ll, "l": l, "m": m, "h": h, "hh": hh})
  return pd.DataFrame(rows)

def plot_bci(idat, xlim=None, title="", ax=None):
  """Horizontal credible-interval plot. Replaces R's plotBci."""
  if ax is None:
      _, ax = plt.subplots(figsize=(6, max(3, len(idat) * 0.3)))
  if len(idat) == 0:
      ax.set_title(title + " (empty)"); return ax
  if xlim is None:
      rng = idat["hh"].max() - idat["ll"].min()
      xlim = [idat["ll"].min() - 0.05*rng, idat["hh"].max() + 0.05*rng]
  ax.axvline(0, color="gray", linewidth=1)
  for i, (_, row) in enumerate(idat.iloc[::-1].iterrows()):
      ax.plot([row["ll"], row["hh"]], [i, i], color="#3B528B", linewidth=1)
      ax.plot([row["l"],  row["h"]],  [i, i], color="#21908C", linewidth=3)
      ax.plot(row["m"], i, "o", markersize=5,
              markerfacecolor="#440154", markeredgecolor="#21908C")
  ax.set_yticks(range(len(idat)))
  ax.set_yticklabels(list(reversed(idat["par"].tolist())), fontsize=6)
  ax.set_xlim(xlim); ax.set_title(title, fontweight="bold")
  return ax

def contr_poly(n):
  """Orthonormal polynomial contrasts, matching R's contr.poly() up to sign."""
  x = np.arange(1, n + 1, dtype=float); x -= x.mean()
  Q, _ = np.linalg.qr(np.vander(x, n, increasing=True))
  Z = Q[:, 1:].copy()
  for c in range(Z.shape[1]):                   # sign: last row positive (R convention)
      if Z[-1, c] < 0: Z[:, c] = -Z[:, c]
  return Z

def prior_intervals(mean, sd, names, seed=0):
  # coefficient prior is exactly Normal(mean, sd) (R's prior_PD=TRUE marginals)
  rng = np.random.default_rng(seed)
  return intervals_data(pd.DataFrame(
      rng.normal(np.asarray(mean), np.asarray(sd), size=(NSIM, len(mean))), columns=names))

def _pm_sample(draws, tune, seed, progressbar=True):
  """pm.sample with the configured NUTS backend, falling back to default pymc."""
  kw = dict(draws=draws, tune=tune, chains=4, cores=CORES,
            target_accept=TARGET_ACCEPT, random_seed=seed, progressbar=progressbar)
  try:
      return pm.sample(**kw, nuts_sampler=NUTS_SAMPLER)
  except Exception as e:
      print(f"  [{NUTS_SAMPLER} sampler failed: {e}; falling back to default pymc]")
      return pm.sample(**kw)

def mcmc_poisson_glm(X, y, offset, prior_mean, prior_sd, draws, tune, seed, ppc=False):
  """Full NUTS Poisson GLM with Normal priors + log-offset == R stan_glm(...)."""
  with pm.Model() as model:
      beta = pm.Normal("beta", mu=np.asarray(prior_mean), sigma=np.asarray(prior_sd),
                       shape=X.shape[1])
      pm.Poisson("obs", mu=pm.math.exp(pm.math.dot(np.asarray(X, float), beta) + offset),
                 observed=np.asarray(y, float))
      idata = _pm_sample(draws, tune, seed)
      if ppc:
          idata = pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                                 random_seed=seed)
  return model, idata

def bayesglm_prior_scales(X, base_scale):
  """arm::bayesglm(scaled=TRUE) prior-scale adjustment (non-gaussian family):
  per column, divide prior scale by (max-min) if 2 distinct values, 2*sd if >2."""
  scales = np.asarray(base_scale, float).copy()
  for j in range(X.shape[1]):
      u = np.unique(X[:, j])
      if len(u) == 2:
          d = u[-1] - u[0]
      elif len(u) > 2:
          d = 2.0 * np.std(X[:, j], ddof=1)     # R's sd() uses n-1
      else:
          d = 1.0
      if d > 0:
          scales[j] = max(scales[j] / d, MIN_PRIOR_SCALE)
  return scales

def bayesglm_laplace(X, y, offset, prior_mean, prior_sd, maxit=100, tol=1e-8):
  """Penalized Newton to the MAP == arm::bayesglm(prior.df=Inf).
  Returns (beta_MAP, H); posterior ≈ N(beta_MAP, H^-1) as in arm::sim()."""
  X = np.asarray(X, float); y = np.asarray(y, float); offset = np.asarray(offset, float)
  m0 = np.asarray(prior_mean, float); prec0 = 1.0 / np.asarray(prior_sd, float) ** 2
  beta = m0.copy()
  def pen_ll(b):
      eta = np.clip(X @ b + offset, -50, 50)
      return np.sum(y*eta - np.exp(eta)) - 0.5*np.sum(prec0*(b-m0)**2)
  ll = pen_ll(beta)
  for it in range(maxit):
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
          ll = new; break
      ll = new
  eta = np.clip(X @ beta + offset, -50, 50); mu = np.exp(eta)
  grad = X.T @ (y - mu) - prec0*(beta-m0)
  H = X.T @ (X * mu[:, None]) + np.diag(prec0)
  print(f"  Newton: {it+1} iterations, |grad| = {np.linalg.norm(grad):.3e}")
  return beta, 0.5*(H + H.T)

def sim_intervals(beta, H, names, seed=1):
  """arm::sim(): draws ~ MVN(MAP, H^-1), via Cholesky solve (no explicit inverse)."""
  rng = np.random.default_rng(seed)
  Z = rng.standard_normal((len(beta), NSIM))
  try:
      L = np.linalg.cholesky(H)
      draws = beta[:, None] + np.linalg.solve(L.T, Z)
  except np.linalg.LinAlgError:
      w, U = np.linalg.eigh(H)
      w = np.maximum(w, 1e-12)
      draws = beta[:, None] + U @ (Z / np.sqrt(w)[:, None])
  return intervals_data(pd.DataFrame(draws.T, columns=names))

# ─────────────────── DATA LOADING (positional, per R) ───────────────────
def load_maintenance():
  raw = pd.read_excel(DATASET_FILEPATH, sheet_name=MAINT_SHEET, header=0)
  ncol = raw.shape[1]; print(f"raw maintenance sheet: {raw.shape}  (R expects 300 x 49)")
  if ncol < 49:
      raise ValueError(f"maintenance sheet has {ncol} columns, expected >= 49 "
                       "(Sub/Type pairs at 1-based cols 4..41, Factors at 43..49)")
  def num(c): return pd.to_numeric(raw.iloc[:, c], errors="coerce").values
  tau, system = num(1), num(2)
  sub_pos = [3 + 3*p for p in range(NUM_PAIRS)]
  subs  = np.column_stack([num(c)   for c in sub_pos])
  types = np.column_stack([num(c+1) for c in sub_pos])
  factors = np.column_stack([num(42 + f) for f in range(NFAC)])
  return tau, system, subs, types, factors

def load_theta0():
  raw = pd.read_excel(DATASET_FILEPATH, sheet_name=MTBF_SHEET, header=0)
  t = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")   # R: theta0[-1,-1]
  th = np.full((NSUB, NTYP), np.nan); v = t.values
  r, c = min(NSUB, v.shape[0]), min(NTYP, v.shape[1]); th[:r, :c] = v[:r, :c]
  print(f"theta0 (initial MTBF): {th.shape}  (R expects 29 x 3)")
  return th


# ──────────────────── PER-CELL MTBF EXTRACTION ────────────────────
def _laplace_draws(beta, H, seed=1):
    """Raw MVN(MAP, H^-1) draws (same logic as sim_intervals), shape (p, NSIM)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((len(beta), NSIM))
    try:
        L = np.linalg.cholesky(H)
        return beta[:, None] + np.linalg.solve(L.T, Z)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(H)
        w = np.maximum(w, 1e-12)
        return beta[:, None] + U @ (Z / np.sqrt(w)[:, None])


def _save_cell_mtbf(mtbf_draws, mtbf_point, ind_names, fname, keep=None):
    """Per-cell MTBF summary CSV — same columns for every model so Model 1/2/3 overlay
    directly. keep: optional bool mask; cells with keep[c]=False are dropped (NA-contractor
    cells, which the gamma analysis excludes). mtbf_draws: (n_cells, n_draws)."""
    rows = []
    for c, key in enumerate(ind_names):
        if keep is not None and not keep[c]:
            continue
        d = mtbf_draws[c]
        lo95, lo50, med, hi50, hi95 = np.percentile(d, [2.5, 25, 50, 75, 97.5])
        rows.append({"cell": key, "subsystem": c // NTYP + 1, "ftype": FTYP[c % NTYP],
                     "mtbf_map": float(mtbf_point[c]), "mtbf_mean": float(d.mean()),
                     "mtbf_median": med, "mtbf_lo95": lo95, "mtbf_lo50": lo50,
                     "mtbf_hi50": hi50, "mtbf_hi95": hi95})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PLOTS_DIR, fname), index=False)
    print(f"  per-cell MTBF -> {os.path.join(PLOTS_DIR, fname)}  "
          f"median range [{df.mtbf_median.min():.0f}, {df.mtbf_median.max():.0f}] "
          f"(sane ~ 1e2-1e4)")
    return df


def _save_marginal_mtbf(X, off, indc, beta_draws, beta_point, ind_names, fname, keep=None):
    """Per-cell exposure-weighted MTBF at each cell's ACTUAL flights (robust — no
    extrapolation to an average profile that no flight has):
        MTBF_c = sum(tau_i) / sum(mu_i),   mu_i = exp(offset_i + X_i . beta)
    over the flights of cell c. beta_draws: (n_draws, p); beta_point: (p,).
    keep: optional bool mask; skipped cells (NA-contractor) are filled NaN and dropped."""
    off = np.asarray(off, float); nd = beta_draws.shape[0]
    md, mp = [], []
    for c in range(len(ind_names)):
        if keep is not None and not keep[c]:
            md.append(np.full(nd, np.nan)); mp.append(np.nan); continue
        m = (indc == c)
        Xc = X[m]; oc = off[m]; T = np.exp(oc).sum()
        mu = np.exp(np.clip(oc[:, None] + Xc @ beta_draws.T, -50, 50))   # (n_c, n_draws)
        md.append(T / mu.sum(axis=0))
        mp.append(T / np.exp(np.clip(oc + Xc @ beta_point, -50, 50)).sum())
    return _save_cell_mtbf(np.vstack(md), np.array(mp), ind_names, fname, keep)


# ──────────────────────── SHARED DESIGN ────────────────────────
def build_design():
    """Load data + build the design matrices/priors shared by all three models."""
    tau_all, system_all, subs, types, factors = load_maintenance()
    nt = subs.shape[0]

    counts = np.zeros((nt, NSUB * NTYP))
    for j in range(1, NSUB + 1):
        for ti, k in enumerate(FTYP):
            counts[:, (j-1)*NTYP + ti] = ((subs == j) & (types == k)).sum(axis=1)

    nf      = counts.reshape(-1)
    ind_col = np.tile(np.arange(NSUB*NTYP), nt)
    tau_d   = np.repeat(tau_all, NSUB*NTYP)
    fac_d   = np.repeat(factors, NSUB*NTYP, axis=0)

    ind_names = [f"s{j}t{k}" for j in range(1, NSUB+1) for k in FTYP]
    IND = np.eye(NSUB*NTYP)[ind_col]

    f3 = fac_d[:, 2]; levels = np.unique(f3[np.isfinite(f3)])
    nlev = len(levels)
    if nlev < 2:
        raise ValueError(f"Factor3 has {nlev} observed level(s); need >= 2 for contrasts")
    lev_idx = np.full(len(f3), -1, dtype=int)
    for i, v in enumerate(levels):
        lev_idx[f3 == v] = i
    ok = lev_idx >= 0

    f3_dummy = np.zeros((len(f3), nlev))
    f3_dummy[np.arange(len(f3))[ok], lev_idx[ok]] = 1.0
    dummy_names = [f"Factor3{int(v) if v == int(v) else v}" for v in levels]

    poly_suffix = ([".L", ".Q", ".C"] + [f"^{d}" for d in range(4, nlev)])[:nlev-1]
    Z = contr_poly(nlev)
    f3_poly = np.zeros((len(f3), nlev-1))
    f3_poly[ok] = Z[lev_idx[ok]]

    FM = np.column_stack([fac_d[:, [0, 1]], f3_dummy, fac_d[:, 3:7]])
    fac_names = (["Factor1", "Factor2"] + dummy_names
                 + ["Factor4", "Factor5", "Factor6", "Factor7"])
    FMI = np.column_stack([fac_d[:, [0, 1]], f3_poly, fac_d[:, 3:7]])
    int_fac_names = (["Factor1", "Factor2"] + [f"Factor3{s}" for s in poly_suffix]
                     + ["Factor4", "Factor5", "Factor6", "Factor7"])
    nfm, nfm_int = FM.shape[1], FMI.shape[1]
    print(f"Factor3: {nlev} levels -> main-effect dummies {dummy_names}, "
          f"interaction contrasts {['Factor3'+s for s in poly_suffix]}")
    if nfm != NFAC + 3 or nfm_int != NFAC + 2:
        print(f"  NOTE: R's prior vectors hardcode nfac+3={NFAC+3} factor mains and "
              f"nfac+2={NFAC+2} interaction columns per cell; actual counts here are "
              f"{nfm}/{nfm_int} (Factor3 level count differs from 4).")

    flight_ok = np.isfinite(tau_all) & (tau_all > 0) & np.all(np.isfinite(factors), axis=1)
    V = np.repeat(flight_ok, NSUB*NTYP)
    if (~flight_ok).sum():
        print(f"na.omit: dropping {(~flight_ok).sum()} of {nt} flights "
              f"(non-finite tau or factor values)")

    INDc, FMc, FMIc = IND[V], FM[V], FMI[V]
    yc, offc  = nf[V], np.log(tau_d[V])
    indc      = ind_col[V]
    nt_ok     = int(flight_ok.sum())

    theta0 = load_theta0()
    th0 = theta0.reshape(-1).copy(); th0[~np.isfinite(th0)] = TH0MX
    mu_hyp = -np.euler_gamma - np.log(th0)
    cell_keep = np.isfinite(theta0.reshape(-1))   # finite-contractor cells only (drop NA cells, as the gamma analysis does)

    common_names = ind_names + fac_names
    X_common = np.column_stack([INDc, FMc])
    print(f"common design matrix: {X_common.shape}  (R: 26100 x {NSUB*NTYP + nfm}, pre-na.omit)")
    prior_mean = np.concatenate([mu_hyp, np.zeros(nfm)])
    prior_sd   = np.concatenate([np.full(NSUB*NTYP, SIG_SCALAR), np.full(nfm, 2.5)])

    plt.hist(nf, bins=20); plt.xlabel("Observed Counts")
    plt.title("Histogram of Observed Counts"); save("hist_nf.png")

    return SimpleNamespace(
        yc=yc, offc=offc, indc=indc, nt_ok=nt_ok,
        INDc=INDc, FMc=FMc, FMIc=FMIc,
        ind_names=ind_names, fac_names=fac_names, int_fac_names=int_fac_names,
        nfm=nfm, nfm_int=nfm_int, mu_hyp=mu_hyp, cell_keep=cell_keep,
        common_names=common_names, X_common=X_common,
        prior_mean=prior_mean, prior_sd=prior_sd,
    )


def run_frequentist(D):
    """Stage 1 — frequentist Poisson GLM (== R glm), a diagnostic preamble."""
    print("Frequentist GLM...")
    try:
        glm = sm.GLM(D.yc, pd.DataFrame(D.X_common, columns=D.common_names),
                     family=sm.families.Poisson(), offset=D.offc).fit(maxiter=100)
        print(glm.summary()); print("\nexp(-coef):\n", np.exp(-glm.params))
    except Exception as e:
        print(f"[frequentist GLM skipped]: {e}")


def run_model_1(D):
    """Model 1 — common-factor Bayesian GLM (NUTS). == R stan_glm (stage 2)."""
    print("Model 1 (common factor effects) — NUTS...")
    idat0 = prior_intervals(D.prior_mean, D.prior_sd, D.common_names)
    _, idc = mcmc_poisson_glm(D.X_common, D.yc, D.offc, D.prior_mean, D.prior_sd,
                              COMMON_DRAWS, COMMON_TUNE, seed=1, ppc=True)
    post = idc.posterior["beta"].values.reshape(-1, D.X_common.shape[1])
    idat1 = intervals_data(pd.DataFrame(post, columns=D.common_names))

    # per-cell MTBF at each cell's ACTUAL flights (robust marginal; no extrapolation)
    _save_marginal_mtbf(D.X_common, D.offc, D.indc, post, post.mean(axis=0),
                        D.ind_names, "mtbf_model1.csv", keep=D.cell_keep)

    fig, ax = plt.subplots(figsize=(6, 16)); plot_bci(idat0, title="Prior", ax=ax); save("prior.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 16))
    plot_bci(idat0, title="Prior", ax=axes[0]); plot_bci(idat1, title="Posterior", ax=axes[1])
    save("prior_posterior.png")

    try:
        summ = az.summary(idc, var_names=["beta"])
        summ.index = D.common_names
        print("Max R-hat:", summ["r_hat"].max(), " Min ESS_bulk:", summ["ess_bulk"].min())
        summ.to_csv(os.path.join(PLOTS_DIR, "convergence_summary.csv"))
    except Exception as e:
        print(f"[summary skipped]: {e}")

    pv = idc.posterior["beta"].values
    fac_idx = list(range(NSUB*NTYP, D.X_common.shape[1]))
    fig, axes = plt.subplots(len(fac_idx), 1, figsize=(8, 1.4*len(fac_idx)), squeeze=False)
    for r, pi in enumerate(fac_idx):
        for ch in range(pv.shape[0]):
            axes[r, 0].plot(pv[ch, :, pi], lw=0.4)
        axes[r, 0].set_ylabel(D.common_names[pi], fontsize=6)
    save("trace_factors.png")

    try:
        rep = idc.posterior_predictive["obs"].values.reshape(-1, len(D.yc))
        plt.hist(D.yc, bins=40, density=True, alpha=0.5, label="observed")
        for i in np.random.default_rng(0).choice(rep.shape[0], 5, replace=False):
            plt.hist(rep[i], bins=40, density=True, histtype="step")
        plt.xlabel("Counts"); plt.legend(); plt.title("Posterior predictive check"); save("ppc.png")
    except Exception as e:
        print(f"[ppc skipped]: {e}")


def run_model_2(D):
    """Model 2 — 87 per-(subsystem x ftyp) Bayesian GLMs, specific factor effects
    only (each cell fit independently). == R stan_glm loop (stage 3). Writes beta0_draws.npz."""
    print(f"Model 2 (subsystem-specific factor effects) — {NSUB*NTYP} NUTS fits...")
    term_names = ["Intercept"] + D.fac_names
    pdim   = 1 + D.nfm
    sub_sd = np.concatenate([[SIG_SCALAR], np.full(D.nfm, 2.5)])

    def sub_dataset(c):
        mask = D.indc == c
        Fm = D.FMc[mask]
        # RAW factors, matching Rcode.r stage-3 (nf ~ -1 + s + Factor1..7, no centering).
        # The earlier `Fm - Fm.mean(0)` was carried over from the phase-3 validate work;
        # centering breaks the raw Factor3-dummies=intercept collinearity and shifts every
        # factor slope. Raw is rank-deficient (prior regularizes it, exactly as R does).
        Xs = np.column_stack([np.ones(mask.sum()), Fm])
        mu_c = np.concatenate([[D.mu_hyp[c]], np.zeros(D.nfm)])
        return Xs, D.yc[mask].astype(float), D.offc[mask], mu_c

    fit_probe = None
    try:
        import nutpie
        with pm.Model() as sub_model:
            Xd   = pm.Data("Xd",   np.zeros((D.nt_ok, pdim)))
            yd   = pm.Data("yd",   np.zeros(D.nt_ok))
            offd = pm.Data("offd", np.zeros(D.nt_ok))
            mud  = pm.Data("mud",  np.zeros(pdim))
            beta = pm.Normal("beta", mu=mud, sigma=sub_sd, shape=pdim)
            pm.Poisson("obs", mu=pm.math.exp(pm.math.dot(Xd, beta) + offd), observed=yd)
        compiled = nutpie.compile_pymc_model(sub_model)

        def fit_sub(c):
            Xs, y, off, mu_c = sub_dataset(c)
            cm = compiled.with_data(Xd=Xs, yd=y, offd=off, mud=mu_c)
            return nutpie.sample(cm, draws=SUB_DRAWS, tune=SUB_TUNE, chains=4,
                                 target_accept=TARGET_ACCEPT, seed=1000 + c,
                                 progress_bar=False)

        fit_probe = fit_sub(0)
        print("  [nutpie direct: compiled once, reusing across fits]")
    except Exception as e:
        print(f"  [nutpie direct path unavailable ({e}); using pm.sample per fit]")
        def fit_sub(c):
            Xs, y, off, mu_c = sub_dataset(c)
            with pm.Model():
                b = pm.Normal("beta", mu=mu_c, sigma=sub_sd, shape=pdim)
                pm.Poisson("obs", mu=pm.math.exp(pm.math.dot(Xs, b) + off), observed=y)
                return _pm_sample(SUB_DRAWS, SUB_TUNE, seed=1000 + c, progressbar=False)
        fit_probe = None

    ipri = [[] for _ in term_names]; ipos = [[] for _ in term_names]
    conv = []
    beta0_draws = {}
    m2_draws, m2_point = [], []
    for c, key in enumerate(D.ind_names):
        ids = fit_probe if (c == 0 and fit_probe is not None) else fit_sub(c)
        Xs_c, _, off_c, mu_c = sub_dataset(c)
        colnames = [key] + D.fac_names
        dpri = prior_intervals(mu_c, sub_sd, colnames, seed=c)
        p = ids.posterior["beta"].values.reshape(-1, pdim)
        beta0_draws[key] = p[:, 0].copy()
        Tc = np.exp(off_c).sum()
        mu2 = np.exp(np.clip(off_c[:, None] + Xs_c @ p.T, -50, 50))
        m2_draws.append(Tc / mu2.sum(axis=0))
        m2_point.append(Tc / np.exp(np.clip(off_c + Xs_c @ p.mean(0), -50, 50)).sum())
        dpos = intervals_data(pd.DataFrame(p, columns=colnames))
        for t in range(len(term_names)):
            ipri[t].append(dpri.iloc[t]); ipos[t].append(dpos.iloc[t])
        row = {"cell": key}
        try:
            row["max_rhat"] = float(az.rhat(ids)["beta"].max())
            row["min_ess"]  = float(az.ess(ids)["beta"].min())
            row["divergences"] = int(np.asarray(ids.sample_stats["diverging"]).sum())
        except Exception:
            pass
        conv.append(row)
        print(f"  {key} ({c+1}/{NSUB*NTYP})" +
              (f"  rhat={row['max_rhat']:.3f}" if "max_rhat" in row else ""))
    pd.DataFrame(conv).to_csv(os.path.join(PLOTS_DIR, "persub_convergence.csv"), index=False)
    bad = [r for r in conv if r.get("max_rhat", 1.0) > 1.01]
    if bad:
        print(f"  WARNING: {len(bad)} cells with R-hat > 1.01 — see persub_convergence.csv")
    np.savez_compressed(os.path.join(PLOTS_DIR, "beta0_draws.npz"), **beta0_draws)
    print(f"  beta0 posterior draws -> {os.path.join(PLOTS_DIR, 'beta0_draws.npz')}")

    # per-cell MTBF at each cell's ACTUAL flights (robust marginal), collected in the loop
    _save_cell_mtbf(np.vstack(m2_draws), np.array(m2_point), D.ind_names, "mtbf_model2.csv", D.cell_keep)

    for t, tname in enumerate(term_names):
        dpri = pd.DataFrame(ipri[t]).reset_index(drop=True); dpri["par"] = D.ind_names
        dpos = pd.DataFrame(ipos[t]).reset_index(drop=True); dpos["par"] = D.ind_names
        fig, axes = plt.subplots(1, 2, figsize=(8, 10))
        plot_bci(dpri, xlim=[-25, 8], title="Prior", ax=axes[0])
        plot_bci(dpos, xlim=[-25, 8], title="Posterior", ax=axes[1])
        plt.suptitle(tname, fontweight="bold"); save(f"{tname}.png")


def run_model_3(D):
    """Model 3 — combined: common mains + all (subsystem x ftyp):factor interactions.
    == arm::bayesglm(prior.df=Inf) + sim() (stage 4), incl. scaled=TRUE rescaling."""
    print("Model 3 (common mains + specific interactions) — bayesglm/Laplace (scaled=TRUE)...")
    Nc = D.INDc.shape[0]
    INT = np.zeros((Nc, (NSUB*NTYP)*D.nfm_int)); rows = np.arange(Nc)
    for m in range(D.nfm_int):
        INT[rows, D.indc*D.nfm_int + m] = D.FMIc[:, m]
    int_names = [f"{D.ind_names[a]}:{D.int_fac_names[m]}"
                 for a in range(NSUB*NTYP) for m in range(D.nfm_int)]
    X_full = np.column_stack([D.INDc, D.FMc, INT])
    full_names = D.ind_names + D.fac_names + int_names
    nmain = NSUB*NTYP + D.nfm

    idat0 = prior_intervals(D.prior_mean, D.prior_sd, D.common_names)  # main-effect prior for the plot

    fm_full = np.concatenate([D.mu_hyp, np.zeros(X_full.shape[1] - NSUB*NTYP)])
    fs_base = np.concatenate([np.full(NSUB*NTYP, SIG_SCALAR),
                              np.full(X_full.shape[1] - NSUB*NTYP, 2.5)])
    fs_full = bayesglm_prior_scales(X_full, fs_base)
    for f in ["Factor1", "Factor3.L", "Factor4"]:
        ii = [i for i, n_ in enumerate(full_names) if n_.endswith(":" + f)]

    b_f, H_f = bayesglm_laplace(X_full, D.yc, D.offc, fm_full, fs_full)
    idat_c = sim_intervals(b_f, H_f, full_names)

    # per-cell MTBF at each cell's ACTUAL flights (robust marginal). The interaction
    # coefficients are huge, but their NET effect is constrained at real flights, so this
    # stays in-data and can't blow up the way extrapolating to an average profile did.
    _save_marginal_mtbf(X_full, D.offc, D.indc, _laplace_draws(b_f, H_f, seed=1).T, b_f,
                        D.ind_names, "mtbf_model3.csv", keep=D.cell_keep)

    fig, axes = plt.subplots(1, 2, figsize=(8, 10))
    plot_bci(idat0, xlim=[-25, 8], title="Prior of Main Effects", ax=axes[0])
    plot_bci(idat_c.iloc[:nmain], xlim=[-25, 8], title="Posterior of Main Effects", ax=axes[1])
    save("MainEffect.png")

    def group_plot(specs, fname, figsize):
        fig, axes = plt.subplots(1, len(specs), figsize=figsize); axes = np.atleast_1d(axes)
        for ax, (sub, title, xlim) in zip(axes, specs):
            plot_bci(idat_c[idat_c["par"].str.contains(sub, regex=False)],
                     xlim=xlim, title=title, ax=ax)
        save(fname)

    group_plot([(":Factor1", "Factor 1 Interactions", [-25, 8]),
                (":Factor2", "Factor 2 Interactions", [-25, 8])], "Factor1&2.png", (8, 10))
    group_plot([(":Factor3.L", "Factor 3.L", [-45, 45]), (":Factor3.Q", "Factor 3.Q", [-45, 45]),
                (":Factor3.C", "Factor 3.C", [-45, 45])], "Factor3.png", (12, 10))
    group_plot([(":Factor4", "Factor 4 Interactions", [-25, 8]),
                (":Factor5", "Factor 5 Interactions", [-25, 8])], "Factor4&5.png", (8, 10))
    group_plot([(":Factor6", "Factor 6 Interactions", [-25, 8]),
                (":Factor7", "Factor 7 Interactions", [-25, 8])], "Factor6&7.png", (8, 10))


def main():
    ap = argparse.ArgumentParser(
        description="Bayesian GLM MTBF factor analysis (Rcode.r port).")
    ap.add_argument("--model", type=int, choices=[1, 2, 3], default=None,
                    help="run only this model: 1=common factor effects, "
                         "2=(subsystem x ftyp)-specific only, 3=common + specific. "
                         "Omit to run all three (plus the frequentist preamble).")
    args = ap.parse_args()

    D = build_design()
    m = args.model
    if m is None:
        run_frequentist(D)
    if m in (None, 1):
        run_model_1(D)
    if m in (None, 2):
        run_model_2(D)
    if m in (None, 3):
        run_model_3(D)
    print(f"Done. Plots in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
