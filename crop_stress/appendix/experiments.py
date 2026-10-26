"""
experiments.py — the six experiments of experiment4.

Every experiment reports on the `test` split, which nothing is ever chosen on.
Where a hyperparameter has to be picked (threshold percentile, k), it is picked
on `cal`. This is the difference that makes the numbers here defensible where a
sweep-and-report-the-best-cell would not be.

  E1  Threshold calibrated on clean data, percentile chosen on `cal`
  E2  Range-based (event-level) precision and recall
  E3  Same-date peer-deviation feature
  E4  LOF neighbourhood size sweep
  E5  Significance test behind adaptive P85 vs global P90
  E6  Extreme-value (POT/GPD) thresholding
"""

import numpy as np
from scipy import stats

import common as C


# ══════════════════════════════════════════════════════════════════════
#  Shared setup — fit once, reuse everywhere
# ══════════════════════════════════════════════════════════════════════
class Study:
    """Holds the splits, the fitted detector and the per-seed injected draws."""

    def __init__(self, n_neighbors=None, use_peer_feature=False):
        self.base = C.load_base()
        self.tiles = self.base["tiles"]
        self.dates = self.base["dates"]
        self.splits = C.make_splits(self.tiles)
        self.use_peer = use_peer_feature

        self.mask = {k: np.isin(self.tiles, v) for k, v in self.splits.items()}
        self.tiles_of = {k: self.tiles[m] for k, m in self.mask.items()}
        self.slices = {k: C.tile_slices(t) for k, t in self.tiles_of.items()}

        X = self.base["X"]
        if self.use_peer:
            X = C.with_peer_feature(X, self.base["df"]["p50"].to_numpy(float),
                                    self.dates)
        self.X_clean = X

        self.scaler, self.lof = C.fit_detector(
            self.X_clean[self.mask["fit"]], n_neighbors=n_neighbors)

        # Clean scores for cal/test — the basis of every honest threshold.
        self.clean_scores = {
            k: C.score(self.scaler, self.lof, self.X_clean[self.mask[k]])
            for k in ("cal", "test")
        }

        self._draws = {}

    def draw(self, seed):
        """Injected scores and ground truth for one seed, per split."""
        if seed in self._draws:
            return self._draws[seed]
        Xm, gt, p50 = C.injected(self.base["df"], seed, self.tiles)
        if self.use_peer:
            Xm = C.with_peer_feature(Xm, p50, self.dates)
        out = {}
        for k in ("cal", "test"):
            m = self.mask[k]
            out[k] = {
                "scores": C.score(self.scaler, self.lof, Xm[m]),
                "gt": gt[m],
            }
        self._draws[seed] = out
        return out

    def frozen(self, split, pctl):
        return C.clean_thresholds(self.clean_scores[split], self.slices[split],
                                  pctl)


# ══════════════════════════════════════════════════════════════════════
#  E1 — clean-calibrated threshold, percentile chosen on `cal`
# ══════════════════════════════════════════════════════════════════════
# Percentiles of 25 clean scores. Above ~P96 the empirical percentile is
# interpolating into the top one or two observations, which is exactly the
# saturation E6 is meant to escape — the grid goes there anyway so the
# saturation is visible rather than assumed.
PCTL_GRID = np.round(np.r_[np.arange(80.0, 96.0, 1.0),
                           np.arange(96.0, 99.91, 0.25)], 2)


def e1_threshold(study, seeds=C.SEEDS):
    """
    The main result. Three configurations on the same test fields:

      baseline  — Study 2's rule: per-field P85 of the POST-injection scores
      clean@85  — same percentile, but calibrated on clean data
      clean@p*  — clean-calibrated, percentile chosen on the calibration split
    """
    # ── choose p* on the calibration split ───────────────────────────
    cal_curve = []
    for p in PCTL_GRID:
        fz = study.frozen("cal", p)
        rows = []
        for seed in seeds:
            d = study.draw(seed)["cal"]
            pred = C.predict(d["scores"], study.slices["cal"], frozen=fz)
            rows.append(C.point_metrics(d["gt"], pred, d["scores"]))
        m = C.agg(rows, ["precision", "recall", "f1", "false_alarms"])
        m["pctl"] = float(p)
        cal_curve.append(m)

    p_star = float(max(cal_curve, key=lambda r: r["f1"])["pctl"])

    # ── report on the test split ─────────────────────────────────────
    configs = {
        "baseline_p85_postinj": dict(pctl=C.BASELINE_PCTL, frozen=None),
        "clean_p85": dict(pctl=None, frozen=study.frozen("test", C.BASELINE_PCTL)),
        "clean_pstar": dict(pctl=None, frozen=study.frozen("test", p_star)),
    }
    results = {}
    preds_cache = {}
    for name, cfg in configs.items():
        rows = []
        for seed in seeds:
            d = study.draw(seed)["test"]
            pred = C.predict(d["scores"], study.slices["test"],
                             pctl=cfg["pctl"], frozen=cfg["frozen"])
            preds_cache[(name, seed)] = pred
            rows.append(C.point_metrics(d["gt"], pred, d["scores"]))
        results[name] = C.agg(rows, ["precision", "recall", "f1", "pr_auc",
                                     "roc_auc", "specificity", "false_alarms"])

    # ── test-split curve, for the figure only (never selected on) ────
    test_curve = []
    for p in PCTL_GRID:
        fz = study.frozen("test", p)
        rows = []
        for seed in seeds:
            d = study.draw(seed)["test"]
            pred = C.predict(d["scores"], study.slices["test"], frozen=fz)
            rows.append(C.point_metrics(d["gt"], pred, d["scores"]))
        m = C.agg(rows, ["precision", "recall", "f1", "false_alarms"])
        m["pctl"] = float(p)
        test_curve.append(m)

    return {"p_star": p_star, "cal_curve": cal_curve, "test_curve": test_curve,
            "results": results, "_preds": preds_cache}


# ══════════════════════════════════════════════════════════════════════
#  E2 — range-based (event-level) metrics
# ══════════════════════════════════════════════════════════════════════
def e2_range_metrics(study, e1, seeds=C.SEEDS, alphas=(0.0, 0.5)):
    """
    Score E1's configurations again, event-wise instead of observation-wise.

    Nothing about the detector changes — this is the same set of predictions
    measured with a metric that asks "was the event caught" rather than "was
    every observation of it caught". Both are reported; the point-wise number
    is the conservative one and stays in every table.
    """
    out = {}
    for name in ("baseline_p85_postinj", "clean_p85", "clean_pstar"):
        for alpha in alphas:
            rows = []
            for seed in seeds:
                d = study.draw(seed)["test"]
                pred = e1["_preds"][(name, seed)]
                rows.append(C.range_metrics(d["gt"], pred,
                                            study.slices["test"], alpha=alpha))
            out[f"{name}@alpha{alpha}"] = C.agg(
                rows, ["range_precision", "range_recall", "range_f1"])
            out[f"{name}@alpha{alpha}"]["n_true_events"] = int(
                np.mean([r["n_true_events"] for r in rows]))
    return out


# ══════════════════════════════════════════════════════════════════════
#  E3 — same-date peer-deviation feature
# ══════════════════════════════════════════════════════════════════════
def e3_peer_feature(seeds=C.SEEDS, p_star=None):
    """
    Refit the whole detector with a 17th feature: how far the field sits from
    the median of its same-date peers. Compared against the 16-feature detector
    at the identical operating point.
    """
    out = {}
    for label, peer in (("base_16", False), ("peer_17", True)):
        s = Study(use_peer_feature=peer)
        pct = p_star if p_star is not None else C.BASELINE_PCTL
        fz = s.frozen("test", pct)
        rows, rrows = [], []
        for seed in seeds:
            d = s.draw(seed)["test"]
            pred = C.predict(d["scores"], s.slices["test"], frozen=fz)
            rows.append(C.point_metrics(d["gt"], pred, d["scores"]))
            rrows.append(C.range_metrics(d["gt"], pred, s.slices["test"], 0.5))
        out[label] = C.agg(rows, ["precision", "recall", "f1", "pr_auc",
                                  "false_alarms"])
        out[label].update(C.agg(rrows, ["range_precision", "range_recall",
                                        "range_f1"]))
        del s
    return out


# ══════════════════════════════════════════════════════════════════════
#  E4 — LOF neighbourhood size
# ══════════════════════════════════════════════════════════════════════
K_GRID = [5, 10, 20, 35, 50, 75]


def e4_kneighbors(seeds=C.SEEDS, p_star=None):
    """
    The main study fixed k=10 and never swept it — a stated limitation.

    The curve is reported on `test` as a sensitivity analysis. The *selected* k
    is whichever maximises F1 on `cal`, so the reported test number is not the
    best cell of the test curve.
    """
    rows_cal, rows_test = [], []
    for k in K_GRID:
        s = Study(n_neighbors=k)
        pct = p_star if p_star is not None else C.BASELINE_PCTL
        for split, sink in (("cal", rows_cal), ("test", rows_test)):
            fz = s.frozen(split, pct)
            rr = []
            for seed in seeds:
                d = s.draw(seed)[split]
                pred = C.predict(d["scores"], s.slices[split], frozen=fz)
                rr.append(C.point_metrics(d["gt"], pred, d["scores"]))
            m = C.agg(rr, ["precision", "recall", "f1", "pr_auc",
                           "false_alarms"])
            m["k"] = k
            sink.append(m)
        del s
    k_star = int(max(rows_cal, key=lambda r: r["f1"])["k"])
    return {"cal": rows_cal, "test": rows_test, "k_star": k_star,
            "k_default": C.LOF["n_neighbors"]}


# ══════════════════════════════════════════════════════════════════════
#  E5 — is adaptive P85 really different from global P90?
# ══════════════════════════════════════════════════════════════════════
def _global_predict(scores, slices, pctl, min_consecutive=C.MIN_CONSECUTIVE):
    """One threshold across every field, then the same persistence rule."""
    thr = np.percentile(scores, pctl)
    labels = np.zeros(len(scores), dtype=int)
    for _t, idx in slices:
        s = scores[idx]
        labels[idx[C.strict_consecutive(s > thr, min_consecutive)]] = 1
    return labels


def e5_significance(study, seeds=C.SEEDS, n_boot=2000, rng_seed=7):
    """
    Study 2 picked adaptive P85 over global P90 on an F1 difference of 0.006,
    with no test behind it. Five seeds cannot support a test, so the resampling
    is over FIELDS — the unit that actually varies — pooling all seeds.

    Reports the paired difference, a bootstrap CI, and a paired permutation
    p-value. A CI spanning zero means the argmax was noise, which is the point.
    """
    slices = study.slices["test"]
    per_field = {"adaptive_p85": [], "global_p90": []}
    agg_rows = {"adaptive_p85": [], "global_p90": []}

    for seed in seeds:
        d = study.draw(seed)["test"]
        preds = {
            "adaptive_p85": C.predict(d["scores"], slices,
                                      pctl=C.BASELINE_PCTL),
            "global_p90": _global_predict(d["scores"], slices, 90),
        }
        for name, pred in preds.items():
            agg_rows[name].append(C.point_metrics(d["gt"], pred, d["scores"]))
            # per-field confusion counts, for the bootstrap
            for _t, idx in slices:
                g, p = d["gt"][idx], pred[idx]
                per_field[name].append((int((g & p).sum()),
                                        int(((1 - g) & p).sum()),
                                        int((g & (1 - p)).sum())))

    def f1_from(counts):
        tp, fp, fn = counts.sum(axis=0)
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else 2 * tp / denom

    A = np.array(per_field["adaptive_p85"])
    B = np.array(per_field["global_p90"])
    observed = f1_from(A) - f1_from(B)

    rng = np.random.RandomState(rng_seed)
    n = len(A)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.randint(0, n, n)
        boot[i] = f1_from(A[pick]) - f1_from(B[pick])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # paired permutation: swap the two rules within a field at random
    perm = np.empty(n_boot)
    for i in range(n_boot):
        swap = rng.rand(n) < 0.5
        AA = np.where(swap[:, None], B, A)
        BB = np.where(swap[:, None], A, B)
        perm[i] = f1_from(AA) - f1_from(BB)
    p_value = float((np.abs(perm) >= abs(observed)).mean())

    return {
        "adaptive_p85": C.agg(agg_rows["adaptive_p85"],
                              ["precision", "recall", "f1", "false_alarms"]),
        "global_p90": C.agg(agg_rows["global_p90"],
                            ["precision", "recall", "f1", "false_alarms"]),
        "delta_f1": float(observed),
        "ci95": [float(lo), float(hi)],
        "p_value": p_value,
        "n_boot": n_boot,
        "n_units": int(n),
        "boot": boot.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════
#  E6 — extreme-value (POT/GPD) thresholding
# ══════════════════════════════════════════════════════════════════════
def _pot_thresholds(clean_scores, slices, q, u_pctl=80.0, pooled_shape=True):
    """
    Peaks-over-threshold, after Siffer et al. (KDD 2017).

    For each field take the excesses of its clean scores over a low anchor u,
    fit a Generalised Pareto tail, and invert it for the score that would be
    exceeded with probability q.

    With 25 observations per field there are only ~5 excesses, far too few to
    fit a shape parameter per field. So the shape is fitted ONCE on the pooled,
    scale-normalised excesses of every field and only the scale varies per
    field. This is the entire reason E6 might fail, and it is fitted on clean
    data only, like every other threshold here.
    """
    anchors, excesses = [], []
    for _t, idx in slices:
        s = clean_scores[idx]
        u = np.percentile(s, u_pctl)
        anchors.append(u)
        excesses.append(s[s > u] - u)

    if pooled_shape:
        pool = np.concatenate([e / e.mean() for e in excesses
                               if len(e) >= 2 and e.mean() > 0])
        xi, _loc, _beta = stats.genpareto.fit(pool, floc=0)
    else:
        xi = 0.1

    out = np.empty(len(slices))
    for i, ((_t, idx), u, e) in enumerate(zip(slices, anchors, excesses)):
        n = len(clean_scores[idx])
        nu = len(e)
        if nu < 2 or e.mean() <= 0:
            out[i] = np.percentile(clean_scores[idx], 99.0)
            continue
        # method-of-moments scale given the pooled shape
        beta = e.mean() * (1 - xi) if xi < 1 else e.mean()
        if beta <= 0:
            out[i] = np.percentile(clean_scores[idx], 99.0)
            continue
        ratio = (q * n / nu)
        if abs(xi) < 1e-6:
            out[i] = u - beta * np.log(ratio)
        else:
            out[i] = u + (beta / xi) * (ratio ** (-xi) - 1)
    return out, float(xi)


Q_GRID = [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]


def e6_evt(study, seeds=C.SEEDS):
    """
    Sweep the risk q on `cal`, report the chosen q on `test`.

    The interesting property is not the F1 — it is that q is a false-alarm
    risk, an interpretable knob, where a percentile on 25 samples saturates and
    cannot express a cut above roughly P96.
    """
    cal_curve = []
    for q in Q_GRID:
        fz, xi = _pot_thresholds(study.clean_scores["cal"], study.slices["cal"], q)
        rows = []
        for seed in seeds:
            d = study.draw(seed)["cal"]
            pred = C.predict(d["scores"], study.slices["cal"], frozen=fz)
            rows.append(C.point_metrics(d["gt"], pred, d["scores"]))
        m = C.agg(rows, ["precision", "recall", "f1", "false_alarms"])
        m["q"] = q
        m["xi"] = xi
        cal_curve.append(m)

    q_star = float(max(cal_curve, key=lambda r: r["f1"])["q"])
    fz, xi = _pot_thresholds(study.clean_scores["test"], study.slices["test"],
                             q_star)
    rows, rrows = [], []
    for seed in seeds:
        d = study.draw(seed)["test"]
        pred = C.predict(d["scores"], study.slices["test"], frozen=fz)
        rows.append(C.point_metrics(d["gt"], pred, d["scores"]))
        rrows.append(C.range_metrics(d["gt"], pred, study.slices["test"], 0.5))
    res = C.agg(rows, ["precision", "recall", "f1", "pr_auc", "false_alarms"])
    res.update(C.agg(rrows, ["range_precision", "range_recall", "range_f1"]))
    return {"q_star": q_star, "xi": xi, "cal_curve": cal_curve, "test": res}
