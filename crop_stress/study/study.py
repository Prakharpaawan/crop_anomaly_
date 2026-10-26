"""
study.py — every ablation, all under the one protocol defined in core.py.

Order matters: the operating point is settled first (A), and every later
ablation is then measured at that operating point. That is the whole reason
this folder exists — in the three predecessor studies each table used whatever
threshold was current at the time, so no two were comparable.
"""

import numpy as np

import core as C


# ══════════════════════════════════════════════════════════════════════
#  Fitted state, built once
# ══════════════════════════════════════════════════════════════════════
class Bench:
    def __init__(self, verbose=True):
        self.base = C.load_base()
        self.tiles = self.base["tiles"]
        self.splits = C.make_splits(self.tiles)
        self.mask = {k: np.isin(self.tiles, v) for k, v in self.splits.items()}
        self.tiles_of = {k: self.tiles[m] for k, m in self.mask.items()}
        self.slices = {k: C.tile_slices(t) for k, t in self.tiles_of.items()}
        self.X = self.base["X"]

        Xfit = self.X[self.mask["fit"]]
        if verbose:
            print(f"  fitting LOF on {Xfit.shape[0]:,} clean rows...")
        self.lof = C.fit_lof(Xfit)
        if verbose:
            print("  fitting Isolation Forest...")
        self.iso = C.fit_iso(Xfit)
        if verbose:
            print("  training autoencoder (200 epochs)...")
        self.ae = C.fit_ae(Xfit, verbose=verbose)

        self.clean = {k: C.score_lof(*self.lof, self.X[self.mask[k]])
                      for k in ("cal", "test")}
        self._draw = {}

    # ── injected draws, per (kind, severity, duration, seed) ─────────
    def draw(self, seed, severity=C.SEVERITY, duration=C.DURATION,
             kind="sudden"):
        key = (kind, severity, duration, seed)
        if key in self._draw:
            return self._draw[key]
        Xm, gt, _p50 = C.injected(self.base["df"], self.tiles, seed,
                                  severity, duration, kind)
        out = {}
        for k in ("cal", "test"):
            m = self.mask[k]
            out[k] = {"X": Xm[m], "gt": gt[m]}
        self._draw[key] = out
        return out

    def lof_scores(self, split, seed, **kw):
        return C.score_lof(*self.lof, self.draw(seed, **kw)[split]["X"])


# ══════════════════════════════════════════════════════════════════════
#  A · The operating point — settled first, used by everything after
# ══════════════════════════════════════════════════════════════════════
PCTL_GRID = np.round(np.r_[np.arange(80.0, 96.0, 1.0),
                           np.arange(96.0, 99.91, 0.25)], 2)
Q_GRID = [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]


def a_operating_point(b, seeds=C.SEEDS):
    """
    Four candidate threshold rules, all with strict 2-consecutive persistence.
    The percentile and the EVT risk are each chosen on `cal`; everything is
    then reported on `test`.
    """
    def eval_on(split, **thr):
        rows = []
        for s in seeds:
            d = b.draw(s)[split]
            sc = C.score_lof(*b.lof, d["X"])
            pred = C.predict(sc, b.slices[split], C.MIN_CONSECUTIVE, **thr)
            rows.append(C.metrics(d["gt"], pred, sc))
        return C.agg(rows)

    # choose the clean percentile on cal
    pctl_curve = []
    for p in PCTL_GRID:
        m = eval_on("cal", frozen=C.clean_thresholds(b.clean["cal"],
                                                     b.slices["cal"], p))
        m["pctl"] = float(p)
        pctl_curve.append(m)
    p_star = float(max(pctl_curve, key=lambda r: r["f1"])["pctl"])

    # choose the EVT risk on cal
    evt_curve = []
    for q in Q_GRID:
        fz, xi = C.evt_thresholds(b.clean["cal"], b.slices["cal"], q)
        m = eval_on("cal", frozen=fz)
        m["q"] = q
        m["xi"] = xi
        evt_curve.append(m)
    q_star = float(max(evt_curve, key=lambda r: r["f1"])["q"])

    fz_test_p, _ = (C.clean_thresholds(b.clean["test"], b.slices["test"],
                                       p_star), None)
    fz_test_e, xi = C.evt_thresholds(b.clean["test"], b.slices["test"], q_star)

    results = {
        f"Study 2 rule: adaptive P{C.LEGACY_PCTL}, post-injection":
            eval_on("test", pctl=C.LEGACY_PCTL),
        "Global P90, post-injection":
            eval_on("test", global_pctl=90),
        f"Clean-calibrated P{p_star:g}":
            eval_on("test", frozen=fz_test_p),
        f"EVT / GPD tail (q={q_star:g})":
            eval_on("test", frozen=fz_test_e),
    }
    # the winner on cal decides what every later ablation uses
    best_name = max(("Clean-calibrated", "EVT"),
                    key=lambda t: (max(r["f1"] for r in pctl_curve)
                                   if t == "Clean-calibrated"
                                   else max(r["f1"] for r in evt_curve)))
    chosen = {"kind": "evt" if best_name == "EVT" else "pctl",
              "p_star": p_star, "q_star": q_star, "xi": xi}
    return {"pctl_curve": pctl_curve, "evt_curve": evt_curve,
            "p_star": p_star, "q_star": q_star, "xi": xi,
            "results": results, "chosen": chosen}


def operating_thresholds(b, split, chosen):
    """The frozen per-field thresholds for the chosen operating point."""
    if chosen["kind"] == "evt":
        fz, _ = C.evt_thresholds(b.clean[split], b.slices[split],
                                 chosen["q_star"])
        return fz
    return C.clean_thresholds(b.clean[split], b.slices[split],
                              chosen["p_star"])


# ══════════════════════════════════════════════════════════════════════
#  B · Detector comparison — the ensemble result, re-measured honestly
# ══════════════════════════════════════════════════════════════════════
def b_models(b, chosen, seeds=C.SEEDS):
    fz = operating_thresholds(b, "test", chosen)
    names = ["Isolation Forest", "LOF", "Autoencoder", "ISO + LOF",
             "Hybrid (all 3)"]
    rows = {n: [] for n in names}
    for s in seeds:
        d = b.draw(s)["test"]
        iso = C.score_iso(*b.iso, d["X"])
        lof = C.score_lof(*b.lof, d["X"])
        ae = C.score_ae(*b.ae, d["X"])
        ni, nl, na = C.minmax(iso), C.minmax(lof), C.minmax(ae)
        blends = {"Isolation Forest": iso, "LOF": lof, "Autoencoder": ae,
                  "ISO + LOF": 0.5 * ni + 0.5 * nl,
                  "Hybrid (all 3)": 0.33 * ni + 0.33 * nl + 0.34 * na}
        for n, sc in blends.items():
            # each detector's own score scale needs its own clean threshold
            f = fz if n == "LOF" else _clean_thr_for(b, n, chosen)
            pred = C.predict(sc, b.slices["test"], C.MIN_CONSECUTIVE, frozen=f)
            rows[n].append(C.metrics(d["gt"], pred, sc))
    return {n: C.agg(v) for n, v in rows.items()}


_THR_CACHE = {}


def _clean_thr_for(b, name, chosen):
    """
    Clean-data thresholds for a non-LOF scorer.

    Each detector produces scores on its own scale, so a threshold calibrated
    on LOF scores is meaningless for Isolation Forest. Every scorer gets its
    own clean-data threshold at the same percentile, which is what keeps the
    comparison fair.
    """
    if name in _THR_CACHE:
        return _THR_CACHE[name]
    Xc = b.X[b.mask["test"]]
    if name == "Isolation Forest":
        sc = C.score_iso(*b.iso, Xc)
    elif name == "Autoencoder":
        sc = C.score_ae(*b.ae, Xc)
    elif name == "ISO + LOF":
        sc = 0.5 * C.minmax(C.score_iso(*b.iso, Xc)) + \
             0.5 * C.minmax(C.score_lof(*b.lof, Xc))
    else:
        sc = 0.33 * C.minmax(C.score_iso(*b.iso, Xc)) + \
             0.33 * C.minmax(C.score_lof(*b.lof, Xc)) + \
             0.34 * C.minmax(C.score_ae(*b.ae, Xc))
    if chosen["kind"] == "evt":
        thr, _ = C.evt_thresholds(sc, b.slices["test"], chosen["q_star"])
    else:
        thr = C.clean_thresholds(sc, b.slices["test"], chosen["p_star"])
    _THR_CACHE[name] = thr
    return thr


# ══════════════════════════════════════════════════════════════════════
#  C · Rule-based baselines
# ══════════════════════════════════════════════════════════════════════
def c_baselines(b, chosen, seeds=C.SEEDS):
    fz = operating_thresholds(b, "test", chosen)
    m = b.mask["test"]
    rows = {"LOF (this work)": []}
    rows.update({k: [] for k in C.RULES})
    for s in seeds:
        df_mod, gt = C.injected_frame(b.base["df"], b.tiles, s)
        sub = df_mod[m]
        gt_t = gt[m]
        lof = C.score_lof(*b.lof, C.feature_matrix(sub))
        pred = C.predict(lof, b.slices["test"], C.MIN_CONSECUTIVE, frozen=fz)
        rows["LOF (this work)"].append(C.metrics(gt_t, pred, lof))
        for name, fn in C.RULES.items():
            sc = fn(sub)
            # rules get a clean-data threshold too, on the same footing
            thr = C.clean_thresholds(fn(b.base["df"][m]), b.slices["test"],
                                     chosen["p_star"])
            p = C.predict(sc, b.slices["test"], C.MIN_CONSECUTIVE, frozen=thr)
            rows[name].append(C.metrics(gt_t, p, sc))
    return {k: C.agg(v) for k, v in rows.items()}


# ══════════════════════════════════════════════════════════════════════
#  D · Persistence
# ══════════════════════════════════════════════════════════════════════
def d_persistence(b, chosen, seeds=C.SEEDS):
    fz = operating_thresholds(b, "test", chosen)
    rules = {"None": "none", "Strict 2-consecutive": 2,
             "Sliding 2-of-3 (causal)": "sliding_2of3",
             "Strict 3-consecutive": 3}
    rows = {k: [] for k in rules}
    for s in seeds:
        d = b.draw(s)["test"]
        sc = C.score_lof(*b.lof, d["X"])
        for name, rule in rules.items():
            pred = C.predict(sc, b.slices["test"], rule, frozen=fz)
            rows[name].append(C.metrics(d["gt"], pred, sc))
    return {k: C.agg(v) for k, v in rows.items()}


# ══════════════════════════════════════════════════════════════════════
#  E · Detection limits
# ══════════════════════════════════════════════════════════════════════
GRID = [(0.30, 1), (0.40, 1), (0.50, 1),
        (0.10, 2), (0.20, 2), (0.30, 2), (0.40, 2), (0.50, 2),
        (0.30, 3), (0.40, 3), (0.50, 3)]


def e_limits(b, chosen, seeds=C.SEEDS, verbose=True):
    fz = operating_thresholds(b, "test", chosen)
    out = []
    for sev, dur in GRID:
        if verbose:
            print(f"    severity {int(sev*100)}%, duration {dur}")
        rows = []
        for s in seeds:
            d = b.draw(s, severity=sev, duration=dur)["test"]
            sc = C.score_lof(*b.lof, d["X"])
            pred = C.predict(sc, b.slices["test"], C.MIN_CONSECUTIVE, frozen=fz)
            rows.append(C.metrics(d["gt"], pred, sc))
        m = C.agg(rows)
        m["severity"] = sev
        m["duration"] = dur
        out.append(m)
    return out


# ══════════════════════════════════════════════════════════════════════
#  F · Anomaly signature
# ══════════════════════════════════════════════════════════════════════
def f_signatures(b, chosen, seeds=C.SEEDS, verbose=True):
    fz = operating_thresholds(b, "test", chosen)
    labels = {"sudden": "Sudden drop", "gradual": "Gradual onset",
              "spectral": "Spectral shift"}
    out = {}
    for kind, label in labels.items():
        if verbose:
            print(f"    {label}")
        rows = []
        for s in seeds:
            d = b.draw(s, kind=kind)["test"]
            sc = C.score_lof(*b.lof, d["X"])
            pred = C.predict(sc, b.slices["test"], C.MIN_CONSECUTIVE, frozen=fz)
            rows.append(C.metrics(d["gt"], pred, sc))
        out[label] = C.agg(rows)
    return out


# ══════════════════════════════════════════════════════════════════════
#  G · LOF neighbourhood size
# ══════════════════════════════════════════════════════════════════════
K_GRID = [5, 10, 20, 35, 50, 75]


def g_kneighbors(b, chosen, seeds=C.SEEDS, verbose=True):
    cal_rows, test_rows = [], []
    for k in K_GRID:
        if verbose:
            print(f"    k={k}")
        sc_pair = C.fit_lof(b.X[b.mask["fit"]], k=k)
        for split, sink in (("cal", cal_rows), ("test", test_rows)):
            clean = C.score_lof(*sc_pair, b.X[b.mask[split]])
            if chosen["kind"] == "evt":
                fz, _ = C.evt_thresholds(clean, b.slices[split],
                                         chosen["q_star"])
            else:
                fz = C.clean_thresholds(clean, b.slices[split],
                                        chosen["p_star"])
            rows = []
            for s in seeds:
                d = b.draw(s)[split]
                sc = C.score_lof(*sc_pair, d["X"])
                pred = C.predict(sc, b.slices[split], C.MIN_CONSECUTIVE,
                                 frozen=fz)
                rows.append(C.metrics(d["gt"], pred, sc))
            m = C.agg(rows)
            m["k"] = k
            sink.append(m)
    return {"cal": cal_rows, "test": test_rows,
            "k_star": int(max(cal_rows, key=lambda r: r["f1"])["k"]),
            "k_default": C.LOF["n_neighbors"]}


# ══════════════════════════════════════════════════════════════════════
#  H · Protocol effect — what held-out evaluation actually costs
# ══════════════════════════════════════════════════════════════════════
def h_protocol(b, seeds=C.SEEDS, verbose=True):
    """
    Same-fields versus held-out, with everything else held fixed.

    This is the experiment that corrects the record: `experiment3` reported
    F1 0.72 -> 0.35 and read it as a protocol effect. Fitting on all fields
    versus fitting on a disjoint half is the only thing that changes here.
    """
    if verbose:
        print("    refitting LOF on ALL fields for the same-fields arm...")
    lof_all = C.fit_lof(b.X)
    rows_all, rows_held = [], []
    for s in seeds:
        d = b.draw(s)["test"]
        sc_all = C.score_lof(*lof_all, d["X"])
        sc_held = C.score_lof(*b.lof, d["X"])
        for sc, sink in ((sc_all, rows_all), (sc_held, rows_held)):
            pred = C.predict(sc, b.slices["test"], C.MIN_CONSECUTIVE,
                             pctl=C.LEGACY_PCTL)
            sink.append(C.metrics(d["gt"], pred, sc))
    return {"Same fields (fitted on all, incl. test)": C.agg(rows_all),
            "Held-out fields (fitted on a disjoint half)": C.agg(rows_held)}


# ══════════════════════════════════════════════════════════════════════
#  I · Feature ablation
# ══════════════════════════════════════════════════════════════════════
def i_features(b, chosen, seeds=C.SEEDS, verbose=True):
    """Baseline 16 features versus the 3 extra trajectory features."""
    if verbose:
        print("    building trajectory features...")
    df_ext = C.add_trajectory(b.base["df"])
    ext_cols = list(C.FEATURE_COLUMNS) + C.TRAJECTORY_FEATURES
    Xe = C.feature_matrix(df_ext, ext_cols)

    sc_pair = C.fit_lof(Xe[b.mask["fit"]])
    clean_e = C.score_lof(*sc_pair, Xe[b.mask["test"]])
    if chosen["kind"] == "evt":
        fz_e, _ = C.evt_thresholds(clean_e, b.slices["test"], chosen["q_star"])
    else:
        fz_e = C.clean_thresholds(clean_e, b.slices["test"], chosen["p_star"])
    fz_b = operating_thresholds(b, "test", chosen)

    rows_b, rows_e = [], []
    for s in seeds:
        d = b.draw(s)["test"]
        sc_b = C.score_lof(*b.lof, d["X"])
        rows_b.append(C.metrics(d["gt"],
                                C.predict(sc_b, b.slices["test"],
                                          C.MIN_CONSECUTIVE, frozen=fz_b),
                                sc_b))
        df_mod, gt = C.injected_frame(b.base["df"], b.tiles, s)
        Xme = C.feature_matrix(C.add_trajectory(df_mod), ext_cols)[b.mask["test"]]
        sc_e = C.score_lof(*sc_pair, Xme)
        rows_e.append(C.metrics(gt[b.mask["test"]],
                                C.predict(sc_e, b.slices["test"],
                                          C.MIN_CONSECUTIVE, frozen=fz_e),
                                sc_e))
    return {"Baseline (16 features)": C.agg(rows_b),
            "+ trajectory features (19)": C.agg(rows_e)}
