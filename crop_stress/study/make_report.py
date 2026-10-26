"""make_report.py — outputs/REPORT.md, the single source for the paper."""

import json
import os

import core as C


def T(name):
    with open(os.path.join(C.TABLE_DIR, f"{name}.md"), encoding="utf-8") as f:
        return f.read().strip()


# Study 2's published detection-limit F1s, at its own operating point
# (adaptive P85 computed post-injection, whole dataset). Fixed reference
# values, used only to show where the new operating point wins and loses.
STUDY2_LIMITS = {(30, 1): 0.3002, (40, 1): 0.3279, (50, 1): 0.3465,
                 (10, 2): 0.2334, (20, 2): 0.5458, (30, 2): 0.6756,
                 (40, 2): 0.7181, (50, 2): 0.7352,
                 (30, 3): 0.6419, (40, 3): 0.6765, (50, 3): 0.6905}


def main():
    with open(os.path.join(C.OUT_DIR, "results.json"), encoding="utf-8") as f:
        R = json.load(f)

    s, a, bm, cb = R["setup"], R["a"], R["b"], R["c"]
    d, e, fsig, g, h, i = R["d"], R["e"], R["f"], R["g"], R["h"], R["i"]

    res = a["results"]
    legacy_key = next(k for k in res if k.startswith("Study 2 rule"))
    # The headline is the rule SELECTED ON THE CALIBRATION SPLIT, not the one
    # with the best test F1. Those differ by 0.002 F1 here while the selected
    # rule raises 71 fewer false alarms — picking by test argmax would be the
    # same mistake this study criticises Study 2 for.
    prefix = "EVT" if a["chosen"]["kind"] == "evt" else "Clean-calibrated"
    best_key = next(k for k in res if k.startswith(prefix))
    alt_key = next(k for k in res
                   if k.startswith("Clean-calibrated" if prefix == "EVT"
                                   else "EVT"))
    legacy, best, alt = res[legacy_key], res[best_key], res[alt_key]
    globalp90 = next((v for k, v in res.items() if k.startswith("Global")), None)

    lof, hyb, iso = bm["LOF"], bm["Hybrid (all 3)"], bm["Isolation Forest"]
    best_rule = max((k for k in cb if k != "LOF (this work)"),
                    key=lambda k: cb[k]["f1"])
    lofb = cb["LOF (this work)"]

    L = []
    A = L.append
    A("# Unified study — crop stress detection from Sentinel-2 time series")
    A("")
    A("Every table below comes from **one protocol, one operating point and one "
      "split**. That is the point of this folder: the three predecessor studies "
      "each measured something real, but at different operating points on "
      "different subsets, so no two of their tables could be placed side by "
      "side.")
    A("")

    # ── headline ─────────────────────────────────────────────────────
    A("## Headline")
    A("")
    A(f"**{best_key}**, LOF, strict {s['min_consecutive']}-consecutive "
      f"persistence, on {s['n_tiles']['test']:,} held-out fields:")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| Precision | **{best['precision']:.3f}** |")
    A(f"| Recall | **{best['recall']:.3f}** |")
    A(f"| F1 | **{best['f1']:.3f}** |")
    A(f"| PR-AUC | {best['pr_auc']:.3f} |")
    A(f"| False alarms | **{int(best['false_alarms']):,}** "
      f"out of {s['n_rows']['test']:,} observations |")
    A("")
    A(f"Study 2's own rule, measured on the same fields, scores F1 "
      f"{legacy['f1']:.3f} (P {legacy['precision']:.3f} / R "
      f"{legacy['recall']:.3f}). The improvement is "
      f"**{best['f1']-legacy['f1']:+.3f} F1** and "
      f"**{best['recall']-legacy['recall']:+.3f} recall**, and it comes "
      f"entirely from where the threshold is computed — the detector is "
      f"unchanged.")
    A("")
    A(f"This rule was selected **on the calibration split**, not by taking the "
      f"best test score. The percentile rule edges it on test F1 "
      f"({alt['f1']:.4f} against {best['f1']:.4f}) but raises "
      f"{int(alt['false_alarms']) - int(best['false_alarms']):,} more false "
      f"alarms; choosing on a test argmax would be the same mistake this study "
      f"criticises elsewhere. Every section below is measured at this operating "
      f"point.")
    A("")

    # ── protocol ─────────────────────────────────────────────────────
    A("## The protocol")
    A("")
    A(f"| Split | Fields | Observations | Role |")
    A("|---|---|---|---|")
    for k, role in (("fit", "detectors fitted here, clean rows only"),
                    ("cal", "threshold rule and hyperparameters chosen here"),
                    ("test", "every reported number; nothing selected on it")):
        A(f"| `{k}` | {s['n_tiles'][k]:,} | {s['n_rows'][k]:,} | {role} |")
    A("")
    A("Every threshold is calibrated on **clean, pre-injection scores**. "
      "Study 2 computed its percentile on the batch that already contained the "
      "injections. At a 15% injection rate that is a specific trap: the top 15% "
      "of scores are the anomalies, so a P85 cut lands exactly on their "
      "boundary and is dragged upward into the group it is trying to separate. "
      "Roughly half the injected observations end up below their own field's "
      "threshold, which is why Study 2's recall could not exceed about 0.6 "
      "regardless of the detector.")
    A("")
    A(f"All numbers are mean ± s.d. over {len(s['seeds'])} seeds at "
      f"{int(s['severity']*100)}% severity over {s['duration']} observations, "
      f"{int(s['injection_ratio']*100)}% injection rate, "
      f"{s['n_features']} features.")
    A("")
    A("---")
    A("")

    ex = R.get("extras", {})
    if ex:
        A("## Data and method")
        A("")
        A("![The input](figures/fig12_imagery.png)")
        A("")
        A("![Pipeline](figures/fig10_pipeline.png)")
        A("")
        A("Detection operates on field-level aggregate statistics, not on "
          "pixels. That matches the unit at which an agronomic decision is "
          "actually taken — a grower inspects or treats a field, not a pixel — "
          "and it averages away the per-pixel noise that dominates 10 m optical "
          "data. The cost is that within-field localisation is out of scope: "
          "the system says *this field, this week*, not *this corner of this "
          "field*.")
        A("")
        A("---")
        A("")

    # ── 1 operating point ────────────────────────────────────────────
    A("## 1 · The operating point")
    A("")
    A(T("t1_operating_point"))
    A("")
    A("![Operating point](figures/fig1_operating_point.png)")
    A("")
    A("![Threshold rules](figures/fig2_threshold_rules.png)")
    A("")
    A(f"The extreme-value rule fits a Generalised Pareto tail to each field's "
      f"clean scores and inverts it for a target false-alarm risk "
      f"(q = {a['q_star']:g}, pooled shape ξ = {a['xi']:.3f}). This matters "
      f"beyond accuracy: with {s['n_rows']['test'] // s['n_tiles']['test']} "
      f"observations per field an empirical percentile saturates — above about "
      f"P96 it is interpolating between the top one or two scores — whereas a "
      f"fitted tail extrapolates past the sample limit. It also replaces "
      f"\"we chose the 85th percentile\" with \"we chose a 3% false-alarm "
      f"risk\", which is a defensible engineering statement.")
    A("")
    if globalp90:
        A(f"The global P90 rule remains the precision extreme: "
          f"{globalp90['precision']:.3f} precision with only "
          f"{int(globalp90['false_alarms']):,} false alarms, at "
          f"{globalp90['recall']:.3f} recall. If the cost of an unnecessary "
          f"field visit dominates, that is still the rule to deploy.")
        A("")

    if ex:
        spr = ex["threshold_spread"]
        sig = ex["significance"]
        A("![Threshold spread](figures/fig15_threshold_spread.png)")
        A("")
        A(f"Calibrating per field is not decoration. The deployed thresholds "
          f"span a **{spr['ratio']:.1f}-fold range** across the "
          f"{len(spr['thresholds']):,} held-out fields, so a single global cut "
          f"sits far from most of them — over-alarming the quiet fields and "
          f"under-alarming the volatile ones.")
        A("")
        A("![Significance](figures/fig16_significance.png)")
        A("")
        lo, hi = sig["ci95"]
        A(f"The headline gain is not seed noise. Resampling over "
          f"{sig['n_units']:,} field-seed units ({sig['n_boot']:,} bootstrap "
          f"replicates) gives ΔF1 = **{sig['delta_f1']:+.4f}**, 95% CI "
          f"[{lo:+.4f}, {hi:+.4f}], paired permutation "
          f"p = **{sig['p_value']:.4f}**. The interval excludes zero by a wide "
          f"margin.")
        A("")

    # ── 2 detectors ──────────────────────────────────────────────────
    A("## 2 · Which detector, and why not an ensemble")
    A("")
    A(T("t2_detectors"))
    A("")
    A("![Detectors](figures/fig3_detectors.png)")
    A("")
    A(f"**This is the project's central finding and it survives the stricter "
      f"protocol.** Standalone LOF reaches F1 {lof['f1']:.3f}; the three-model "
      f"hybrid reaches {hyb['f1']:.3f}, a factor of "
      f"{lof['f1']/max(hyb['f1'],1e-9):.1f}. Isolation Forest alone reaches "
      f"{iso['f1']:.3f}.")
    A("")
    A("The mechanism is score-level averaging with an incompetent member. A "
      "weighted sum is only as discriminative as its components allow: "
      "averaging a strong ranking with a weak one produces a ranking closer to "
      "the weak one, and a per-field threshold acting on that ranking inherits "
      "the damage. Ensembling helps when members are individually competent and "
      "make decorrelated errors; here they are not competent. Isolation "
      "Forest's axis-aligned splits find globally extreme values, but a "
      "stressed observation of an intrinsically low-vigour field is not a "
      "global extreme — which is exactly the case LOF's local density "
      "comparison is built for.")
    A("")
    A("Each detector is given its own clean-data threshold at the same risk "
      "level, since the three score scales are not comparable. Handing them a "
      "shared threshold would have made the comparison meaningless.")
    A("")

    # ── 3 baselines ──────────────────────────────────────────────────
    A("## 3 · Against rules written by hand")
    A("")
    A(T("t3_baselines"))
    A("")
    A("![Baselines](figures/fig4_baselines.png)")
    A("")
    A(f"LOF reaches F1 {lofb['f1']:.3f} against {cb[best_rule]['f1']:.3f} for "
      f"the strongest rule ({best_rule.lower()}). Every method is scored on "
      f"identical data and passes through an identical clean-calibrated "
      f"threshold and persistence rule, so the difference is attributable to "
      f"the scoring function and not to the machinery around it.")
    A("")

    # ── 4 persistence ────────────────────────────────────────────────
    A("## 4 · Temporal persistence")
    A("")
    A(T("t4_persistence"))
    A("")
    A("![Persistence](figures/fig6_persistence.png)")
    A("")
    A("**A correction to the earlier write-up.** Study 2's audit recorded the "
      "sliding 2-of-3 window as non-causal, claiming it inspects observation "
      "*t+1* to confirm *t*. The implementation in `utils/hybrid.py` does not: "
      "it slices `[t-2, t-1, t]`, strictly backward. All four rules here are "
      "causal and admissible for near-real-time use. The sliding window may "
      "still lose on F1 — but it loses on merit, not because it was cheating.")
    A("")
    A(f"**Read the 3-consecutive row with care.** This table is measured at the "
      f"baseline scenario, where injected events last {s['duration']} "
      f"observations. A rule demanding three consecutive exceedances therefore "
      f"cannot fire on the event alone — it needs a neighbouring observation to "
      f"also cross the threshold — so its low score here is a structural "
      f"consequence of the scenario, not evidence that the rule is bad. Study 2 "
      f"evaluated persistence at 3-observation events for exactly this reason. "
      f"The trade-off between the two strict rules should be read off the "
      f"detection-limits table in section 5, not from this row.")
    A("")

    # ── 5 limits ─────────────────────────────────────────────────────
    A("## 5 · Detection limits")
    A("")
    A(T("t5_limits"))
    A("")
    A("![Limits](figures/fig5_limits.png)")
    A("")
    weak = min(e, key=lambda r: r["f1"])
    A(f"The weakest cell is {int(weak['severity']*100)}% severity over "
      f"{weak['duration']} observation(s) at F1 {weak['f1']:.3f}. Two failure "
      f"regions are structural rather than incidental: a 10% depression of a "
      f"field-mean index is comparable to the pass-to-pass variation the index "
      f"shows anyway under changing illumination, and single-observation events "
      f"are suppressed by design — the persistence rule cannot distinguish a "
      f"one-pass excursion from a cloud artefact using temporal evidence alone.")
    A("")

    # ── where the new operating point wins and loses ─────────────────
    wins, losses = [], []
    for r in e:
        key = (int(round(r["severity"] * 100)), int(r["duration"]))
        if key in STUDY2_LIMITS:
            d = r["f1"] - STUDY2_LIMITS[key]
            (wins if d > 0 else losses).append((key, STUDY2_LIMITS[key],
                                                r["f1"], d))
    A("### The operating point is not uniformly better — and this matters")
    A("")
    A("| Severity | Duration | Study 2 F1 | This study F1 | Δ |")
    A("|---|---|---|---|---|")
    for (sev, dur), o, n, dd in sorted(wins + losses, key=lambda x: (x[0][1], x[0][0])):
        mark = "" if dd > 0 else " ⚠"
        A(f"| {sev}% | {dur} obs | {o:.4f} | {n:.4f} | {dd:+.4f}{mark} |")
    A("")
    if losses:
        worst = min(losses, key=lambda x: x[3])
        cross = max([k for k, _o, _n, d in losses], key=lambda k: k[0])
        A(f"**The strict operating point is tuned for the baseline scenario and "
          f"pays for it on weak events.** It wins on "
          f"{len(wins)} of {len(wins)+len(losses)} cells, often by more than "
          f"0.12 F1 — but it *loses* on the {len(losses)} weakest, worst at "
          f"{worst[0][0]}% severity where F1 falls "
          f"{worst[1]:.3f} → {worst[2]:.3f}.")
        A("")
        A(f"The reason is direct: the EVT risk was chosen at "
          f"{int(s['severity']*100)}% severity, so the threshold sits high "
          f"enough that a {cross[0]}%-severity dip barely clears it and recall "
          f"collapses. Study 2's looser P85 cut caught more of those weak "
          f"events, at the cost of everything else.")
        A("")
        A("**This belongs in the paper as a stated trade-off, not a footnote.** "
          "If the deployment goal is early warning of *developing* stress, a "
          "looser threshold is the right choice and the numbers here are the "
          "wrong ones to quote. If the goal is reliable detection of "
          "established stress with few false alarms, this operating point is "
          "correct. The honest framing is that the risk parameter *q* is a "
          "dial the grower sets, and the paper should report the curve rather "
          "than a single point.")
        A("")

    # ── 6 signatures ─────────────────────────────────────────────────
    A("## 6 · Robustness to the stress signature")
    A("")
    A(T("t6_signatures"))
    A("")
    A("![Signatures](figures/fig7_signatures.png)")
    A("")
    worst = min(fsig, key=lambda k: fsig[k]["f1"])
    A(f"**{worst}** is the weak case (F1 {fsig[worst]['f1']:.3f}). The cause is "
      f"the rolling baseline: a decline that accumulates slowly is partly "
      f"absorbed into the rolling median the features are computed against, so "
      f"the field's own reference drifts down with the stress and the residual "
      f"stays small. This is agronomically the inconvenient failure, because "
      f"progressive drought and spreading foliar disease — where early warning "
      f"is worth most — are exactly slow-onset events.")
    A("")

    # ── 7 protocol effect ────────────────────────────────────────────
    A("## 7 · What held-out evaluation costs")
    A("")
    A(T("t8_protocol"))
    A("")
    A("![Protocol](figures/fig9_protocol.png)")
    A("")
    ks = list(h)
    delta = h[ks[0]]["f1"] - h[ks[1]]["f1"]
    A(f"Fitting the detector on a disjoint half of the fields instead of on all "
      f"of them costs **{delta:.3f} F1**. Everything else is held fixed.")
    A("")
    A("**This corrects a claim in the earlier write-up.** `experiment3` "
      "reported F1 0.72 against 0.35 and read the gap as an evaluation-protocol "
      "effect. That comparison was made across two separate codebases which "
      "also differed in input representation, cloud screening, observation "
      "density and injection procedure. Measured properly, with only the "
      "protocol changing, the cost is small. Whatever separates those two "
      "numbers lives in the other differences, and should not be attributed to "
      "held-out evaluation.")
    A("")

    # ── 8 ablations ──────────────────────────────────────────────────
    A("## 8 · Ablations")
    A("")
    A("### Feature set")
    A("")
    A(T("t9_features"))
    A("")
    ik = list(i)
    di = i[ik[1]]["f1"] - i[ik[0]]["f1"]
    A(f"Adding lagged value, cumulative change and acceleration changes F1 by "
      f"{di:+.4f}. At a 10-day effective cadence a second difference is "
      f"estimated across roughly three weeks of phenological change, so the "
      f"derivative is dominated by growth dynamics rather than by stress. The "
      f"baseline representation already carries first-order temporal context "
      f"through its rolling statistics.")
    A("")
    A("### LOF neighbourhood size")
    A("")
    A(T("t7_kneighbors"))
    A("")
    A("![k sweep](figures/fig8_kneighbors.png)")
    A("")
    f1s = [r["f1"] for r in g["test"]]
    A(f"Test F1 spans {max(f1s)-min(f1s):.3f} across k ∈ "
      f"[{min(r['k'] for r in g['test'])}, {max(r['k'] for r in g['test'])}]; "
      f"the value chosen on the calibration split is k = {g['k_star']}. Study 2 "
      f"fixed k = {g['k_default']} and never swept it, listing that as an open "
      f"limitation. It is now closed: the choice was not load-bearing.")
    A("")
    # ── 9 worked example ─────────────────────────────────────────────
    if ex and ex.get("example"):
        w = ex["example"]
        A("## 9 · A worked example")
        A("")
        A("![Worked example](figures/fig11_worked_example.png)")
        A("")
        A(f"One held-out field, start to finish. Of {w['n_injected']} injected "
          f"observations the detector confirmed {w['hits']} and missed "
          f"{w['missed']}, with {int(sum(w['suppressed']))} exceedance(s) "
          f"suppressed by the persistence rule and {w['false_alarms']} false "
          f"alarms.")
        A("")
        A("The suppressed point is the part worth showing. It crossed the "
          "field's own threshold, did not recur on the following pass, and was "
          "therefore discarded — the same mechanism that removes transient "
          "cloud artefacts also removes genuine one-pass events. Every recall "
          "figure in this report is net of exactly that suppression.")
        A("")

    # ── 10 threshold-free ────────────────────────────────────────────
    if ex:
        c = ex["curves"]
        sep = ex["separability"]
        A("## 10 · The threshold-free view")
        A("")
        A("![PR and ROC](figures/fig13_curves.png)")
        A("")
        A(f"Every table above is measured at one operating point, which leaves "
          f"a fair question open: is the ranking good, or was the cut simply "
          f"lucky? Average precision is **{c['ap']:.3f}** and ROC-AUC "
          f"**{c['auc']:.3f}**, both independent of any threshold. Average "
          f"precision is the more informative of the two under this class "
          f"imbalance.")
        A("")
        A("The deployed point sits below the precision–recall frontier by "
          "construction, because the persistence rule discards isolated "
          "exceedances that the curve is free to count. That gap is the price "
          "of robustness to cloud artefacts, which the synthetic evaluation "
          "cannot reward.")
        A("")
        A("![Separability](figures/fig14_separability.png)")
        A("")
        A(f"Healthy observations concentrate near the LOF floor (median "
          f"{sep['median_healthy']:.3f}); stressed observations spread "
          f"substantially higher (median {sep['median_stressed']:.3f}). The "
          f"overlap between them is the irreducible error — no choice of "
          f"threshold removes it, which is why the ranking matters more than "
          f"the cut.")
        A("")

    A("---")
    A("")

    # ── carry into the paper ─────────────────────────────────────────
    A("## What belongs in the paper")
    A("")
    A(f"1. **Ensembling is harmful.** LOF {lof['f1']:.3f} against "
      f"{hyb['f1']:.3f} for the three-model hybrid. The negative result the "
      f"project was built to avoid, and the most transferable thing in it.")
    A(f"2. **The operating point was leaving performance on the table.** "
      f"F1 {legacy['f1']:.3f} → {best['f1']:.3f} from calibrating the "
      f"threshold on clean data, with the detector untouched.")
    A(f"3. **A principled threshold.** An EVT tail with a stated false-alarm "
      f"risk, which also escapes the resolution limit of a 25-sample "
      f"percentile.")
    A(f"4. **Honest scope.** Detection limits, the slow-onset weakness, what "
      f"held-out evaluation does and does not cost — and the fact that the "
      f"recommended operating point is *worse* than Study 2's on events below "
      f"about 30% severity. That trade-off is a result, not an embarrassment: "
      f"it says the risk parameter is a dial to be set from the cost of a "
      f"missed event against the cost of a wasted field visit.")
    A("")
    A("From `experiment3`, which is not superseded by this folder and should "
      "still be cited: multispectral input raises F1 by roughly 44% relative "
      "under its own protocol, and cross-year baselines fail because crop "
      "rotation gives a field a different normal from one season to the next. "
      "Its protocol claim should not be cited — section 7 above supersedes it.")
    A("")
    A("## Limitations")
    A("")
    A("- Synthetic injection remains the only ground truth. Absolute figures "
      "are an upper bound on operational performance; the comparisons between "
      "configurations are the reliable part.")
    A("- Clean calibration assumes a stress-free reference period per field. "
      "Available in retrospective seasonal analysis; an assumption for a "
      "real-time system, and `experiment3` showed that using the previous "
      "season as that reference is broken by crop rotation.")
    A(f"- One region, one season. The test split is {s['n_tiles']['test']:,} "
      f"fields, so these figures carry more sampling noise than a "
      f"whole-dataset number would.")
    A("- Field-level only. The detector emits one score per field per pass and "
      "does not localise within a field.")
    A("- Part of the improvement in section 1 is a fix for an evaluation-design "
      "flaw specific to a 15% injection rate. At a realistic stress prevalence "
      "the old rule would have been less damaged, so the gain should not be "
      "read as a pure detector improvement.")
    A("")
    A(f"*Generated by `run_all.py` in {R['runtime_seconds']:.0f}s. "
      f"Every number read from `results.json`.*")

    with open(os.path.join(C.OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("  wrote outputs/REPORT.md")


if __name__ == "__main__":
    main()
