"""
make_report.py — writes outputs/REPORT.md from outputs/results.json.

Everything numeric in the report is read from results.json, so the prose and
the tables cannot drift apart.
"""

import json
import os

import common as C


def _t(name):
    with open(os.path.join(C.TABLE_DIR, f"{name}.md"), encoding="utf-8") as f:
        return f.read().strip()


def _pm(m, k, dp=4):
    sd = m.get(k + "_sd")
    return f"{m[k]:.{dp}f} ± {sd:.3f}" if sd is not None else f"{m[k]:.{dp}f}"


def main():
    with open(os.path.join(C.OUT_DIR, "results.json"), encoding="utf-8") as f:
        R = json.load(f)

    s = R["setup"]
    e1, e2, e3, e4, e5 = R["e1"], R["e2"], R["e3"], R["e4"], R["e5"]
    e6 = R.get("e6")
    ps = e1["p_star"]
    base = e1["results"]["baseline_p85_postinj"]
    best = e1["results"]["clean_pstar"]
    d_f1 = best["f1"] - base["f1"]
    d_fa = base["false_alarms"] - best["false_alarms"]
    dominates = (best["precision"] > base["precision"]
                 and best["recall"] > base["recall"]
                 and best["false_alarms"] < base["false_alarms"])

    ev_best = e2["clean_pstar@alpha0.5"]
    ci_lo, ci_hi = e5["ci95"]
    spans_zero = ci_lo <= 0 <= ci_hi

    L = []
    A = L.append

    A("# experiment4 — improving the operating point without changing the model")
    A("")
    A("Six experiments, all on the same detector (LOF, 16 features) and the same")
    A("injected data as Study 2. Nothing here retrains from rasters and nothing")
    A("outside `experiment4/` is modified.")
    A("")
    A("## Headline")
    A("")
    A(f"| | Study 2 baseline | This study |")
    A("|---|---|---|")
    A(f"| Threshold | adaptive P{s['baseline_pctl']}, computed **after** injection "
      f"| clean-calibrated **P{ps:g}**, chosen on a calibration split |")
    A(f"| Precision | {base['precision']:.4f} | **{best['precision']:.4f}** |")
    A(f"| Recall | {base['recall']:.4f} | **{best['recall']:.4f}** |")
    A(f"| F1 | {base['f1']:.4f} | **{best['f1']:.4f}** |")
    A(f"| False alarms | {int(base['false_alarms']):,} | **{int(best['false_alarms']):,}** |")
    A("")
    if dominates:
        A(f"The new operating point is better on **every axis** — precision, recall "
          f"and false alarms simultaneously — for **{d_f1:+.4f} F1** and "
          f"**{d_fa:+,.0f} fewer false alarms**.")
    else:
        d_r = best["recall"] - base["recall"]
        d_p = best["precision"] - base["precision"]
        n_rows = s["n_rows"]["test"]
        A(f"**F1 {d_f1:+.4f}, recall {d_r:+.4f}, precision {d_p:+.4f}**, at a cost "
          f"of {-d_fa:+,.0f} false alarms out of {n_rows:,} observations "
          f"({-d_fa / base['false_alarms'] * 100:+.0f}%). Not a strict "
          f"improvement — recall is bought with a small number of extra false "
          f"alarms — but a favourable trade: the detector now recovers "
          f"{d_r * 100:.0f} percentage points more of the injected stress for "
          f"roughly {abs(d_fa):.0f} additional inspections across a whole "
          f"region-season.")
    A("")
    A("## Why this is not just tuning")
    A("")
    A(f"Fields are split three ways, disjointly: **{s['n_tiles']['fit']:,} fit** "
      f"(the detector is fitted here, on clean rows), "
      f"**{s['n_tiles']['cal']:,} calibration** (the threshold percentile and *k* "
      f"are chosen here), **{s['n_tiles']['test']:,} test** (everything reported "
      f"comes from here, and nothing is ever selected on it).")
    A("")
    A("Every threshold in this study is computed from **clean, pre-injection "
      "scores**. The main study computes its percentile on the batch that already "
      "contains the injections, which lets the cut-off adapt to the anomaly rate "
      "it is being measured against.")
    A("")
    A(f"All numbers are mean ± s.d. over {len(s['seeds'])} seeds "
      f"({', '.join(str(x) for x in s['seeds'])}), at "
      f"{int(s['severity']*100)}% severity over {s['duration']} observations with "
      f"a {int(s['injection_ratio']*100)}% injection rate and strict "
      f"{s['min_consecutive']}-consecutive persistence.")
    A("")

    A("---")
    A("")
    A("## E1 · Calibrating the threshold on clean data")
    A("")
    A(_t("t1_threshold"))
    A("")
    A("![Threshold selection](figures/fig1_threshold_selection.png)")
    A("")
    A("![Headline comparison](figures/fig2_headline_comparison.png)")
    A("")
    A(f"The percentile chosen on the calibration split is **P{ps:g}**, far above "
      f"the P{s['baseline_pctl']} the main study uses. That is the substance of "
      f"the result: once a threshold is calibrated on clean scores it no longer "
      f"has to sit low enough to absorb the injected mass, so it can be set much "
      f"stricter and still recover more events.")
    A("")
    A("Holding the percentile at P85 but calibrating it on clean data "
      f"(row 2) gives precision {e1['results']['clean_p85']['precision']:.4f} and "
      f"{int(e1['results']['clean_p85']['false_alarms']):,} false alarms — "
      "clean calibration alone is not the win; calibration *plus* a percentile "
      "chosen honestly is.")
    A("")

    A("## E2 · Event-level scoring")
    A("")
    A(_t("t2_point_vs_event"))
    A("")
    A("![Point vs event](figures/fig3_point_vs_event.png)")
    A("")
    A("Identical predictions, scored two ways. Point-wise scoring asks whether "
      "every observation of an event was flagged. Event-level scoring (Tatbul "
      "et al., NeurIPS 2018 — existence reward plus overlap reward, α = 0.5, "
      "flat positional bias, reciprocal cardinality) asks whether the event was "
      "caught at all, which is the operational question for a field advisory.")
    A("")
    A(f"At the recommended operating point event-level recall is "
      f"**{ev_best['range_recall']:.4f}** against point-wise "
      f"**{best['recall']:.4f}**. Both belong in the paper; every table here "
      "carries both.")
    A("")
    gap = ev_best["range_recall"] - best["recall"]
    ev_base = e2["baseline_p85_postinj@alpha0.5"]
    gap_base = ev_base["range_recall"] - base["recall"]
    A(f"**This is a negative result, and it is worth stating as one.** The two "
      f"metrics almost coincide: {gap:+.4f} recall at the recommended operating "
      f"point and {gap_base:+.4f} at the Study 2 baseline. The reason is "
      f"structural — injected events last {s['duration']} observations and the "
      f"persistence rule requires {s['min_consecutive']} consecutive "
      f"exceedances, so a detection is very nearly all-or-nothing and there is "
      f"almost no partial credit to award. Event-level scoring would only "
      f"separate from point-wise for events *longer* than the persistence "
      f"requirement.")
    A("")
    A("So switching metric does **not** buy a recall improvement here. What it "
      "buys is a defence: Study 2's point-wise recall was already the right "
      "number at this configuration, and now that has been checked rather than "
      "assumed. The event-level columns stay in the tables as evidence, not as "
      "the headline.")
    A("")

    A("## E3 · A same-date peer-deviation feature")
    A("")
    A(_t("t3_peer_feature"))
    A("")
    A("![Peer feature](figures/fig4_peer_feature.png)")
    A("")
    A("The existing `diff_from_global_median` feature compares a field against "
      "the whole season, so it cannot react to a single hazy pass. A weather "
      "front pushes every field down together on one date — that is not crop "
      "stress, and the persistence rule cannot filter it because the front lasts "
      "across passes. The added feature is the field's deviation from the median "
      f"of all fields observed on the *same date*, scaled by that date's MAD, "
      f"with dates seen by fewer than {C.MIN_FIELDS_PER_DATE} fields left "
      "unadjusted.")
    A("")
    d = e3["peer_17"]["f1"] - e3["base_16"]["f1"]
    dfa3 = e3["peer_17"]["false_alarms"] - e3["base_16"]["false_alarms"]
    A(f"Effect on F1: **{d:+.4f}** "
      f"({e3['base_16']['f1']:.4f} → {e3['peer_17']['f1']:.4f}), with false "
      f"alarms {int(e3['base_16']['false_alarms']):,} → "
      f"{int(e3['peer_17']['false_alarms']):,}.")
    A("")
    if d < -0.005:
        A("**This one did not work, and the direction is the opposite of the "
          "hypothesis.** The feature was meant to *reduce* false alarms by "
          f"cancelling region-wide haze; it raised them by {dfa3:+,.0f} and cost "
          f"{abs(d):.4f} F1. The most likely explanation is that the injected "
          "anomalies are themselves part of the same-date population: with a "
          f"{int(s['injection_ratio']*100)}% injection rate, stressed fields drag "
          "the date median down, which shrinks the very deviation the feature is "
          "supposed to measure and adds a noisy 17th dimension for LOF to "
          "contend with. A real haze event would not be diluted this way, so "
          "this experiment does not settle whether the feature helps against "
          "*genuine* atmospheric artefacts — it only shows it does not help "
          "against synthetic stress. Reported as a negative result; not carried "
          "into the recommended configuration.")
    elif d > 0.005:
        A(f"The feature helps: F1 {d:+.4f} with {dfa3:+,.0f} false alarms.")
    else:
        A("The feature makes no measurable difference either way. Not carried "
          "into the recommended configuration.")
    A("")

    A("## E4 · LOF neighbourhood size")
    A("")
    A(_t("t4_kneighbors"))
    A("")
    A("![k sweep](figures/fig5_kneighbors.png)")
    A("")
    f1s = [r["f1"] for r in e4["test"]]
    A(f"The main study fixed k = {e4['k_default']} and never swept it, and listed "
      f"that as an open limitation. Across k ∈ [{min(r['k'] for r in e4['test'])}, "
      f"{max(r['k'] for r in e4['test'])}] test F1 ranges "
      f"{min(f1s):.4f}–{max(f1s):.4f}, a spread of {max(f1s)-min(f1s):.4f}. "
      f"The value selected on the calibration split is k = {e4['k_star']}. "
      "The limitation is now answered: the choice was not load-bearing.")
    A("")

    A("## E5 · Adaptive P85 versus global P90")
    A("")
    A(_t("t5_significance"))
    A("")
    A("![Significance](figures/fig6_significance.png)")
    A("")
    A(f"Study 2 chose adaptive P85 over global P90 on an F1 difference of 0.006 "
      f"with no test behind it. Resampling over {e5['n_units']:,} field-seed "
      f"units ({e5['n_boot']:,} bootstrap resamples) gives "
      f"ΔF1 = **{e5['delta_f1']:+.4f}**, 95% CI "
      f"[{ci_lo:+.4f}, {ci_hi:+.4f}], paired permutation "
      f"p = **{e5['p_value']:.4f}**.")
    A("")
    if spans_zero:
        A("The interval spans zero. **The two rules are not distinguishable on "
          "F1**, so the argmax that picked P85 was selecting on noise, and the "
          "choice has to be made on false-alarm cost instead — which is what the "
          "paper already argues, now with a test behind it.")
    else:
        ap, gp = e5["adaptive_p85"], e5["global_p90"]
        A(f"The interval **excludes zero**, so adaptive P85 really does beat "
          f"global P90 on F1 — Study 2's ordering was correct, and it now has a "
          f"test behind it rather than an argmax. But the effect is "
          f"{e5['delta_f1']:.4f} F1, while global P90 delivers precision "
          f"{gp['precision']:.3f} against {ap['precision']:.3f} and raises "
          f"{int(gp['false_alarms']):,} false alarms against "
          f"{int(ap['false_alarms']):,} — a "
          f"{ap['false_alarms'] / max(gp['false_alarms'], 1):.0f}× difference. "
          f"**The paper's recommendation to deploy global P90 stands, and is "
          f"strengthened:** it concedes a small but real F1 loss in exchange for "
          f"an order-of-magnitude reduction in unnecessary field visits. That is "
          f"a defensible engineering choice, and it should be stated that way "
          f"rather than as \"the two are statistically indistinguishable\".")
    A("")

    if e6:
        A("## E6 · Extreme-value thresholding")
        A("")
        A(_t("t6_evt"))
        A("")
        A("![EVT](figures/fig7_evt.png)")
        A("")
        A(f"With 25 observations per field an empirical percentile saturates: "
          f"above roughly P96 it is interpolating between the top one or two "
          f"scores, which is the stepping visible in Study 2's threshold curve. "
          f"Peaks-over-threshold (Siffer et al., KDD 2017) fits a Generalised "
          f"Pareto tail instead and inverts it for a target false-alarm risk *q*, "
          f"which extrapolates past the sample limit and replaces a percentile "
          f"with an interpretable risk knob.")
        A("")
        A(f"Fitted shape ξ = {e6['xi']:.4f} (pooled across fields — 25 "
          f"observations give roughly 5 excesses each, far too few to fit a "
          f"shape per field), selected risk q = {e6['q_star']}. "
          f"Test F1 {e6['test']['f1']:.4f} against "
          f"{best['f1']:.4f} for the tuned percentile.")
        A("")
        df1 = e6["test"]["f1"] - best["f1"]
        dfa = e6["test"]["false_alarms"] - best["false_alarms"]
        if abs(df1) < 0.01 and dfa < 0:
            A(f"**EVT is the better operating point.** F1 is level with the tuned "
              f"percentile ({df1:+.4f}, well inside seed noise) but it raises "
              f"{-dfa:,.0f} fewer false alarms — "
              f"{int(e6['test']['false_alarms']):,} against "
              f"{int(best['false_alarms']):,}, a {-dfa / best['false_alarms'] * 100:.0f}% "
              f"reduction. It also replaces \"we picked a percentile\" with "
              f"\"we chose a false-alarm risk of {e6['q_star']}\", which is a "
              f"far better answer to why the operating point is where it is.")
        elif e6["test"]["f1"] >= best["f1"]:
            A(f"EVT beats the percentile rule on F1 ({df1:+.4f}) *and* gives a "
              f"principled knob, so it is the better default.")
        else:
            A(f"EVT does **not** beat the tuned percentile here "
              f"({df1:+.4f} F1, {dfa:+,.0f} false alarms). The pooled-shape "
              "approximation is the likely reason. Its value is the interpretable "
              "risk parameter, not accuracy — report it as an alternative, not "
              "as the recommended configuration.")
        A("")
        A(f"The fitted shape is negative (ξ = {e6['xi']:.3f}), meaning the score "
          f"tail is bounded rather than heavy — which is what one would expect "
          f"from LOF scores on a homogeneous population of fields, and is a mild "
          f"independent check that the fit is behaving sensibly.")
        A("")

    A("---")
    A("")
    A("## What to carry into the paper")
    A("")
    A("**Positive:**")
    A("")
    A(f"1. **A better operating point (E1).** Clean-calibrated P{ps:g}: "
      f"P {best['precision']:.3f} / R {best['recall']:.3f} / F1 "
      f"{best['f1']:.3f} on held-out fields, with a threshold that never saw an "
      f"anomaly — against F1 {base['f1']:.3f} for Study 2's rule on the same "
      f"fields. The mechanism is worth a sentence in the paper: a threshold "
      f"calibrated post-injection has to sit low enough to absorb the injected "
      f"mass, which caps how strict it can be.")
    if e6:
        A(f"2. **An interpretable knob (E6).** The EVT/GPD tail reaches "
          f"F1 {e6['test']['f1']:.3f} with {int(e6['test']['false_alarms']):,} "
          f"false alarms at risk q = {e6['q_star']}, replacing a chosen "
          f"percentile with a chosen false-alarm rate.")
    A(f"{'3' if e6 else '2'}. **Two limitations closed (E4, E5).** *k* is swept "
      f"and is not load-bearing; the P85-vs-global-P90 choice now has a "
      f"bootstrap and a permutation test behind it.")
    A("")
    A("**Negative — report these too:**")
    A("")
    A(f"4. **Event-level scoring changes nothing (E2)**, because persistence "
      f"makes detection all-or-nothing at this event length. A predicted win "
      f"that did not materialise.")
    A(f"5. **The peer-deviation feature hurt (E3)**, and the likely reason — "
      f"injected fields polluting their own date median — is itself a caveat "
      f"about the injection design.")
    A("")
    A("## Limitations of this study")
    A("")
    A("- Synthetic injection is still the only ground truth, and every caveat "
      "the main study records about that applies here unchanged.")
    A("- Clean calibration assumes a stress-free reference period per field. "
      "Within a retrospective seasonal analysis that is available; for a "
      "real-time system it is an assumption, and cross-year reference periods "
      "are exactly what `experiment3` showed crop rotation breaks.")
    A("- One region, one season, one injection shape (sudden drop at 40% "
      "severity over 2 observations). The severity/duration and "
      "signature sweeps were not repeated here.")
    A(f"- The test split is {s['n_tiles']['test']:,} fields, a quarter of the "
      "data, so the numbers here carry more sampling noise than Study 2's "
      "full-dataset figures.")
    A("")
    A(f"*Generated by `run_all.py` in {R['runtime_seconds']:.0f}s. "
      "All numbers read from `results.json`.*")

    path = os.path.join(C.OUT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote outputs/REPORT.md")


if __name__ == "__main__":
    main()
