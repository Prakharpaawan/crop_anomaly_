"""
run_all.py — the whole unified study, end to end.

  cd hybrid_pipeline_experiments3
  C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe run_all.py

First run takes roughly 30-45 minutes: it injects the full severity/duration
grid and both alternative signatures, and trains the autoencoder. Later runs
reuse the cached injections and are much faster.
"""

import json
import os
import time

import core as C
import study as S
import make_figures
import make_report

ORDER = ["a_operating_point", "b_models", "c_baselines", "d_persistence",
         "e_limits", "f_signatures", "g_kneighbors", "h_protocol",
         "i_features"]


def row(m, keys):
    out = []
    for k in keys:
        v = m.get(k)
        if v is None:
            out.append("—")
        elif k == "false_alarms":
            out.append(f"{int(round(v)):,}")
        else:
            sd = m.get(k + "_sd")
            out.append(f"{v:.4f}±{sd:.3f}" if sd else f"{v:.4f}")
    return out


def write_table(name, header, rows):
    md = "| " + " | ".join(header) + " |\n"
    md += "|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(str(x) for x in r) + " |\n"
    with open(os.path.join(C.TABLE_DIR, f"{name}.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(f"  wrote tables/{name}.md")


MAIN = ["precision", "recall", "f1", "pr_auc", "false_alarms"]
HDR = ["Precision", "Recall", "F1", "PR-AUC", "False alarms"]


def main():
    t0 = time.time()
    C.ensure_dirs()
    R = {}

    print("=" * 68)
    print("hybrid_pipeline_experiments3 — unified study")
    print("=" * 68)
    b = S.Bench()
    R["setup"] = {
        "n_tiles": {k: len(v) for k, v in b.splits.items()},
        "n_rows": {k: int(m.sum()) for k, m in b.mask.items()},
        "seeds": C.SEEDS, "injection_ratio": C.INJECTION_RATIO,
        "severity": C.SEVERITY, "duration": C.DURATION,
        "min_consecutive": C.MIN_CONSECUTIVE, "split_seed": C.SPLIT_SEED,
        "legacy_pctl": C.LEGACY_PCTL, "k_default": C.LOF["n_neighbors"],
        "n_features": len(C.FEATURE_COLUMNS),
    }
    for k in ("fit", "cal", "test"):
        print(f"  {k:5s}: {len(b.splits[k]):>5,} fields  "
              f"{int(b.mask[k].sum()):>7,} observations")

    print("\n[A] operating point")
    R["a"] = S.a_operating_point(b)
    ch = R["a"]["chosen"]
    print(f"  p*=P{R['a']['p_star']:g}  q*={R['a']['q_star']:g}  "
          f"xi={R['a']['xi']:.4f}  chosen={ch['kind']}")
    for k, v in R["a"]["results"].items():
        print(f"    {k:48s} F1={v['f1']:.4f} FA={int(v['false_alarms']):,}")

    print("\n[B] detector comparison")
    R["b"] = S.b_models(b, ch)
    for k, v in R["b"].items():
        print(f"    {k:20s} F1={v['f1']:.4f} FA={int(v['false_alarms']):,}")

    print("\n[C] rule-based baselines")
    R["c"] = S.c_baselines(b, ch)
    for k, v in R["c"].items():
        print(f"    {k:26s} F1={v['f1']:.4f} FA={int(v['false_alarms']):,}")

    print("\n[D] persistence")
    R["d"] = S.d_persistence(b, ch)
    for k, v in R["d"].items():
        print(f"    {k:26s} F1={v['f1']:.4f} FA={int(v['false_alarms']):,}")

    print("\n[E] detection limits")
    R["e"] = S.e_limits(b, ch)

    print("\n[F] anomaly signatures")
    R["f"] = S.f_signatures(b, ch)
    for k, v in R["f"].items():
        print(f"    {k:16s} F1={v['f1']:.4f}")

    print("\n[G] LOF neighbourhood size")
    R["g"] = S.g_kneighbors(b, ch)
    print(f"    k*={R['g']['k_star']}")

    print("\n[H] protocol effect")
    R["h"] = S.h_protocol(b)
    for k, v in R["h"].items():
        print(f"    {k:46s} F1={v['f1']:.4f}")

    print("\n[I] feature ablation")
    R["i"] = S.i_features(b, ch)
    for k, v in R["i"].items():
        print(f"    {k:28s} F1={v['f1']:.4f}")

    R["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(C.OUT_DIR, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print("\nWrote outputs/results.json")

    # ── tables ───────────────────────────────────────────────────────
    print("\nWriting tables...")
    write_table("t1_operating_point", ["Threshold rule"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["a"]["results"].items()])
    write_table("t2_detectors", ["Detector"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["b"].items()])
    write_table("t3_baselines", ["Method"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["c"].items()])
    write_table("t4_persistence", ["Persistence rule"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["d"].items()])
    write_table("t5_limits",
                ["Severity", "Duration"] + HDR,
                [[f"{int(m['severity']*100)}%", f"{m['duration']} obs"]
                 + row(m, MAIN) for m in R["e"]])
    write_table("t6_signatures", ["Stress signature"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["f"].items()])
    write_table("t7_kneighbors", ["k"] + HDR,
                [[m["k"]] + row(m, MAIN) for m in R["g"]["test"]])
    write_table("t8_protocol", ["Protocol"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["h"].items()])
    write_table("t9_features", ["Feature set"] + HDR,
                [[k] + row(v, MAIN) for k, v in R["i"].items()])

    # ── illustrative material (worked example, imagery, curves, tests) ──
    print("\n[extras] worked example, imagery, curves, significance")
    import extras
    R["extras"] = extras.compute(b, ch)
    with open(os.path.join(C.OUT_DIR, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(R, f, indent=2)

    print()
    make_figures.main()
    print()
    make_report.main()
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
