"""
SAFe programme metrics — Manuscript Section 4.5 / Table 10.

Table 10 is an organisational / systems-level comparison
(code duplication, circuit reuse, latency, clinician NPS), not a
model evaluation. This script computes those aggregates from a
CSV you supply — it does not contain Table 10's values as input,
and it does not invent them.

CSV format (see data/safe/safe_template.csv):
  team,period,code_duplication_lines,circuit_reuse_pct,
  end_to_end_latency_ms,clinician_nps
period is "baseline" or "safe". One row per team per period.

Usage:
  python -m src.training.safe_metrics --csv data/safe/safe_template.csv
  (the template contains EXAMPLE rows, labelled as such, so the
  script runs; replace them with your programme data.)
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/safe/safe_template.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/safe_results.json"))
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")
    def agg(period):
        sel = [r for r in rows if r["period"] == period]
        if not sel:
            raise SystemExit(f"No rows with period={period} in {args.csv}")
        return {
            "code_duplication_lines": sum(float(r["code_duplication_lines"]) for r in sel),
            "circuit_reuse_pct_mean": sum(float(r["circuit_reuse_pct"]) for r in sel) / len(sel),
            "end_to_end_latency_ms_mean": sum(float(r["end_to_end_latency_ms"]) for r in sel) / len(sel),
            "clinician_nps_mean": sum(float(r["clinician_nps"]) for r in sel) / len(sel),
            "n_teams": len(sel),
        }
    base, safe = agg("baseline"), agg("safe")
    result = {
        "source_csv": str(args.csv),
        "baseline": base, "safe": safe,
        "code_duplication_change_pct": 100 * (safe["code_duplication_lines"] - base["code_duplication_lines"]) / base["code_duplication_lines"],
        "circuit_reuse_change_pp": safe["circuit_reuse_pct_mean"] - base["circuit_reuse_pct_mean"],
        "latency_change_pct": 100 * (safe["end_to_end_latency_ms_mean"] - base["end_to_end_latency_ms_mean"]) / base["end_to_end_latency_ms_mean"],
        "nps_change": safe["clinician_nps_mean"] - base["clinician_nps_mean"],
        "note": "Demo output computed from the template CSV (data/safe/safe_template.csv). The template rows encode Table 10's reported baseline and SAFe values as a worked example, so the deltas here follow by construction. Supply your own programme CSV for an independent evaluation.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
