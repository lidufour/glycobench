from __future__ import annotations

from pathlib import Path
import argparse
import json

from glycobench.qc.glycan_frame_qc import build_qc_rows, summarize_qc_rows, write_qc_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a global QC table for glycan-frame alignment RMSDs."
    )
    parser.add_argument("--manifest", default="results/analysis_manifest.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/glycan_frame_qc.csv")
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also include reference+GlycoShield secondary sites.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/glycan_frame_qc_summary.json",
        help="Path for a compact JSON summary. Use an empty value to skip.",
    )

    args = parser.parse_args()

    rows = build_qc_rows(
        manifest_path=Path(args.manifest),
        results_root=Path(args.results_root),
        include_secondary=args.include_secondary,
    )
    output_path = write_qc_csv(rows, args.output)
    summary = summarize_qc_rows(rows)

    print(f"[OK] Glycan-frame QC table written: {output_path}")
    print(f"[OK] {summary['n_rows']} row(s)")
    print("[SUMMARY] scope")
    for key, value in sorted(summary["by_scope"].items()):
        print(f"  {key}: {value}")
    print("[SUMMARY] tool status")
    for key, value in sorted(summary["by_tool_status"].items()):
        print(f"  {key}: {value}")
    print("[SUMMARY] recommended use")
    for key, value in sorted(summary["by_recommended_use"].items()):
        print(f"  {key}: {value}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[OK] Summary JSON written: {summary_path}")


if __name__ == "__main__":
    main()
