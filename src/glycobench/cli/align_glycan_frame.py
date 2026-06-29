from __future__ import annotations

from pathlib import Path
import argparse
import csv

from glycobench.mapping.glycan_frame_alignment import align_site_in_glycan_frame


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align glycan ensembles in a local glycan frame using mapped monosaccharide centroids."
    )

    parser.add_argument("--manifest", default="results/analysis_manifest.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--pdb", default=None)
    parser.add_argument("--site", default=None)
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also process reference+GlycoShield secondary cases.",
    )
    parser.add_argument(
        "--min-centroids",
        type=int,
        default=2,
        help="Minimum number of mapped monosaccharide centroids required. Use 3 for strict 3D alignment; default 2 keeps disaccharides with a fragile axis-only alignment.",
    )

    args = parser.parse_args()

    rows = load_rows(Path(args.manifest))
    selected: list[dict[str, str]] = []

    for row in rows:
        if args.pdb and row["pdb_id"].upper() != args.pdb.upper():
            continue

        if args.site and row["site"] != args.site:
            continue

        if row["use_for_triple_benchmark"] == "True":
            selected.append(row)
            continue

        if args.include_secondary and row["use_for_glycoshield_secondary"] == "True":
            selected.append(row)
            continue

    reports = []
    for row in selected:
        include_glycoshape = bool(row.get("glycoshape_glycan_ensemble_pdb"))
        report = align_site_in_glycan_frame(
            row=row,
            results_root=Path(args.results_root),
            include_glycoshape=include_glycoshape,
            include_glycoshield=bool(row.get("glycoshield_glycan_ensemble_pdb")),
            min_centroids=args.min_centroids,
        )
        reports.append(report)

        tool_summaries = []
        for tool_name, tool_report in sorted(report["tools"].items()):
            status = tool_report.get("status")
            n_models = tool_report.get("n_models_aligned", 0)
            n_input = tool_report.get("n_models_input", 0)
            mean_rmsd = tool_report.get("mean_centroid_rmsd")
            n_mapped = tool_report.get("n_mapped_residues")
            n_ref = tool_report.get("n_reference_residues")
            tool_summaries.append(
                f"{tool_name}: {status}, models {n_models}/{n_input}, mapping {n_mapped}/{n_ref}, mean RMSD={mean_rmsd}"
            )

        print(f'[{report["status"].upper()}] {report["pdb_id"]} {report["site"]}')
        for summary in tool_summaries:
            print(f"  {summary}")
        print(
            "  report: "
            f"{Path(args.results_root) / report['pdb_id'] / 'aligned' / report['site'] / 'glycan_frame' / 'glycan_frame_alignment_report.json'}"
        )

        warnings = report.get("warnings", [])
        tool_warning_count = sum(len(tool_report.get("warnings", [])) for tool_report in report["tools"].values())
        if warnings or tool_warning_count:
            print(f"  warnings: site={len(warnings)}, tools={tool_warning_count}")

    print(f"[DONE] {len(reports)} site(s) aligné(s) dans le repère glycane.")


if __name__ == "__main__":
    main()
