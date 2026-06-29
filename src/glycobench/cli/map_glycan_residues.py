from __future__ import annotations

from pathlib import Path
import argparse
import csv

from glycobench.mapping.glycan_residue_mapping import (
    build_site_mapping_report,
    write_site_mapping_report,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build residue-level glycan mappings between reference, GlycoShield and GlycoShape."
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
        "--bond-cutoff",
        type=float,
        default=1.85,
        help="Maximum heavy-atom distance used to infer inter-residue covalent bonds.",
    )

    args = parser.parse_args()
    rows = load_rows(Path(args.manifest))

    selected = []

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
        include_glycoshape = row.get("use_for_triple_benchmark") == "True"
        include_glycoshield = row.get("glycoshield_glycan_ensemble_pdb", "") not in {"", "nan", "NaN"}

        report = build_site_mapping_report(
            pdb_id=row["pdb_id"],
            site=row["site"],
            results_root=Path(args.results_root),
            include_glycoshape=include_glycoshape,
            include_glycoshield=include_glycoshield,
            bond_cutoff=args.bond_cutoff,
        )
        report_path = write_site_mapping_report(report, results_root=Path(args.results_root))
        reports.append(report)

        tool_summaries = []
        for tool_name, tool_report in sorted(report["tools"].items()):
            tool_summaries.append(
                f'{tool_name}: {tool_report["status"]}, '
                f'{tool_report["n_mapped_residues"]}/{tool_report["n_reference_residues"]} residues'
            )

        summary = "; ".join(tool_summaries) if tool_summaries else "no tools"
        print(f'[OK] {row["pdb_id"]} {row["site"]} | {summary}')
        print(f"  report: {report_path}")

        site_warnings = list(report.get("warnings", []))
        for tool_report in report["tools"].values():
            site_warnings.extend(tool_report.get("warnings", []))

        if site_warnings:
            print(f"  warnings: {len(site_warnings)}")

    print(f"[DONE] {len(reports)} site(s) traité(s) pour le mapping des résidus glycaniques.")


if __name__ == "__main__":
    main()
