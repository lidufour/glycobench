from __future__ import annotations

from pathlib import Path
import argparse
import csv

from glycobench.mapping.protein_frame_alignment import align_site_in_protein_frame


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align glycan ensembles in the experimental protein frame."
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
        report = align_site_in_protein_frame(
            row=row,
            results_root=Path(args.results_root),
        )
        reports.append(report)

        tools = ", ".join(sorted(report["tools"]))

        print(
            f'[OK] {report["pdb_id"]} {report["site"]} | '
            f'tools: {tools}'
        )

        for tool_name, tool_report in sorted(report["tools"].items()):
            print(
                f'  {tool_name}: '
                f'{tool_report["n_common_ca"]} CA, '
                f'RMSD={tool_report["protein_alignment_rmsd"]} Å'
            )

    print(f"[DONE] {len(reports)} site(s) aligné(s) dans le repère protéine.")


if __name__ == "__main__":
    main()
