from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open() as handle:
        return json.load(handle)


def as_bool(value: str) -> bool:
    return str(value).lower() == "true"


def site_paths(results_root: Path, pdb_id: str, site: str) -> dict[str, Path]:
    base = results_root / pdb_id

    return {
        "reference_bundle": base / "reference" / site / "reference_bundle.json",
        "glycoshield_bundle": base / "glycoshield" / site / "glycoshield_bundle.json",
        "glycoshape_bundle": base / "glycoshape" / site / "glycoshape_bundle.json",
    }


def classify_site(qc_row: dict[str, str]) -> tuple[str, str]:
    status = qc_row["comparison_status"]

    ref_ok = as_bool(qc_row["reference_present"])
    gs_ok = as_bool(qc_row["glycoshield_present"])
    rg_ok = as_bool(qc_row["glycoshape_present"])

    ref_gs_same = qc_row["residue_count_ref_vs_glycoshield"] != "False"
    ref_rg_same = qc_row["residue_count_ref_vs_glycoshape"] != "False"

    gs_models = qc_row["glycoshield_n_models"]
    gs_has_enough_models = gs_models not in {"", "1"}

    if qc_row["manifest_status"] == "exclu":
        return "excluded", "site marked as excluded in dataset.yaml"

    if not ref_ok:
        return "missing_reference", "reference glycan was not extracted"

    if status == "triple" and ref_gs_same and ref_rg_same:
        return "triple_core", "reference, GlycoShield and GlycoShape are available"

    if status == "triple":
        return "triple_needs_review", "triple case but residue counts do not fully match"

    if ref_ok and gs_ok and not rg_ok and ref_gs_same and gs_has_enough_models:
        return "reference_glycoshield_secondary", "reference and GlycoShield are available"

    if ref_ok and gs_ok and not rg_ok:
        return "reference_glycoshield_needs_review", "GlycoShield is available but QC is fragile"

    if ref_ok and rg_ok and not gs_ok:
        return "reference_glycoshape_secondary", "reference and GlycoShape are available"

    if ref_ok and not gs_ok and not rg_ok:
        return "reference_only", "only the experimental reference is available"

    return "unclassified", "unclassified QC pattern"


def build_analysis_manifest(
    qc_csv: Path,
    results_root: Path,
    output_csv: Path,
) -> list[dict[str, str]]:
    qc_rows = read_csv(qc_csv)
    rows: list[dict[str, str]] = []

    for qc in qc_rows:
        pdb_id = qc["pdb_id"]
        site = qc["site"]

        paths = site_paths(results_root, pdb_id, site)

        reference_bundle = read_json(paths["reference_bundle"])
        glycoshield_bundle = read_json(paths["glycoshield_bundle"])
        glycoshape_bundle = read_json(paths["glycoshape_bundle"])

        analysis_group, skip_or_note = classify_site(qc)

        reference_glycan_pdb = ""
        reference_apo_pdb = ""
        reference_glyco_pdb = ""

        if reference_bundle:
            reference_glycan_pdb = reference_bundle.get("output_glycan_pdb", "")
            reference_apo_pdb = reference_bundle.get("reference_apo_pdb", "")
            reference_glyco_pdb = reference_bundle.get("reference_pdb", "")

        glycoshield_glycan_pdb = ""
        if glycoshield_bundle:
            glycoshield_glycan_pdb = glycoshield_bundle.get(
                "output_glycan_ensemble_pdb", ""
            )

        glycoshape_glycan_pdb = ""
        glycoshape_stats_csv = ""
        glycoshape_job_meta_json = ""

        if glycoshape_bundle:
            glycoshape_glycan_pdb = glycoshape_bundle.get(
                "output_glycan_ensemble_pdb", ""
            )
            glycoshape_stats_csv = glycoshape_bundle.get("ensemble_stats_csv", "")
            glycoshape_job_meta_json = glycoshape_bundle.get("job_meta_json", "")

        use_for_triple_benchmark = analysis_group == "triple_core"
        use_for_glycoshield_secondary = (
            analysis_group == "reference_glycoshield_secondary"
        )

        use_for_alignment = analysis_group in {
            "triple_core",
            "reference_glycoshield_secondary",
            "reference_glycoshape_secondary",
        }

        row = {
            "pdb_id": pdb_id,
            "site": site,
            "analysis_group": analysis_group,
            "analysis_note": skip_or_note,
            "manifest_status": qc["manifest_status"],
            "glycan_class": qc["glycan_class"],
            "glycan_name": qc["glycan_name"],
            "glytoucan": qc["glytoucan"],
            "use_for_alignment": use_for_alignment,
            "use_for_triple_benchmark": use_for_triple_benchmark,
            "use_for_glycoshield_secondary": use_for_glycoshield_secondary,
            "reference_n_residues": qc["reference_n_residues"],
            "glycoshield_n_residues": qc["glycoshield_n_residues"],
            "glycoshape_n_residues": qc["glycoshape_n_residues"],
            "glycoshield_n_models": qc["glycoshield_n_models"],
            "glycoshape_n_models": qc["glycoshape_n_models"],
            "qc_note": qc["note"],
            "reference_glyco_pdb": reference_glyco_pdb,
            "reference_apo_pdb": reference_apo_pdb,
            "reference_glycan_pdb": reference_glycan_pdb,
            "glycoshield_glycan_ensemble_pdb": glycoshield_glycan_pdb,
            "glycoshape_glycan_ensemble_pdb": glycoshape_glycan_pdb,
            "glycoshape_stats_csv": glycoshape_stats_csv,
            "glycoshape_job_meta_json": glycoshape_job_meta_json,
        }

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analysis manifest from ingestion QC table."
    )

    parser.add_argument("--qc-csv", default="results/ingestion_qc.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/analysis_manifest.csv")

    args = parser.parse_args()

    rows = build_analysis_manifest(
        qc_csv=Path(args.qc_csv),
        results_root=Path(args.results_root),
        output_csv=Path(args.output),
    )

    print(f"[DONE] Analysis manifest écrit dans {args.output}")
    print(f"Total rows: {len(rows)}")

    counts: dict[str, int] = {}

    for row in rows:
        counts[row["analysis_group"]] = counts.get(row["analysis_group"], 0) + 1

    print()
    print("analysis_group:")

    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")

    print()
    print("usable:")
    print(
        "  use_for_triple_benchmark:",
        sum(str(row["use_for_triple_benchmark"]) == "True" for row in rows),
    )
    print(
        "  use_for_glycoshield_secondary:",
        sum(str(row["use_for_glycoshield_secondary"]) == "True" for row in rows),
    )
    print(
        "  use_for_alignment:",
        sum(str(row["use_for_alignment"]) == "True" for row in rows),
    )


if __name__ == "__main__":
    main()
