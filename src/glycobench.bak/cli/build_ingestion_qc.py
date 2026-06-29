from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

from glycobench.ingest.reference_reader import load_manifest, site_from_manifest


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open() as handle:
        return json.load(handle)


def site_label_from_dict(site: dict) -> str:
    label = f'{site["resname"]}{site["resnum"]}_{site["chain"]}'

    if site.get("icode"):
        label += f'.{site["icode"]}'

    return label


def get_n_residues(bundle: dict | None) -> int | str:
    if bundle is None:
        return ""

    if "n_glycan_residues" in bundle:
        return bundle["n_glycan_residues"]

    if "glycan_residues" in bundle:
        return len(bundle["glycan_residues"])

    return ""


def get_n_atoms(bundle: dict | None) -> int | str:
    if bundle is None:
        return ""

    return bundle.get("n_glycan_atoms_per_model", bundle.get("n_glycan_atoms", ""))


def get_n_models(bundle: dict | None) -> int | str:
    if bundle is None:
        return ""

    return bundle.get("n_models", 1)


def build_qc_table(
    manifest_path: Path,
    results_root: Path,
    output_csv: Path,
) -> list[dict]:
    manifest = load_manifest(manifest_path)
    rows: list[dict] = []

    for entry in manifest["entries"]:
        pdb_id = entry["pdb_id"]

        for site_raw in entry["sites"]:
            site = site_from_manifest(site_raw)
            site_label = site.label()

            reference_bundle = read_json(
                results_root / pdb_id / "reference" / site_label / "reference_bundle.json"
            )

            glycoshield_bundle = read_json(
                results_root / pdb_id / "glycoshield" / site_label / "glycoshield_bundle.json"
            )

            glycoshape_bundle = read_json(
                results_root / pdb_id / "glycoshape" / site_label / "glycoshape_bundle.json"
            )

            reference_present = reference_bundle is not None
            glycoshield_present = glycoshield_bundle is not None
            glycoshape_present = glycoshape_bundle is not None

            ref_n_res = get_n_residues(reference_bundle)
            gs_n_res = get_n_residues(glycoshield_bundle)
            rg_n_res = get_n_residues(glycoshape_bundle)

            if reference_present and glycoshield_present and glycoshape_present:
                comparison_status = "triple"
            elif reference_present and glycoshield_present and not glycoshape_present:
                comparison_status = "reference_glycoshield"
            elif reference_present and glycoshape_present and not glycoshield_present:
                comparison_status = "reference_glycoshape"
            elif reference_present:
                comparison_status = "reference_only"
            else:
                comparison_status = "missing_reference"

            row = {
                "pdb_id": pdb_id,
                "site": site_label,
                "manifest_status": entry.get("status", ""),
                "glycan_class": entry.get("glycan_class", ""),
                "glycan_name": entry.get("glycan_name", ""),
                "glytoucan": entry.get("glytoucan", ""),
                "reference_present": reference_present,
                "glycoshield_present": glycoshield_present,
                "glycoshape_present": glycoshape_present,
                "comparison_status": comparison_status,
                "reference_n_residues": ref_n_res,
                "glycoshield_n_residues": gs_n_res,
                "glycoshape_n_residues": rg_n_res,
                "glycoshield_n_models": get_n_models(glycoshield_bundle),
                "glycoshape_n_models": get_n_models(glycoshape_bundle),
                "reference_n_atoms": get_n_atoms(reference_bundle),
                "glycoshield_n_atoms_per_model": get_n_atoms(glycoshield_bundle),
                "glycoshape_n_atoms_per_model": get_n_atoms(glycoshape_bundle),
                "glycoshield_anchor_distance": (
                    glycoshield_bundle.get("anchor_distance_angstrom", "")
                    if glycoshield_bundle
                    else ""
                ),
                "glycoshape_anchor_distance": (
                    glycoshape_bundle.get("anchor_distance_angstrom", "")
                    if glycoshape_bundle
                    else ""
                ),
                "residue_count_ref_vs_glycoshield": (
                    ref_n_res == gs_n_res
                    if reference_present and glycoshield_present
                    else ""
                ),
                "residue_count_ref_vs_glycoshape": (
                    ref_n_res == rg_n_res
                    if reference_present and glycoshape_present
                    else ""
                ),
                "note": "",
            }

            if glycoshield_bundle and glycoshield_bundle.get("n_models") == 1:
                row["note"] = "GlycoShield has only 1 model"

            if reference_present and not glycoshield_present:
                row["note"] = "GlycoShield missing"

            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ingestion QC table for reference, GlycoShield and GlycoShape outputs."
    )

    parser.add_argument("--manifest", default="config/dataset.yaml")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/ingestion_qc.csv")

    args = parser.parse_args()

    rows = build_qc_table(
        manifest_path=Path(args.manifest),
        results_root=Path(args.results_root),
        output_csv=Path(args.output),
    )

    print(f"[DONE] QC écrit dans {args.output}")
    print(f"Total sites: {len(rows)}")

    for key in [
        "reference_present",
        "glycoshield_present",
        "glycoshape_present",
    ]:
        print(f"{key}: {sum(bool(row[key]) for row in rows)}")

    print()
    print("comparison_status:")
    counts = {}

    for row in rows:
        counts[row["comparison_status"]] = counts.get(row["comparison_status"], 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    print()
    print("Rows with notes:")

    for row in rows:
        if row["note"]:
            print(f'  {row["pdb_id"]} {row["site"]}: {row["note"]}')


if __name__ == "__main__":
    main()
