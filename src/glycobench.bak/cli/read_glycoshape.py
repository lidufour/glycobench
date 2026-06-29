from __future__ import annotations

from pathlib import Path
import argparse

from glycobench.ingest.glycoshape_reader import read_all_glycoshape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract site-specific glycan ensembles from GlycoShape/Re-Glyco outputs."
    )

    parser.add_argument(
        "--manifest",
        default="config/dataset.yaml",
        help="Path to dataset YAML file.",
    )

    parser.add_argument(
        "--pdb",
        default=None,
        help="Optional PDB ID to process only one entry.",
    )

    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for extracted results.",
    )

    args = parser.parse_args()

    bundles = read_all_glycoshape(
        manifest_path=Path(args.manifest),
        pdb_id=args.pdb,
        output_root=Path(args.output_root),
    )

    for bundle in bundles:
        site = bundle["site"]
        anchor = bundle["anchor_residue_nearest_site"]

        site_label = f'{site["resname"]}{site["resnum"]}_{site["chain"]}'
        anchor_label = f'{anchor["resname"]}{anchor["resnum"]}_{anchor["chain"]}'

        if anchor.get("icode"):
            anchor_label += f'.{anchor["icode"]}'

        print(
            f'[OK] {bundle["pdb_id"]} {site_label} -> {anchor_label} | '
            f'{bundle["n_models"]} models, '
            f'{bundle["n_glycan_residues"]} residues, '
            f'{bundle["n_glycan_atoms_per_model"]} atoms/model, '
            f'd={bundle["anchor_distance_angstrom"]} Å'
        )

    print(f"[DONE] {len(bundles)} site(s) GlycoShape traité(s).")


if __name__ == "__main__":
    main()
