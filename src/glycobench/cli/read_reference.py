from __future__ import annotations

import argparse
from pathlib import Path

from glycobench.ingest.reference_reader import (
    load_manifest,
    read_reference_entry,
)


def iter_entries(manifest: dict, requested_pdb: str | None):
    for entry in manifest["entries"]:
        if entry["status"] == "exclu":
            continue

        if requested_pdb and entry["pdb_id"].upper() != requested_pdb.upper():
            continue

        yield entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract experimental reference glycans from PDB LINK records."
    )
    parser.add_argument(
        "--config",
        default="config/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--pdb",
        default=None,
        help="Optional PDB id, e.g. 6F9T. If omitted, all non-excluded entries are processed.",
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Results root directory.",
    )

    args = parser.parse_args()

    manifest = load_manifest(Path(args.config))
    results_root = Path(args.results)

    total_sites = 0

    for entry in iter_entries(manifest, args.pdb):
        reports = read_reference_entry(entry, manifest, results_root)
        total_sites += len(reports)

        for report in reports:
            site = report["site"]
            root = report["glycan_root"]
            print(
                f"[OK] {report['pdb_id']} "
                f"{site['resname']}{site['resnum']}_{site['chain']} -> "
                f"{root['resname']}{root['resnum']}_{root['chain']} | "
                f"{report['n_glycan_residues']} residues, "
                f"{report['n_glycan_atoms']} atoms"
            )

    print(f"[DONE] {total_sites} site(s) de référence traité(s).")


if __name__ == "__main__":
    main()
