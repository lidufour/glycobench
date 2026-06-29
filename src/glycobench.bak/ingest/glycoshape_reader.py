from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json

from glycobench.ingest.reference_reader import (
    ResidueKey,
    load_manifest,
    site_from_manifest,
)

from glycobench.ingest.glycoshield_reader import (
    ModelAtom,
    assign_components_to_sites,
    build_glycan_components,
    is_glycan_residue,
    parse_multimodel_pdb,
    uniquify_repeated_residue_blocks,
)


def resolve_glycoshape_job_path(entry: dict, manifest: dict) -> Path:
    root = Path(manifest["roots"]["glycoshape"])
    job_dir = entry["glycoshape"]["job_dir"]
    return root / job_dir


def read_glycoshape_pdb_lines(entry: dict, manifest: dict) -> tuple[list[str], Path]:
    job_path = resolve_glycoshape_job_path(entry, manifest)
    pdb_path = job_path / "all.pdb"

    if not pdb_path.exists():
        raise FileNotFoundError(f"Fichier GlycoShape absent: {pdb_path}")

    return pdb_path.read_text().splitlines(), pdb_path


def read_ensemble_stats(job_path: Path) -> list[dict[str, str]]:
    stats_path = job_path / "ensemble_stats.csv"

    if not stats_path.exists():
        return []

    with stats_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_glycan_ensemble_pdb(
    models: dict[int, list[ModelAtom]],
    glycan_residues: set[ResidueKey],
    output_pdb: Path,
) -> int:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    n_atoms_first_model = 0

    with output_pdb.open("w") as handle:
        for model_id in sorted(models):
            selected_atoms = [
                atom for atom in models[model_id]
                if atom.residue in glycan_residues
            ]

            if model_id == sorted(models)[0]:
                n_atoms_first_model = len(selected_atoms)

            handle.write(f"MODEL     {model_id:4d}\n")

            for atom in selected_atoms:
                handle.write(atom.line.rstrip() + "\n")

            handle.write("ENDMDL\n")

        handle.write("END\n")

    return n_atoms_first_model


def read_glycoshape_entry(
    entry: dict,
    manifest: dict,
    output_root: Path = Path("results"),
) -> list[dict]:
    pdb_id = entry["pdb_id"]

    if not entry.get("glycoshape", {}).get("available", False):
        return []

    job_path = resolve_glycoshape_job_path(entry, manifest)
    lines, source_pdb = read_glycoshape_pdb_lines(entry, manifest)

    models = parse_multimodel_pdb(lines)
    models = uniquify_repeated_residue_blocks(models)

    if not models:
        raise ValueError(f"Aucun modèle lu dans {source_pdb}")

    first_model_id = sorted(models)[0]
    first_model_atoms = models[first_model_id]

    sites = [site_from_manifest(site) for site in entry["sites"]]

    components = build_glycan_components(first_model_atoms)

    if not components:
        raise ValueError(f"Aucun composant glycanique détecté dans {source_pdb}")

    assignments = assign_components_to_sites(
        first_model_atoms,
        sites,
        components,
        max_anchor_distance=7.0,
    )

    ensemble_stats = read_ensemble_stats(job_path)
    bundles: list[dict] = []

    for site, (glycan_residues, anchor_residue, anchor_distance) in assignments.items():
        site_dir = output_root / pdb_id / "glycoshape" / site.label()
        output_pdb = site_dir / "glycan_ensemble.pdb"

        n_atoms_per_model = write_glycan_ensemble_pdb(
            models=models,
            glycan_residues=glycan_residues,
            output_pdb=output_pdb,
        )

        glycan_chains = sorted({residue.chain for residue in glycan_residues})

        site_ensemble_stats = [
            row for row in ensemble_stats
            if row.get("Chain") in glycan_chains
        ]

        bundle = {
            "pdb_id": pdb_id,
            "site": asdict(site),
            "source_pdb": str(source_pdb),
            "job_dir": str(job_path),
            "n_models": len(models),
            "n_glycan_residues": len(glycan_residues),
            "n_glycan_atoms_per_model": n_atoms_per_model,
            "anchor_residue_nearest_site": asdict(anchor_residue),
            "anchor_distance_angstrom": round(anchor_distance, 3),
            "glycan_residues": [
                asdict(residue)
                for residue in sorted(
                    glycan_residues,
                    key=lambda r: (r.chain, r.resnum, r.icode, r.resname),
                )
            ],
            "glycan_chains": glycan_chains,
            "ensemble_stats_csv": str(job_path / "ensemble_stats.csv"),
            "job_meta_json": str(job_path / "job_meta.json"),
            "ensemble_stats_rows": site_ensemble_stats,
            "output_glycan_ensemble_pdb": str(output_pdb),
        }

        site_dir.mkdir(parents=True, exist_ok=True)

        with (site_dir / "glycoshape_bundle.json").open("w") as handle:
            json.dump(bundle, handle, indent=2)

        bundles.append(bundle)

    return bundles


def read_all_glycoshape(
    manifest_path: Path = Path("config/dataset.yaml"),
    pdb_id: str | None = None,
    output_root: Path = Path("results"),
) -> list[dict]:
    manifest = load_manifest(manifest_path)

    entries = manifest["entries"]

    if pdb_id is not None:
        entries = [entry for entry in entries if entry["pdb_id"].upper() == pdb_id.upper()]

    all_bundles: list[dict] = []

    for entry in entries:
        bundles = read_glycoshape_entry(
            entry=entry,
            manifest=manifest,
            output_root=output_root,
        )
        all_bundles.extend(bundles)

    return all_bundles
