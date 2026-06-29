from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import zipfile

import numpy as np


@dataclass(frozen=True)
class Atom:
    model: int
    line: str
    record: str
    atom_name: str
    resname: str
    chain: str
    resnum: int
    icode: str
    altloc: str
    x: float
    y: float
    z: float


def read_text_maybe_zip(source: str | Path) -> str:
    source = str(source)

    if "!" not in source:
        return Path(source).read_text()

    zip_path, internal_path = source.split("!", 1)

    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(internal_path).decode()


def parse_atom_line(line: str, model: int) -> Atom | None:
    record = line[0:6].strip()

    if record not in {"ATOM", "HETATM"}:
        return None

    try:
        return Atom(
            model=model,
            line=line.rstrip("\n"),
            record=record,
            atom_name=line[12:16].strip(),
            altloc=line[16].strip(),
            resname=line[17:20].strip(),
            chain=line[21].strip(),
            resnum=int(line[22:26]),
            icode=line[26].strip(),
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
        )
    except Exception:
        return None


def parse_models(text: str) -> dict[int, list[Atom]]:
    models: dict[int, list[Atom]] = {}
    current_model = 1
    saw_model_record = False

    for line in text.splitlines():
        if line.startswith("MODEL"):
            saw_model_record = True
            try:
                current_model = int(line[10:14])
            except ValueError:
                current_model = len(models) + 1

            models.setdefault(current_model, [])
            continue

        if line.startswith("ENDMDL"):
            continue

        atom = parse_atom_line(line, current_model)

        if atom is not None:
            models.setdefault(current_model, []).append(atom)

    if not saw_model_record and 1 not in models:
        models[1] = []

    return models


def ca_atoms_by_key(atoms: list[Atom]) -> dict[tuple[str, int, str], Atom]:
    out: dict[tuple[str, int, str], Atom] = {}

    for atom in atoms:
        if atom.record != "ATOM":
            continue

        if atom.atom_name != "CA":
            continue

        if atom.altloc not in {"", "A"}:
            continue

        key = (atom.chain, atom.resnum, atom.icode)

        if key not in out:
            out[key] = atom

    return out


def kabsch_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)

    source_centered = source_xyz - source_center
    target_centered = target_xyz - target_center

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)

    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    transformed = source_centered @ rotation + target_center
    diff = transformed - target_xyz
    rmsd = float(np.sqrt((diff * diff).sum() / len(source_xyz)))

    translation = target_center - source_center @ rotation

    return rotation, translation, rmsd


def build_protein_alignment(
    reference_protein_pdb: str | Path,
    source_full_pdb: str | Path,
) -> dict:
    reference_text = read_text_maybe_zip(reference_protein_pdb)
    source_text = read_text_maybe_zip(source_full_pdb)

    reference_models = parse_models(reference_text)
    source_models = parse_models(source_text)

    reference_atoms = reference_models[sorted(reference_models)[0]]
    source_atoms = source_models[sorted(source_models)[0]]

    reference_ca = ca_atoms_by_key(reference_atoms)
    source_ca = ca_atoms_by_key(source_atoms)

    common_keys = sorted(set(reference_ca) & set(source_ca))

    if len(common_keys) < 3:
        raise ValueError(
            f"Pas assez de CA communs pour aligner {source_full_pdb} "
            f"sur {reference_protein_pdb}: {len(common_keys)} CA communs."
        )

    reference_xyz = np.array(
        [[reference_ca[key].x, reference_ca[key].y, reference_ca[key].z] for key in common_keys],
        dtype=float,
    )

    source_xyz = np.array(
        [[source_ca[key].x, source_ca[key].y, source_ca[key].z] for key in common_keys],
        dtype=float,
    )

    rotation, translation, rmsd = kabsch_transform(source_xyz, reference_xyz)

    return {
        "rotation": rotation,
        "translation": translation,
        "rmsd": rmsd,
        "n_common_ca": len(common_keys),
        "common_ca_keys": common_keys,
    }


def transform_xyz(
    x: float,
    y: float,
    z: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float, float]:
    xyz = np.array([x, y, z], dtype=float)
    transformed = xyz @ rotation + translation
    return float(transformed[0]), float(transformed[1]), float(transformed[2])


def replace_xyz_in_pdb_line(
    line: str,
    x: float,
    y: float,
    z: float,
) -> str:
    padded = line.rstrip("\n")

    if len(padded) < 54:
        padded = padded.ljust(54)

    return f"{padded[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{padded[54:]}"


def transform_pdb_file(
    input_pdb: str | Path,
    output_pdb: str | Path,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    input_text = read_text_maybe_zip(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    current_model = 1

    with output_pdb.open("w") as handle:
        for line in input_text.splitlines():
            if line.startswith("MODEL"):
                try:
                    current_model = int(line[10:14])
                except ValueError:
                    current_model += 1

                handle.write(line.rstrip() + "\n")
                continue

            atom = parse_atom_line(line, current_model)

            if atom is None:
                handle.write(line.rstrip() + "\n")
                continue

            x, y, z = transform_xyz(
                atom.x,
                atom.y,
                atom.z,
                rotation,
                translation,
            )

            handle.write(replace_xyz_in_pdb_line(line, x, y, z) + "\n")


def copy_reference_glycan(
    input_pdb: str | Path,
    output_pdb: str | Path,
) -> None:
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_pdb, output_pdb)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def align_site_in_protein_frame(
    row: dict[str, str],
    results_root: Path = Path("results"),
) -> dict:
    pdb_id = row["pdb_id"]
    site = row["site"]

    output_dir = results_root / pdb_id / "aligned" / site / "protein_frame"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_protein = row["reference_apo_pdb"]
    reference_glycan = row["reference_glycan_pdb"]

    reference_output = output_dir / "reference_glycan.pdb"
    copy_reference_glycan(reference_glycan, reference_output)

    report = {
        "pdb_id": pdb_id,
        "site": site,
        "reference_protein_pdb": reference_protein,
        "reference_glycan_input_pdb": reference_glycan,
        "reference_glycan_output_pdb": str(reference_output),
        "tools": {},
    }

    tool_specs = [
        (
            "glycoshield",
            row["glycoshield_glycan_ensemble_pdb"],
            results_root / pdb_id / "glycoshield" / site / "glycoshield_bundle.json",
            output_dir / "glycoshield_glycan_ensemble.pdb",
        ),
        (
            "glycoshape",
            row["glycoshape_glycan_ensemble_pdb"],
            results_root / pdb_id / "glycoshape" / site / "glycoshape_bundle.json",
            output_dir / "glycoshape_glycan_ensemble.pdb",
        ),
    ]

    for tool_name, glycan_input, bundle_path, glycan_output in tool_specs:
        if not glycan_input:
            continue

        if not bundle_path.exists():
            continue

        bundle = load_json(bundle_path)
        source_full_pdb = bundle["source_pdb"]

        alignment = build_protein_alignment(
            reference_protein_pdb=reference_protein,
            source_full_pdb=source_full_pdb,
        )

        transform_pdb_file(
            input_pdb=glycan_input,
            output_pdb=glycan_output,
            rotation=alignment["rotation"],
            translation=alignment["translation"],
        )

        report["tools"][tool_name] = {
            "source_full_pdb": source_full_pdb,
            "glycan_input_pdb": glycan_input,
            "glycan_output_pdb": str(glycan_output),
            "protein_alignment_rmsd": round(alignment["rmsd"], 6),
            "n_common_ca": alignment["n_common_ca"],
        }

    report_path = output_dir / "protein_frame_alignment_report.json"

    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return report
