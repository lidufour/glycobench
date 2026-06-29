from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
from typing import Any

import numpy as np

from glycobench.mapping.glycan_residue_mapping import (
    ResidueKey,
    group_atoms_by_residue,
    residue_centroid,
    build_site_mapping_report,
    write_site_mapping_report,
)
from glycobench.mapping.protein_frame_alignment import (
    Atom,
    kabsch_transform,
    parse_models,
    parse_atom_line,
    read_text_maybe_zip,
    replace_xyz_in_pdb_line,
    transform_xyz,
)


@dataclass(frozen=True)
class ModelTransform:
    model: int
    rotation: np.ndarray
    translation: np.ndarray
    centroid_rmsd: float
    n_centroids_used: int


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def copy_pdb(input_pdb: str | Path, output_pdb: str | Path) -> None:
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_pdb, output_pdb)


def residue_key_from_report(data: dict[str, Any]) -> ResidueKey:
    return ResidueKey.from_dict(data)


def centroids_by_residue(atoms: list[Atom]) -> dict[ResidueKey, tuple[float, float, float]]:
    grouped = group_atoms_by_residue(atoms)
    return {
        key: residue_centroid(residue_atoms, heavy_only=True)
        for key, residue_atoms in grouped.items()
    }


def build_centroid_arrays(
    reference_centroids: dict[ResidueKey, tuple[float, float, float]],
    source_centroids: dict[ResidueKey, tuple[float, float, float]],
    mapping_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    """Return source and reference centroid arrays in matched order.

    The Kabsch convention used in the pipeline is ``source @ R + t = target``.
    Therefore this function returns ``source_xyz`` first and ``reference_xyz``
    second.
    """

    source_xyz: list[tuple[float, float, float]] = []
    reference_xyz: list[tuple[float, float, float]] = []
    used_mapping: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in mapping_rows:
        reference_key = residue_key_from_report(item["reference"])
        source_key = residue_key_from_report(item["source"])

        if reference_key not in reference_centroids:
            warnings.append(f"reference centroid missing for {reference_key.short()}")
            continue

        if source_key not in source_centroids:
            warnings.append(f"source centroid missing for {source_key.short()}")
            continue

        reference_xyz.append(reference_centroids[reference_key])
        source_xyz.append(source_centroids[source_key])
        used_mapping.append(item)

    return (
        np.array(source_xyz, dtype=float),
        np.array(reference_xyz, dtype=float),
        used_mapping,
        warnings,
    )


def build_model_transforms(
    reference_pdb: str | Path,
    source_pdb: str | Path,
    mapping_rows: list[dict[str, Any]],
    min_centroids: int = 2,
) -> tuple[dict[int, ModelTransform], list[dict[str, Any]], list[str]]:
    reference_text = read_text_maybe_zip(reference_pdb)
    source_text = read_text_maybe_zip(source_pdb)

    reference_models = parse_models(reference_text)
    source_models = parse_models(source_text)

    if not reference_models:
        return {}, [], [f"no model found in reference PDB: {reference_pdb}"]

    if not source_models:
        return {}, [], [f"no model found in source PDB: {source_pdb}"]

    reference_first_model = sorted(reference_models)[0]
    reference_centroids = centroids_by_residue(reference_models[reference_first_model])

    transforms: dict[int, ModelTransform] = {}
    model_reports: list[dict[str, Any]] = []
    warnings: list[str] = []

    for model in sorted(source_models):
        source_centroids = centroids_by_residue(source_models[model])
        source_xyz, reference_xyz, used_mapping, centroid_warnings = build_centroid_arrays(
            reference_centroids=reference_centroids,
            source_centroids=source_centroids,
            mapping_rows=mapping_rows,
        )
        warnings.extend([f"model {model}: {warning}" for warning in centroid_warnings])

        n_centroids = len(source_xyz)
        model_report: dict[str, Any] = {
            "model": model,
            "n_centroids_used": n_centroids,
            "n_mapping_rows": len(mapping_rows),
            "status": "ok",
            "alignment_quality": "robust_3d" if n_centroids >= 3 else "fragile_two_centroids",
        }

        if n_centroids < min_centroids:
            model_report["status"] = "failed"
            model_report["warning"] = (
                f"not enough mapped centroids for robust 3D Kabsch alignment: "
                f"{n_centroids} < {min_centroids}"
            )
            warnings.append(f"model {model}: {model_report['warning']}")
            model_reports.append(model_report)
            continue

        rotation, translation, rmsd = kabsch_transform(source_xyz, reference_xyz)
        transforms[model] = ModelTransform(
            model=model,
            rotation=rotation,
            translation=translation,
            centroid_rmsd=rmsd,
            n_centroids_used=n_centroids,
        )
        model_report["centroid_rmsd"] = round(rmsd, 6)
        model_report["used_mapping"] = [
            {
                "reference": item["reference"],
                "source": item["source"],
                "reference_class": item.get("reference_class"),
                "source_class": item.get("source_class"),
            }
            for item in used_mapping
        ]
        model_reports.append(model_report)

    return transforms, model_reports, warnings


def transform_pdb_by_model(
    input_pdb: str | Path,
    output_pdb: str | Path,
    transforms: dict[int, ModelTransform],
) -> None:
    input_text = read_text_maybe_zip(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    current_model = 1
    saw_model_record = False

    with output_pdb.open("w") as handle:
        for line in input_text.splitlines():
            if line.startswith("MODEL"):
                saw_model_record = True
                try:
                    current_model = int(line[10:14])
                except ValueError:
                    current_model += 1
                handle.write(line.rstrip() + "\n")
                continue

            if line.startswith("ENDMDL"):
                handle.write(line.rstrip() + "\n")
                continue

            atom = parse_atom_line(line, current_model)
            if atom is None:
                handle.write(line.rstrip() + "\n")
                continue

            if atom.model not in transforms:
                raise ValueError(
                    f"No glycan-frame transform available for model {atom.model} in {input_pdb}."
                )

            transform = transforms[atom.model]
            x, y, z = transform_xyz(atom.x, atom.y, atom.z, transform.rotation, transform.translation)
            handle.write(replace_xyz_in_pdb_line(line, x, y, z) + "\n")

    # Silence the currently unused variable but keep it for readability; the
    # parser and writer both support PDBs with or without explicit MODEL records.
    _ = saw_model_record


def summarize_rmsd(model_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rmsds = [item["centroid_rmsd"] for item in model_reports if item.get("status") == "ok" and "centroid_rmsd" in item]

    if not rmsds:
        return {
            "mean_centroid_rmsd": None,
            "median_centroid_rmsd": None,
            "min_centroid_rmsd": None,
            "max_centroid_rmsd": None,
        }

    values = np.array(rmsds, dtype=float)
    return {
        "mean_centroid_rmsd": round(float(values.mean()), 6),
        "median_centroid_rmsd": round(float(np.median(values)), 6),
        "min_centroid_rmsd": round(float(values.min()), 6),
        "max_centroid_rmsd": round(float(values.max()), 6),
    }


def align_tool_in_glycan_frame(
    tool_name: str,
    mapping_report: dict[str, Any],
    output_dir: Path,
    min_centroids: int = 2,
) -> dict[str, Any]:
    tool_mapping = mapping_report["tools"].get(tool_name)

    if tool_mapping is None:
        return {
            "status": "skipped",
            "tool": tool_name,
            "warnings": [f"no mapping report for {tool_name}"],
        }

    reference_pdb = Path(tool_mapping["reference_pdb_path"])
    source_pdb = Path(tool_mapping["source_pdb_path"])
    output_pdb = output_dir / f"{tool_name}_glycan_ensemble.pdb"

    report: dict[str, Any] = {
        "tool": tool_name,
        "status": "ok",
        "method": "monosaccharide_centroid_kabsch",
        "mapping_status": tool_mapping.get("status"),
        "reference_pdb_path": str(reference_pdb),
        "source_pdb_path": str(source_pdb),
        "output_pdb_path": str(output_pdb),
        "n_reference_residues": tool_mapping.get("n_reference_residues"),
        "n_source_residues": tool_mapping.get("n_source_residues"),
        "n_mapped_residues": tool_mapping.get("n_mapped_residues"),
        "min_centroids_required": min_centroids,
        "mapping_warnings": tool_mapping.get("warnings", []),
        "warnings": [],
        "models": [],
    }

    if tool_mapping.get("status") == "failed" or not tool_mapping.get("mapping"):
        report["status"] = "failed"
        report["warnings"].append("cannot align because residue mapping failed or is empty")
        return report

    transforms, model_reports, warnings = build_model_transforms(
        reference_pdb=reference_pdb,
        source_pdb=source_pdb,
        mapping_rows=tool_mapping["mapping"],
        min_centroids=min_centroids,
    )

    report["models"] = model_reports
    report["warnings"].extend(warnings)
    report["n_models_input"] = len(model_reports)
    report["n_models_aligned"] = len(transforms)
    report.update(summarize_rmsd(model_reports))

    if len(transforms) == 0:
        report["status"] = "failed"
        report["warnings"].append("no model could be aligned")
        return report

    if len(transforms) < len(model_reports):
        report["status"] = "partial"
        report["warnings"].append(
            f"only {len(transforms)}/{len(model_reports)} model(s) could be aligned"
        )
        # Avoid creating a mixed output where some models are transformed and
        # others are not. This should not happen for current triple_core cases.
        return report

    transform_pdb_by_model(source_pdb, output_pdb, transforms)

    if any(item.get("alignment_quality") == "fragile_two_centroids" for item in model_reports):
        report["status"] = "fragile"
        report["warnings"].append(
            "alignment used only two mapped centroids; translation and main axis are defined, "
            "but rotation around that axis is underdetermined"
        )

    if tool_mapping.get("status") == "partial":
        report["status"] = "partial"
        report["warnings"].append("alignment used a partial residue mapping; interpreted as common scaffold only")

    return report


def ensure_mapping_report(
    row: dict[str, str],
    results_root: Path,
    include_glycoshape: bool,
    include_glycoshield: bool = True,
) -> tuple[dict[str, Any], Path]:
    pdb_id = row["pdb_id"]
    site = row["site"]
    mapping_report_path = results_root / pdb_id / "aligned" / site / "glycan_mapping" / "glycan_residue_mapping_report.json"

    if mapping_report_path.exists():
        return load_json(mapping_report_path), mapping_report_path

    report = build_site_mapping_report(
        pdb_id=pdb_id,
        site=site,
        results_root=results_root,
        include_glycoshape=include_glycoshape,
        include_glycoshield=include_glycoshield,
    )
    written_path = write_site_mapping_report(report, results_root=results_root)
    return report, written_path


def align_site_in_glycan_frame(
    row: dict[str, str],
    results_root: str | Path = Path("results"),
    include_glycoshape: bool | None = None,
    include_glycoshield: bool = True,
    min_centroids: int = 2,
) -> dict[str, Any]:
    results_root = Path(results_root)
    pdb_id = row["pdb_id"]
    site = row["site"]

    if include_glycoshape is None:
        include_glycoshape = bool(row.get("glycoshape_glycan_ensemble_pdb"))

    output_dir = results_root / pdb_id / "aligned" / site / "glycan_frame"
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_report, mapping_report_path = ensure_mapping_report(
        row=row,
        results_root=results_root,
        include_glycoshape=include_glycoshape,
        include_glycoshield=include_glycoshield,
    )

    protein_frame_dir = results_root / pdb_id / "aligned" / site / "protein_frame"
    reference_input = protein_frame_dir / "reference_glycan.pdb"
    reference_output = output_dir / "reference_glycan.pdb"

    report: dict[str, Any] = {
        "pdb_id": pdb_id,
        "site": site,
        "status": "ok",
        "method": "monosaccharide_centroid_kabsch",
        "mapping_report_path": str(mapping_report_path),
        "reference_input_pdb": str(reference_input),
        "reference_output_pdb": str(reference_output),
        "min_centroids_required": min_centroids,
        "tools": {},
        "warnings": [],
    }

    if not reference_input.exists():
        report["status"] = "failed"
        report["warnings"].append(f"missing protein-frame reference glycan: {reference_input}")
        return report

    copy_pdb(reference_input, reference_output)

    tool_names: list[str] = []
    if include_glycoshield:
        tool_names.append("glycoshield")
    if include_glycoshape:
        tool_names.append("glycoshape")

    for tool_name in tool_names:
        tool_report = align_tool_in_glycan_frame(
            tool_name=tool_name,
            mapping_report=mapping_report,
            output_dir=output_dir,
            min_centroids=min_centroids,
        )
        report["tools"][tool_name] = tool_report

    tool_statuses = [tool_report.get("status") for tool_report in report["tools"].values()]
    if not tool_statuses:
        report["status"] = "failed"
        report["warnings"].append("no tool selected for glycan-frame alignment")
    elif any(status == "failed" for status in tool_statuses):
        report["status"] = "partial"
    elif any(status in {"partial", "fragile"} for status in tool_statuses):
        report["status"] = "partial"

    report_path = output_dir / "glycan_frame_alignment_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return report
