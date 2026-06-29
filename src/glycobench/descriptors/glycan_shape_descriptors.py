from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable
import argparse
import csv
import json
import math
import statistics

import numpy as np

from glycobench.descriptors.glycontact_adapter import (
    DEFAULT_RECOMMENDED_USES,
    GlycanFramePathSet,
    call_extract_coordinates,
    call_get_sequences,
    call_superimpose,
    count_models_and_first_model_atoms,
    fmt_float,
    import_glycontact,
    join_warnings,
    load_csv_rows,
    resolve_glycan_frame_paths,
    safe_float,
    split_pdb_models_to_temp_files,
    write_csv_rows,
    write_json,
)


DEFAULT_OUTPUT_CSV = "results/glycan_shape_descriptors.csv"
DEFAULT_OUTPUT_SUMMARY_JSON = "results/glycan_shape_descriptors_summary.json"


@dataclass(frozen=True)
class ShapeMetrics:
    n_atoms: int
    n_heavy_atoms: int
    n_residues: int
    center_x: float
    center_y: float
    center_z: float
    rg: float
    span: float
    bbox_x: float
    bbox_y: float
    bbox_z: float


def _is_hydrogen_row(row: Any) -> bool:
    element = str(row.get("element", "") or "").strip().upper()
    atom_name = str(row.get("atom_name", "") or "").strip().upper()
    return element == "H" or atom_name.startswith("H")


def coordinates_from_glycontact_df(df: Any, include_hydrogen: bool = False) -> tuple[np.ndarray, int, int, int]:
    """Extract an XYZ array and simple atom/residue counts from a GlyContact DataFrame."""

    if df is None or not hasattr(df, "iterrows"):
        return np.empty((0, 3), dtype=float), 0, 0, 0

    all_rows = []
    selected_xyz: list[list[float]] = []
    residue_keys: set[tuple[str, str, str]] = set()

    for _idx, row in df.iterrows():
        all_rows.append(row)
        chain = str(row.get("chain_id", "") or "")
        resnum = str(row.get("residue_number", "") or "")
        mono = str(row.get("monosaccharide", "") or "")
        residue_keys.add((chain, resnum, mono))

        if not include_hydrogen and _is_hydrogen_row(row):
            continue

        try:
            selected_xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
        except Exception:
            continue

    n_atoms = len(all_rows)
    n_selected = len(selected_xyz)
    n_residues = len(residue_keys)
    xyz = np.array(selected_xyz, dtype=float) if selected_xyz else np.empty((0, 3), dtype=float)
    return xyz, n_atoms, n_selected, n_residues


def calculate_shape_metrics(df: Any, include_hydrogen: bool = False) -> tuple[ShapeMetrics | None, list[str]]:
    """Calculate global glycan-shape descriptors from atom coordinates.

    Metrics are intentionally simple and model-agnostic:
    - radius of gyration over selected atoms;
    - span, i.e. maximal pairwise atom distance;
    - bounding-box extents along x, y and z.
    """

    warnings: list[str] = []
    xyz, n_atoms, n_heavy_atoms, n_residues = coordinates_from_glycontact_df(
        df,
        include_hydrogen=include_hydrogen,
    )

    if xyz.size == 0:
        return None, ["no coordinates available for shape metrics"]

    center = xyz.mean(axis=0)
    centered = xyz - center
    rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))

    if len(xyz) < 2:
        span = 0.0
        warnings.append("span set to 0 because fewer than two selected atoms were available")
    else:
        diff = xyz[:, None, :] - xyz[None, :, :]
        distances_sq = np.sum(diff * diff, axis=2)
        span = float(np.sqrt(np.max(distances_sq)))

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    bbox = maxs - mins

    return (
        ShapeMetrics(
            n_atoms=n_atoms,
            n_heavy_atoms=n_heavy_atoms,
            n_residues=n_residues,
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            rg=rg,
            span=span,
            bbox_x=float(bbox[0]),
            bbox_y=float(bbox[1]),
            bbox_z=float(bbox[2]),
        ),
        warnings,
    )


def metric_fields(metrics: ShapeMetrics | None) -> dict[str, str]:
    if metrics is None:
        return {
            "n_atoms": "",
            "n_heavy_atoms": "",
            "n_residues": "",
            "center_x": "",
            "center_y": "",
            "center_z": "",
            "rg_angstrom": "",
            "span_angstrom": "",
            "bbox_x_angstrom": "",
            "bbox_y_angstrom": "",
            "bbox_z_angstrom": "",
        }

    return {
        "n_atoms": str(metrics.n_atoms),
        "n_heavy_atoms": str(metrics.n_heavy_atoms),
        "n_residues": str(metrics.n_residues),
        "center_x": fmt_float(metrics.center_x),
        "center_y": fmt_float(metrics.center_y),
        "center_z": fmt_float(metrics.center_z),
        "rg_angstrom": fmt_float(metrics.rg),
        "span_angstrom": fmt_float(metrics.span),
        "bbox_x_angstrom": fmt_float(metrics.bbox_x),
        "bbox_y_angstrom": fmt_float(metrics.bbox_y),
        "bbox_z_angstrom": fmt_float(metrics.bbox_z),
    }


def _base_descriptor_row(path_set: GlycanFramePathSet, glycontact: Any) -> dict[str, Any]:
    return {
        "pdb_id": path_set.pdb_id,
        "site": path_set.site,
        "tool": path_set.tool,
        "scope": path_set.scope,
        "analysis_group": path_set.analysis_group,
        "glycan_class": path_set.glycan_class,
        "glycan_name": path_set.glycan_name,
        "glytoucan": path_set.glytoucan,
        "recommended_use": path_set.recommended_use,
        "pdb_path": str(path_set.pdb_path),
        "reference_pdb_path": str(path_set.reference_pdb_path),
        "glycontact_version": str(getattr(glycontact, "__version__", "unknown")),
        "model_index": "",
        "n_models_total": "",
        "n_models_described": "",
        "sequence_status": "not_run",
        "sequence": "",
        "coordinate_status": "not_run",
        "rmsd_to_reference_status": "not_run",
        "rmsd_to_reference": "",
        "descriptor_status": "not_run",
        "warnings": "",
    }


def describe_one_model(
    path_set: GlycanFramePathSet,
    glycontact: Any,
    model_pdb: Path,
    model_index: int,
    reference_model_pdb: Path | None,
    n_models_total: int,
    n_models_described: int,
    include_hydrogen: bool = False,
    superimpose_fast: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    row = _base_descriptor_row(path_set, glycontact)
    row.update(
        {
            "model_index": str(model_index),
            "n_models_total": str(n_models_total),
            "n_models_described": str(n_models_described),
        }
    )

    seq_status, _n_sequences, sequences, seq_warnings = call_get_sequences(glycontact, model_pdb)
    warnings.extend(seq_warnings)
    row["sequence_status"] = seq_status
    row["sequence"] = sequences[0] if sequences else ""

    coordinate_status, coordinates_df, _n_coordinate_rows, _coordinate_columns, coord_warnings = call_extract_coordinates(
        glycontact,
        model_pdb,
    )
    warnings.extend(coord_warnings)
    row["coordinate_status"] = coordinate_status

    metrics, metric_warnings = calculate_shape_metrics(coordinates_df, include_hydrogen=include_hydrogen)
    warnings.extend(metric_warnings)
    row.update(metric_fields(metrics))

    if path_set.tool == "reference":
        row["rmsd_to_reference_status"] = "self_reference"
        row["rmsd_to_reference"] = "0.0"
    elif reference_model_pdb is None:
        row["rmsd_to_reference_status"] = "not_run"
        warnings.append("reference model is missing, RMSD skipped")
    else:
        rmsd_status, rmsd, rmsd_warnings = call_superimpose(
            glycontact,
            reference_model_pdb=reference_model_pdb,
            mobile_model_pdb=model_pdb,
            fast=superimpose_fast,
        )
        warnings.extend(rmsd_warnings)
        row["rmsd_to_reference_status"] = rmsd_status
        row["rmsd_to_reference"] = fmt_float(rmsd)

    if coordinate_status == "ok" and metrics is not None:
        row["descriptor_status"] = "ok"
    elif coordinate_status in {"ok", "partial"}:
        row["descriptor_status"] = "partial"
    else:
        row["descriptor_status"] = "failed"

    row["warnings"] = join_warnings(warnings)
    return row


def describe_site_tool_shape(
    path_set: GlycanFramePathSet,
    glycontact: Any | None = None,
    max_models: int | None = 1,
    include_hydrogen: bool = False,
    superimpose_fast: bool = True,
) -> list[dict[str, Any]]:
    """Build per-model shape descriptors for one reference/tool PDB."""

    if glycontact is None:
        glycontact = import_glycontact()

    if not path_set.pdb_path.exists():
        row = _base_descriptor_row(path_set, glycontact)
        row["descriptor_status"] = "failed"
        row["warnings"] = f"missing PDB file: {path_set.pdb_path}"
        return [row]

    try:
        counts = count_models_and_first_model_atoms(path_set.pdb_path)
        n_models_total = int(counts["n_models"])
    except Exception as exc:
        row = _base_descriptor_row(path_set, glycontact)
        row["descriptor_status"] = "failed"
        row["warnings"] = f"internal PDB parser failed: {type(exc).__name__}: {exc}"
        return [row]

    with TemporaryDirectory(prefix="glycobench_shape_") as tmp:
        tmp_dir = Path(tmp)
        model_paths = split_pdb_models_to_temp_files(
            path_set.pdb_path,
            tmp_dir,
            prefix=f"{path_set.pdb_id}_{path_set.site}_{path_set.tool}",
            max_models=max_models,
        )

        if not model_paths:
            row = _base_descriptor_row(path_set, glycontact)
            row["n_models_total"] = str(n_models_total)
            row["descriptor_status"] = "failed"
            row["warnings"] = "no model file could be created from PDB"
            return [row]

        reference_model_pdb: Path | None = None
        if path_set.reference_pdb_path.exists():
            reference_models = split_pdb_models_to_temp_files(
                path_set.reference_pdb_path,
                tmp_dir,
                prefix=f"{path_set.pdb_id}_{path_set.site}_reference",
                max_models=1,
            )
            if reference_models:
                reference_model_pdb = reference_models[0]

        return [
            describe_one_model(
                path_set=path_set,
                glycontact=glycontact,
                model_pdb=model_pdb,
                model_index=index,
                reference_model_pdb=reference_model_pdb,
                n_models_total=n_models_total,
                n_models_described=len(model_paths),
                include_hydrogen=include_hydrogen,
                superimpose_fast=superimpose_fast,
            )
            for index, model_pdb in enumerate(model_paths, start=1)
        ]


def build_glycan_shape_descriptors(
    glycan_frame_qc_path: str | Path = "results/glycan_frame_qc.csv",
    results_root: str | Path = "results",
    recommended_uses: Iterable[str] = DEFAULT_RECOMMENDED_USES,
    include_fragile: bool = False,
    include_reference: bool = True,
    max_models: int | None = 1,
    include_hydrogen: bool = False,
    superimpose_fast: bool = True,
) -> list[dict[str, Any]]:
    glycontact = import_glycontact()
    qc_rows = load_csv_rows(glycan_frame_qc_path)
    path_sets = resolve_glycan_frame_paths(
        qc_rows,
        results_root=results_root,
        recommended_uses=recommended_uses,
        include_fragile=include_fragile,
        include_reference=include_reference,
    )

    rows: list[dict[str, Any]] = []
    for path_set in path_sets:
        rows.extend(
            describe_site_tool_shape(
                path_set,
                glycontact=glycontact,
                max_models=max_models,
                include_hydrogen=include_hydrogen,
                superimpose_fast=superimpose_fast,
            )
        )
    return rows


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def summarize_shape_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "by_tool": {},
        "by_descriptor_status": {},
        "by_recommended_use": {},
        "by_glycan_class": {},
        "by_rmsd_status": {},
        "rmsd_to_reference": {},
    }

    for row in rows:
        for field, key in [
            ("tool", "by_tool"),
            ("descriptor_status", "by_descriptor_status"),
            ("recommended_use", "by_recommended_use"),
            ("glycan_class", "by_glycan_class"),
            ("rmsd_to_reference_status", "by_rmsd_status"),
        ]:
            value = str(row.get(field, ""))
            summary[key][value] = summary[key].get(value, 0) + 1

    for tool in sorted({str(row.get("tool", "")) for row in rows}):
        tool_rows = [row for row in rows if str(row.get("tool", "")) == tool]
        rmsds = _numeric_values(tool_rows, "rmsd_to_reference")
        if not rmsds:
            continue
        summary["rmsd_to_reference"][tool] = {
            "n": len(rmsds),
            "min": round(min(rmsds), 6),
            "mean": round(statistics.mean(rmsds), 6),
            "median": round(statistics.median(rmsds), 6),
            "max": round(max(rmsds), 6),
        }

    return summary


def write_glycan_shape_descriptors(
    rows: list[dict[str, Any]],
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    output_summary_json: str | Path = DEFAULT_OUTPUT_SUMMARY_JSON,
) -> tuple[Path, Path]:
    csv_path = write_csv_rows(rows, output_csv)
    summary_path = write_json(summarize_shape_rows(rows), output_summary_json)
    return csv_path, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build global glycan-shape descriptors from aligned glycan-frame PDB files.",
    )
    parser.add_argument(
        "--glycan-frame-qc",
        default="results/glycan_frame_qc.csv",
        help="Input glycan-frame QC CSV.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Output descriptor CSV.",
    )
    parser.add_argument(
        "--output-summary-json",
        default=DEFAULT_OUTPUT_SUMMARY_JSON,
        help="Output descriptor summary JSON.",
    )
    parser.add_argument(
        "--recommended-use",
        action="append",
        default=None,
        help=(
            "recommended_use category to include. Can be passed several times. "
            "Default: strict and exploratory_common_scaffold."
        ),
    )
    parser.add_argument(
        "--include-fragile",
        action="store_true",
        help="Also include exploratory_fragile_exclude_strict rows.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Do not add reference_glycan.pdb rows.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=1,
        help="Number of models to describe per ensemble PDB.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Describe all models in each ensemble PDB.",
    )
    parser.add_argument(
        "--include-hydrogen",
        action="store_true",
        help="Include hydrogen atoms in Rg/span/bounding-box calculations.",
    )
    parser.add_argument(
        "--slow-superimpose",
        action="store_true",
        help="Use GlyContact's non-fast superimpose mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    recommended_uses = args.recommended_use or list(DEFAULT_RECOMMENDED_USES)
    max_models = None if args.all_models else args.max_models
    if max_models is not None and max_models < 1:
        raise ValueError("--max-models must be >= 1, or use --all-models.")

    rows = build_glycan_shape_descriptors(
        glycan_frame_qc_path=args.glycan_frame_qc,
        results_root=args.results_root,
        recommended_uses=recommended_uses,
        include_fragile=args.include_fragile,
        include_reference=not args.no_reference,
        max_models=max_models,
        include_hydrogen=args.include_hydrogen,
        superimpose_fast=not args.slow_superimpose,
    )
    csv_path, summary_path = write_glycan_shape_descriptors(
        rows,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )

    summary = summarize_shape_rows(rows)
    print(f"[OK] wrote {csv_path} ({len(rows)} rows)")
    print(f"[OK] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
