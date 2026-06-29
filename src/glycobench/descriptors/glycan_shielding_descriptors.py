from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import argparse
import csv
import json
import math

import mdtraj as md

from glycobench.descriptors.glycontact_adapter import (
    DEFAULT_RECOMMENDED_USES,
    GlycanFramePathSet,
    glycan_frame_dir_from_row,
    join_warnings,
    load_csv_rows,
    resolve_glycan_frame_paths,
    split_pdb_models_to_temp_files,
    write_json,
)


PROBE_RADIUS_NM = 0.14
NM2_TO_A2 = 100.0
DEFAULT_MASK_THRESHOLD_A2 = 1.0


def clean_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return str(round(value, 6))
    return str(value)


def write_csv_union(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("")
        return output_path

    preferred = [
        "pdb_id",
        "site",
        "glycan_class",
        "glycan_name",
        "glytoucan",
        "analysis_group",
        "scope",
        "tool",
        "recommended_use",
        "model_index",
        "chain_id",
        "residue_number",
        "residue_name",
        "residue_key",
        "sasa_apo_a2",
        "sasa_glycosylated_a2",
        "delta_sasa_a2",
        "masked",
        "shielding_status",
        "warnings",
    ]

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv_value(row.get(key, "")) for key in fieldnames})

    return output_path


def protein_frame_dir_from_path_set(path_set: GlycanFramePathSet) -> Path:
    glycan_frame_dir = glycan_frame_dir_from_row(path_set.source_qc_row, results_root="results")
    return glycan_frame_dir.parent / "protein_frame"


def protein_frame_glycan_path(path_set: GlycanFramePathSet) -> Path:
    frame_dir = protein_frame_dir_from_path_set(path_set)
    if path_set.tool == "reference":
        return frame_dir / "reference_glycan.pdb"
    return frame_dir / f"{path_set.tool}_glycan_ensemble.pdb"


def protein_frame_report_path(path_set: GlycanFramePathSet) -> Path:
    return protein_frame_dir_from_path_set(path_set) / "protein_frame_alignment_report.json"


def load_reference_protein_path(path_set: GlycanFramePathSet) -> Path:
    report_path = protein_frame_report_path(path_set)
    report = json.loads(report_path.read_text())
    return Path(report["reference_protein_pdb"])


def is_hydrogen_pdb_line(line: str) -> bool:
    if not line.startswith(("ATOM", "HETATM")):
        return False

    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    atom_name = line[12:16].strip().upper()

    if element == "H":
        return True
    return atom_name.startswith("H")


def atom_lines_without_hydrogen(path: str | Path) -> list[str]:
    lines: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM", "TER")):
                continue
            if is_hydrogen_pdb_line(line):
                continue
            lines.append(line.rstrip())
    return lines


def write_pdb_without_hydrogen(
    input_pdb: str | Path,
    output_pdb: str | Path,
) -> Path:
    lines = atom_lines_without_hydrogen(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    cleaned = [line for line in lines if line != "TER"]
    cleaned.append("END")

    output_pdb.write_text("\n".join(cleaned) + "\n")
    return output_pdb


def write_combined_pdb_without_hydrogen(
    protein_pdb: str | Path,
    glycan_model_pdb: str | Path,
    output_pdb: str | Path,
) -> Path:
    protein_lines = atom_lines_without_hydrogen(protein_pdb)
    glycan_lines = atom_lines_without_hydrogen(glycan_model_pdb)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    merged = []
    merged.extend(line for line in protein_lines if line != "TER")
    merged.append("TER")
    merged.extend(line for line in glycan_lines if line != "TER")
    merged.append("END")

    output_pdb.write_text("\n".join(merged) + "\n")
    return output_pdb


def residue_key_from_mdtraj_residue(residue: Any) -> str:
    chain_id = getattr(residue.chain, "chain_id", "") or str(residue.chain.index)
    return f"{chain_id}:{residue.resSeq}:{residue.name}"


def residue_metadata_from_mdtraj_residue(residue: Any) -> dict[str, Any]:
    chain_id = getattr(residue.chain, "chain_id", "") or str(residue.chain.index)
    return {
        "chain_id": chain_id,
        "residue_number": residue.resSeq,
        "residue_name": residue.name,
        "residue_key": residue_key_from_mdtraj_residue(residue),
    }


def compute_protein_residue_sasa_a2(pdb_path: str | Path) -> dict[str, dict[str, Any]]:
    traj = md.load(str(pdb_path))
    atom_sasa_nm2 = md.shrake_rupley(
        traj,
        probe_radius=PROBE_RADIUS_NM,
        mode="atom",
    )[0]

    residue_data: dict[str, dict[str, Any]] = {}

    for atom in traj.topology.atoms:
        residue = atom.residue

        if not residue.is_protein:
            continue

        key = residue_key_from_mdtraj_residue(residue)

        if key not in residue_data:
            residue_data[key] = {
                **residue_metadata_from_mdtraj_residue(residue),
                "sasa_a2": 0.0,
            }

        residue_data[key]["sasa_a2"] += float(atom_sasa_nm2[atom.index]) * NM2_TO_A2

    return residue_data


def base_metadata(path_set: GlycanFramePathSet, model_index: int) -> dict[str, Any]:
    return {
        "pdb_id": path_set.pdb_id,
        "site": path_set.site,
        "glycan_class": path_set.glycan_class,
        "glycan_name": path_set.glycan_name,
        "glytoucan": path_set.glytoucan,
        "analysis_group": path_set.analysis_group,
        "scope": path_set.scope,
        "tool": path_set.tool,
        "recommended_use": path_set.recommended_use,
        "model_index": model_index,
    }


def process_path_set(
    path_set: GlycanFramePathSet,
    temp_root: Path,
    max_models: int | None,
    mask_threshold_a2: float,
    apo_sasa_cache: dict[Path, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    residue_rows: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []

    reference_protein_pdb = load_reference_protein_path(path_set).resolve()
    glycan_pdb = protein_frame_glycan_path(path_set)

    if reference_protein_pdb not in apo_sasa_cache:
        apo_no_h_pdb = temp_root / f"{path_set.pdb_id}_{path_set.site}_reference_protein_no_h.pdb"
        write_pdb_without_hydrogen(reference_protein_pdb, apo_no_h_pdb)
        apo_sasa_cache[reference_protein_pdb] = compute_protein_residue_sasa_a2(apo_no_h_pdb)

    apo_sasa = apo_sasa_cache[reference_protein_pdb]

    prefix = f"{path_set.pdb_id}_{path_set.site}_{path_set.tool}"

    try:
        glycan_model_files = split_pdb_models_to_temp_files(
            glycan_pdb,
            temp_root,
            prefix=prefix,
            max_models=max_models,
        )
    except Exception as exc:
        model_reports.append({
            **base_metadata(path_set, 0),
            "shielding_status": "failed",
            "warnings": f"could not split glycan models: {type(exc).__name__}: {exc}",
        })
        return residue_rows, model_reports

    for model_index, glycan_model_pdb in enumerate(glycan_model_files, start=1):
        metadata = base_metadata(path_set, model_index)
        warnings: list[str] = []

        combined_pdb = temp_root / f"{prefix}_combined_model_{model_index:04d}.pdb"

        try:
            write_combined_pdb_without_hydrogen(
                reference_protein_pdb,
                glycan_model_pdb,
                combined_pdb,
            )
            gly_sasa = compute_protein_residue_sasa_a2(combined_pdb)
        except Exception as exc:
            model_reports.append({
                **metadata,
                "shielding_status": "failed",
                "warnings": f"SASA calculation failed: {type(exc).__name__}: {exc}",
            })
            continue

        n_residues = 0
        n_masked = 0
        total_delta = 0.0

        for residue_key, apo_row in apo_sasa.items():
            apo_value = float(apo_row["sasa_a2"])
            gly_value = float(gly_sasa.get(residue_key, {}).get("sasa_a2", apo_value))
            delta = apo_value - gly_value

            # Numerical noise can create tiny negative values.
            if abs(delta) < 1e-6:
                delta = 0.0

            masked = delta >= mask_threshold_a2

            n_residues += 1
            n_masked += int(masked)
            total_delta += max(delta, 0.0)

            residue_rows.append({
                **metadata,
                "chain_id": apo_row["chain_id"],
                "residue_number": apo_row["residue_number"],
                "residue_name": apo_row["residue_name"],
                "residue_key": residue_key,
                "sasa_apo_a2": apo_value,
                "sasa_glycosylated_a2": gly_value,
                "delta_sasa_a2": delta,
                "masked": masked,
                "shielding_status": "ok",
                "warnings": join_warnings(warnings),
            })

        model_reports.append({
            **metadata,
            "shielding_status": "ok",
            "n_residues": n_residues,
            "n_masked_residues": n_masked,
            "total_positive_delta_sasa_a2": total_delta,
            "warnings": join_warnings(warnings),
        })

    return residue_rows, model_reports


def summarize_residue_shielding(
    residue_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    keys = [
        "pdb_id",
        "site",
        "glycan_class",
        "glycan_name",
        "glytoucan",
        "analysis_group",
        "scope",
        "tool",
        "recommended_use",
        "chain_id",
        "residue_number",
        "residue_name",
        "residue_key",
    ]

    for row in residue_rows:
        if row.get("shielding_status") != "ok":
            continue
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    summary_rows: list[dict[str, Any]] = []

    for key_values, rows in grouped.items():
        base = dict(zip(keys, key_values))
        deltas = [float(row["delta_sasa_a2"]) for row in rows]
        masked = [bool(row["masked"]) for row in rows]

        summary_rows.append({
            **base,
            "n_models": len(rows),
            "mean_delta_sasa_a2": sum(deltas) / len(deltas),
            "max_delta_sasa_a2": max(deltas),
            "masking_frequency": sum(masked) / len(masked),
            "n_masked_models": sum(masked),
        })

    return summary_rows


def build_shielding_descriptors(
    qc_path: str | Path = "results/glycan_frame_qc.csv",
    results_root: str | Path = "results",
    output_residue_sasa: str | Path = "results/glycan_shielding_residue_sasa.csv",
    output_summary_csv: str | Path = "results/glycan_shielding_summary.csv",
    output_summary_json: str | Path = "results/glycan_shielding_descriptors_summary.json",
    max_models: int | None = 1,
    include_fragile: bool = False,
    mask_threshold_a2: float = DEFAULT_MASK_THRESHOLD_A2,
) -> dict[str, Any]:
    qc_rows = load_csv_rows(qc_path)

    path_sets = resolve_glycan_frame_paths(
        qc_rows,
        results_root=results_root,
        recommended_uses=DEFAULT_RECOMMENDED_USES,
        include_fragile=include_fragile,
        include_reference=True,
    )

    residue_rows: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []
    apo_sasa_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    with TemporaryDirectory() as tmp:
        temp_root = Path(tmp)

        for path_set in path_sets:
            current_rows, current_reports = process_path_set(
                path_set,
                temp_root=temp_root,
                max_models=max_models,
                mask_threshold_a2=mask_threshold_a2,
                apo_sasa_cache=apo_sasa_cache,
            )
            residue_rows.extend(current_rows)
            model_reports.extend(current_reports)

    summary_rows = summarize_residue_shielding(residue_rows)

    write_csv_union(residue_rows, output_residue_sasa)
    write_csv_union(summary_rows, output_summary_csv)

    summary = {
        "qc_path": str(qc_path),
        "results_root": str(results_root),
        "output_residue_sasa": str(output_residue_sasa),
        "output_summary_csv": str(output_summary_csv),
        "max_models": "all" if max_models is None else max_models,
        "probe_radius_nm": PROBE_RADIUS_NM,
        "probe_radius_angstrom": PROBE_RADIUS_NM * 10.0,
        "mask_threshold_a2": mask_threshold_a2,
        "n_path_sets": len(path_sets),
        "n_models_processed": len(model_reports),
        "n_residue_sasa_rows": len(residue_rows),
        "n_summary_rows": len(summary_rows),
        "by_tool_models": dict(Counter(report["tool"] for report in model_reports)),
        "by_shielding_status_models": dict(Counter(report.get("shielding_status", "") for report in model_reports)),
        "warnings_examples": [
            report["warnings"]
            for report in model_reports
            if report.get("warnings")
        ][:20],
    }

    write_json(summary, output_summary_json)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build glycan shielding descriptors by protein-residue ΔSASA."
    )
    parser.add_argument("--qc-path", default="results/glycan_frame_qc.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-residue-sasa", default="results/glycan_shielding_residue_sasa.csv")
    parser.add_argument("--output-summary-csv", default="results/glycan_shielding_summary.csv")
    parser.add_argument("--output-summary-json", default="results/glycan_shielding_descriptors_summary.json")
    parser.add_argument("--max-models", type=int, default=1)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--include-fragile", action="store_true")
    parser.add_argument("--mask-threshold-a2", type=float, default=DEFAULT_MASK_THRESHOLD_A2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    max_models = None if args.all_models else args.max_models

    summary = build_shielding_descriptors(
        qc_path=args.qc_path,
        results_root=args.results_root,
        output_residue_sasa=args.output_residue_sasa,
        output_summary_csv=args.output_summary_csv,
        output_summary_json=args.output_summary_json,
        max_models=max_models,
        include_fragile=args.include_fragile,
        mask_threshold_a2=args.mask_threshold_a2,
    )

    print(f"[OK] wrote {args.output_residue_sasa} ({summary['n_residue_sasa_rows']} rows)")
    print(f"[OK] wrote {args.output_summary_csv} ({summary['n_summary_rows']} rows)")
    print(f"[OK] wrote {args.output_summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
