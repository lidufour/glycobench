from __future__ import annotations

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import argparse
import csv
import json
import math

from glycobench.descriptors.glycontact_adapter import (
    DEFAULT_RECOMMENDED_USES,
    GlycanFramePathSet,
    call_extract_coordinates,
    call_get_sequences,
    import_glycontact,
    join_warnings,
    load_csv_rows,
    resolve_glycan_frame_paths,
    split_pdb_models_to_temp_files,
    write_json,
)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt_float(value: Any, ndigits: int = 6) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return str(round(number, ndigits))


def clean_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return fmt_float(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def dataframe_like_to_records(obj: Any) -> tuple[list[dict[str, Any]], str]:
    """Convert a GlyContact return object to list-of-dicts.

    GlyContact functions usually return pandas DataFrames, but this function
    also accepts dicts/lists to keep the wrapper robust across versions.
    """

    if obj is None:
        return [], ""

    columns = ""
    if hasattr(obj, "columns"):
        columns = ";".join(str(col) for col in list(obj.columns))

    if hasattr(obj, "to_dict"):
        try:
            records = obj.to_dict("records")
            if isinstance(records, list):
                return [dict(row) for row in records], columns
        except Exception:
            pass

    if isinstance(obj, list):
        if not obj:
            return [], columns
        if all(isinstance(item, dict) for item in obj):
            return [dict(item) for item in obj], columns
        return [{"value": item} for item in obj], columns

    if isinstance(obj, tuple):
        return dataframe_like_to_records(list(obj))

    if isinstance(obj, dict):
        if all(isinstance(value, dict) for value in obj.values()):
            rows: list[dict[str, Any]] = []
            for key, value in obj.items():
                row = dict(value)
                row.setdefault("key", key)
                rows.append(row)
            return rows, columns
        return [dict(obj)], columns

    return [{"value": obj}], columns


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
        "source_pdb_path",
        "descriptor_status",
        "torsion_status",
        "ring_status",
        "linkage_id",
        "parent_residue",
        "child_residue",
        "linkage_type",
        "torsion_name",
        "torsion_angle_deg",
        "residue_index",
        "residue_name",
        "residue_iupac",
        "Q",
        "theta",
        "ring_phi",
        "ring_class",
        "raw_json",
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
        "source_pdb_path": str(path_set.pdb_path),
    }


def detect_torsion_name(column_name: str) -> str:
    clean = (
        str(column_name)
        .strip()
        .lower()
        .replace("ϕ", "phi")
        .replace("φ", "phi")
        .replace("ψ", "psi")
        .replace("ω", "omega")
    )

    if clean in {"phi", "phi_deg", "phi_angle", "phi_angle_deg"}:
        return "phi"
    if clean in {"psi", "psi_deg", "psi_angle", "psi_angle_deg"}:
        return "psi"
    if clean in {"omega", "omega_deg", "omega_angle", "omega_angle_deg"}:
        return "omega"

    if clean.endswith("_phi") or clean.endswith("_phi_deg"):
        return "phi"
    if clean.endswith("_psi") or clean.endswith("_psi_deg"):
        return "psi"
    if clean.endswith("_omega") or clean.endswith("_omega_deg"):
        return "omega"

    return ""


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lower_map = {str(key).lower(): key for key in record.keys()}
    for key in keys:
        if key in record and record[key] not in ("", None):
            return record[key]
        original = lower_map.get(key.lower())
        if original is not None and record[original] not in ("", None):
            return record[original]
    return ""


def _atom_coord_from_rows(rows: Any, atom_name: str) -> tuple[float, float, float] | None:
    selected = rows[rows["atom_name"] == atom_name]
    if selected.empty:
        return None
    row = selected.iloc[0]
    return (float(row["x"]), float(row["y"]), float(row["z"]))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def _residue_number_as_int(value: Any) -> int:
    return int(float(value))


def _infer_anomeric_form(monosaccharide: str) -> str:
    """Best-effort label used only as metadata in GlyContact output."""

    mono = str(monosaccharide or "").strip().upper()

    if mono in {"NAG", "BMA", "BGL", "BGN", "BGC"}:
        return "b"

    if mono in {"MAN", "AMA", "FUC", "AFU", "AFL"}:
        return "a"

    return "u"


def build_distance_interaction_dict(
    coordinates_df: Any,
    max_bond_distance: float = 1.85,
) -> tuple[dict[str, list[str]], list[str]]:
    """Build the minimal interaction_dict expected by GlyContact torsions.

    GlyContact can compute torsions from a coordinate DataFrame if we provide
    a glycosidic-linkage dictionary. For GlycoShield temporary PDBs,
    get_annotation(IUPAC, pdb_path) can fail even though coordinates are valid.
    This fallback detects inter-residue C1/C2--O bonds geometrically, then lets
    GlyContact compute phi/psi/omega.
    """

    warnings: list[str] = []

    if coordinates_df is None or not hasattr(coordinates_df, "groupby"):
        return {}, ["distance fallback skipped because coordinates_df is not a DataFrame"]

    required = {"residue_number", "monosaccharide", "atom_name", "x", "y", "z"}
    missing = required - set(str(col) for col in coordinates_df.columns)
    if missing:
        return {}, [f"distance fallback skipped because columns are missing: {sorted(missing)}"]

    residue_groups: dict[int, dict[str, Any]] = {}

    for residue_number, rows in coordinates_df.groupby("residue_number", sort=False):
        if rows.empty:
            continue

        mono = str(rows["monosaccharide"].iloc[0]).strip().upper()
        resnum = _residue_number_as_int(residue_number)

        atom_coords: dict[str, tuple[float, float, float]] = {}
        for _, atom in rows.iterrows():
            atom_name = str(atom["atom_name"]).strip()
            atom_coords[atom_name] = (
                float(atom["x"]),
                float(atom["y"]),
                float(atom["z"]),
            )

        residue_groups[resnum] = {
            "monosaccharide": mono,
            "atoms": atom_coords,
        }

    candidate_edges: list[tuple[float, int, str, int, str, int]] = []

    for donor_resnum, donor_info in residue_groups.items():
        donor_mono = donor_info["monosaccharide"]
        donor_atoms = donor_info["atoms"]

        # Most monosaccharides use C1 as anomeric carbon.
        # Sialic acids and fructose-like residues use C2.
        anomeric_atom = "C2" if donor_mono in {"SIA", "NGC", "0KN", "FRU", "1CU", "0CU", "4CD", "1CD"} else "C1"
        donor_c = donor_atoms.get(anomeric_atom)

        if donor_c is None:
            continue

        for acceptor_resnum, acceptor_info in residue_groups.items():
            if acceptor_resnum == donor_resnum:
                continue

            acceptor_mono = acceptor_info["monosaccharide"]
            acceptor_atoms = acceptor_info["atoms"]

            for position in (2, 3, 4, 5, 6):
                acceptor_o = acceptor_atoms.get(f"O{position}")
                acceptor_c = acceptor_atoms.get(f"C{position}")

                if acceptor_o is None or acceptor_c is None:
                    continue

                distance = _distance(donor_c, acceptor_o)

                if 1.15 <= distance <= max_bond_distance:
                    candidate_edges.append(
                        (
                            distance,
                            donor_resnum,
                            donor_mono,
                            acceptor_resnum,
                            acceptor_mono,
                            position,
                        )
                    )

    if not candidate_edges:
        return {}, ["distance fallback found no inter-residue glycosidic C1/C2--O bonds"]

    # One anomeric carbon should define at most one glycosidic linkage.
    best_by_donor: dict[int, tuple[float, int, str, int, str, int]] = {}
    for edge in sorted(candidate_edges, key=lambda item: item[0]):
        donor_resnum = edge[1]
        best_by_donor.setdefault(donor_resnum, edge)

    interaction_dict: dict[str, list[str]] = {}

    for _, donor_resnum, donor_mono, acceptor_resnum, acceptor_mono, position in best_by_donor.values():
        anomeric_form = _infer_anomeric_form(donor_mono)

        donor_key = f"{donor_resnum}_{donor_mono}"
        acceptor_key = f"{acceptor_resnum}_{acceptor_mono}"
        linkage_key = f"{donor_resnum}_({anomeric_form}1-{position})"

        interaction_dict.setdefault(donor_key, []).append(linkage_key)
        interaction_dict[linkage_key] = [acceptor_key]

    warnings.append(f"distance fallback built {len(best_by_donor)} glycosidic linkage(s)")
    return interaction_dict, warnings


def call_glycosidic_torsions(
    glycontact: Any,
    coordinates_df: Any,
    model_pdb: Path,
) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    """Call GlyContact glycosidic torsion calculation.

    First tries GlyContact's IUPAC+pdb_path annotation mode.
    If that returns empty, falls back to a minimal distance-based
    interaction_dict and still lets GlyContact compute the torsion angles.
    """

    if coordinates_df is None:
        return "not_run", [], "", ["torsions skipped because coordinates are missing"]

    fn = getattr(glycontact, "get_glycosidic_torsions", None)
    if fn is None:
        return "failed", [], "", ["GlyContact has no get_glycosidic_torsions function"]

    warnings: list[str] = []

    sequence_status, n_sequences, sequences, seq_warnings = call_get_sequences(
        glycontact,
        model_pdb,
    )
    warnings.extend(seq_warnings)

    if sequence_status == "ok" and sequences:
        glycan_iupac = sequences[0]

        try:
            result = fn(glycan_iupac, str(model_pdb))
        except Exception as exc:
            warnings.append(
                f"get_glycosidic_torsions failed with IUPAC+pdb_path: {type(exc).__name__}: {exc}"
            )
        else:
            records, columns = dataframe_like_to_records(result)
            if records:
                return "ok", records, columns, warnings
            warnings.append("IUPAC+pdb_path mode returned an empty result")

    else:
        warnings.append("IUPAC sequence unavailable, using distance fallback")

    interaction_dict, fallback_warnings = build_distance_interaction_dict(coordinates_df)
    warnings.extend(fallback_warnings)

    if not interaction_dict:
        return "empty", [], "", warnings

    try:
        result = fn(coordinates_df, interaction_dict)
    except Exception as exc:
        return "failed", [], "", [
            *warnings,
            f"get_glycosidic_torsions failed with distance interaction_dict: {type(exc).__name__}: {exc}",
        ]

    records, columns = dataframe_like_to_records(result)
    if not records:
        return "empty", [], columns, [*warnings, "distance fallback returned an empty result"]

    return "ok", records, columns, warnings

def call_ring_conformations(
    glycontact: Any,
    coordinates_df: Any,
) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    if coordinates_df is None:
        return "not_run", [], "", ["rings skipped because coordinates are missing"]

    try:
        result = glycontact.get_ring_conformations(coordinates_df)
    except Exception as exc:
        return "failed", [], "", [f"get_ring_conformations failed: {type(exc).__name__}: {exc}"]

    records, columns = dataframe_like_to_records(result)
    if not records:
        return "empty", [], columns, ["get_ring_conformations returned an empty result"]

    return "ok", records, columns, []


def expand_torsion_records(
    metadata: dict[str, Any],
    raw_records: list[dict[str, Any]],
    torsion_status: str,
    torsion_columns: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not raw_records:
        return [{
            **metadata,
            "descriptor_status": torsion_status,
            "torsion_status": torsion_status,
            "torsion_columns": torsion_columns,
            "warnings": join_warnings(warnings),
        }]

    for raw in raw_records:
        angle_items: list[tuple[str, Any]] = []
        for key, value in raw.items():
            torsion_name = detect_torsion_name(str(key))
            if torsion_name and safe_float(value) is not None:
                angle_items.append((torsion_name, value))

        common = {
            **metadata,
            "descriptor_status": torsion_status,
            "torsion_status": torsion_status,
            "torsion_columns": torsion_columns,
            "linkage_type": first_present(raw, ("linkage_type", "linkage", "bond", "glycosidic_linkage")),
            "parent_residue": first_present(raw, ("parent_residue", "parent", "donor", "from", "residue1", "residue_1")),
            "child_residue": first_present(raw, ("child_residue", "child", "acceptor", "to", "residue2", "residue_2")),
            "raw_json": raw,
            "warnings": join_warnings(warnings),
        }

        linkage_id = first_present(raw, ("linkage_id", "linkage_name", "name", "id"))
        if not linkage_id:
            linkage_id = f"{common['parent_residue']}->{common['child_residue']}:{common['linkage_type']}"
        common["linkage_id"] = linkage_id

        if angle_items:
            for torsion_name, value in angle_items:
                rows.append({
                    **common,
                    "torsion_name": torsion_name,
                    "torsion_angle_deg": fmt_float(value),
                })
        else:
            rows.append({
                **common,
                "torsion_name": first_present(raw, ("torsion_name", "torsion", "angle_name")),
                "torsion_angle_deg": fmt_float(first_present(raw, ("angle", "angle_deg", "value"))),
            })

    return rows


def expand_ring_records(
    metadata: dict[str, Any],
    raw_records: list[dict[str, Any]],
    ring_status: str,
    ring_columns: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not raw_records:
        return [{
            **metadata,
            "descriptor_status": ring_status,
            "ring_status": ring_status,
            "ring_columns": ring_columns,
            "warnings": join_warnings(warnings),
        }]

    for raw in raw_records:
        rows.append({
            **metadata,
            "descriptor_status": ring_status,
            "ring_status": ring_status,
            "ring_columns": ring_columns,
            "residue_index": first_present(raw, ("residue_index", "residue_id", "resid", "resnum", "residue_number")),
            "residue_name": first_present(raw, ("residue_name", "resname", "monosaccharide")),
            "residue_iupac": first_present(raw, ("IUPAC", "iupac", "residue_iupac")),
            "Q": fmt_float(first_present(raw, ("Q", "q", "amplitude"))),
            "theta": fmt_float(first_present(raw, ("theta", "Theta", "θ"))),
            "ring_phi": fmt_float(first_present(raw, ("phi", "Phi", "φ"))),
            "ring_class": first_present(raw, ("ring_class", "class", "conformation", "pucker", "puckering")),
            "raw_json": raw,
            "warnings": join_warnings(warnings),
        })

    return rows


def process_path_set(
    glycontact: Any,
    path_set: GlycanFramePathSet,
    temp_root: Path,
    max_models: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    torsion_rows: list[dict[str, Any]] = []
    ring_rows: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []

    prefix = f"{path_set.pdb_id}_{path_set.site}_{path_set.tool}"

    try:
        model_files = split_pdb_models_to_temp_files(
            path_set.pdb_path,
            temp_root,
            prefix=prefix,
            max_models=max_models,
        )
    except Exception as exc:
        metadata = base_metadata(path_set, model_index=0)
        warning = f"could not split PDB models: {type(exc).__name__}: {exc}"
        torsion_rows.append({**metadata, "descriptor_status": "failed", "torsion_status": "failed", "warnings": warning})
        ring_rows.append({**metadata, "descriptor_status": "failed", "ring_status": "failed", "warnings": warning})
        model_reports.append({**metadata, "coordinate_status": "not_run", "torsion_status": "failed", "ring_status": "failed"})
        return torsion_rows, ring_rows, model_reports

    for model_index, model_pdb in enumerate(model_files, start=1):
        metadata = base_metadata(path_set, model_index=model_index)

        coordinate_status, coordinates_df, n_coordinate_rows, coordinate_columns, coord_warnings = call_extract_coordinates(
            glycontact,
            model_pdb,
        )

        warnings = list(coord_warnings)

        if coordinate_status != "ok":
            torsion_status, torsion_records, torsion_columns, torsion_warnings = "not_run", [], "", [
                f"torsions skipped because coordinate_status={coordinate_status}"
            ]
            ring_status, ring_records, ring_columns, ring_warnings = "not_run", [], "", [
                f"rings skipped because coordinate_status={coordinate_status}"
            ]
        else:
            torsion_status, torsion_records, torsion_columns, torsion_warnings = call_glycosidic_torsions(
                glycontact,
                coordinates_df,
                model_pdb,
            )
            ring_status, ring_records, ring_columns, ring_warnings = call_ring_conformations(
                glycontact,
                coordinates_df,
            )

        torsion_warnings_all = warnings + torsion_warnings
        ring_warnings_all = warnings + ring_warnings

        torsion_rows.extend(
            expand_torsion_records(
                metadata,
                torsion_records,
                torsion_status=torsion_status,
                torsion_columns=torsion_columns,
                warnings=torsion_warnings_all,
            )
        )

        ring_rows.extend(
            expand_ring_records(
                metadata,
                ring_records,
                ring_status=ring_status,
                ring_columns=ring_columns,
                warnings=ring_warnings_all,
            )
        )

        model_reports.append({
            **metadata,
            "coordinate_status": coordinate_status,
            "n_coordinate_rows": n_coordinate_rows,
            "coordinate_columns": coordinate_columns,
            "torsion_status": torsion_status,
            "n_torsion_raw_rows": len(torsion_records),
            "torsion_columns": torsion_columns,
            "ring_status": ring_status,
            "n_ring_raw_rows": len(ring_records),
            "ring_columns": ring_columns,
            "warnings": join_warnings(torsion_warnings_all + ring_warnings),
        })

    return torsion_rows, ring_rows, model_reports


def build_conformation_descriptors(
    qc_path: str | Path = "results/glycan_frame_qc.csv",
    results_root: str | Path = "results",
    output_torsions: str | Path = "results/glycan_conformation_torsions.csv",
    output_rings: str | Path = "results/glycan_conformation_rings.csv",
    output_summary: str | Path = "results/glycan_conformation_descriptors_summary.json",
    max_models: int | None = 1,
    include_fragile: bool = False,
) -> dict[str, Any]:
    glycontact = import_glycontact()
    qc_rows = load_csv_rows(qc_path)

    path_sets = resolve_glycan_frame_paths(
        qc_rows,
        results_root=results_root,
        recommended_uses=DEFAULT_RECOMMENDED_USES,
        include_fragile=include_fragile,
        include_reference=True,
    )

    torsion_rows: list[dict[str, Any]] = []
    ring_rows: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []

    with TemporaryDirectory() as tmp:
        temp_root = Path(tmp)
        for path_set in path_sets:
            current_torsions, current_rings, current_reports = process_path_set(
                glycontact,
                path_set,
                temp_root=temp_root,
                max_models=max_models,
            )
            torsion_rows.extend(current_torsions)
            ring_rows.extend(current_rings)
            model_reports.extend(current_reports)

    write_csv_union(torsion_rows, output_torsions)
    write_csv_union(ring_rows, output_rings)

    summary = {
        "qc_path": str(qc_path),
        "results_root": str(results_root),
        "output_torsions": str(output_torsions),
        "output_rings": str(output_rings),
        "max_models": "all" if max_models is None else max_models,
        "n_path_sets": len(path_sets),
        "n_models_processed": len(model_reports),
        "n_torsion_rows": len(torsion_rows),
        "n_ring_rows": len(ring_rows),
        "by_tool_models": dict(Counter(report["tool"] for report in model_reports)),
        "by_coordinate_status": dict(Counter(report.get("coordinate_status", "") for report in model_reports)),
        "by_torsion_status_models": dict(Counter(report.get("torsion_status", "") for report in model_reports)),
        "by_ring_status_models": dict(Counter(report.get("ring_status", "") for report in model_reports)),
        "by_torsion_status_rows": dict(Counter(row.get("torsion_status", "") for row in torsion_rows)),
        "by_ring_status_rows": dict(Counter(row.get("ring_status", "") for row in ring_rows)),
        "warnings_examples": [
            report["warnings"]
            for report in model_reports
            if report.get("warnings")
        ][:20],
    }

    write_json(summary, output_summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build local glycan conformation descriptors with GlyContact."
    )
    parser.add_argument("--qc-path", default="results/glycan_frame_qc.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-torsions", default="results/glycan_conformation_torsions.csv")
    parser.add_argument("--output-rings", default="results/glycan_conformation_rings.csv")
    parser.add_argument("--output-summary", default="results/glycan_conformation_descriptors_summary.json")
    parser.add_argument(
        "--max-models",
        type=int,
        default=1,
        help="Maximum number of MODEL blocks per ensemble. Ignored when --all-models is used.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Process all MODEL blocks in each glycan-frame ensemble.",
    )
    parser.add_argument(
        "--include-fragile",
        action="store_true",
        help="Also include exploratory_fragile_exclude_strict rows from glycan-frame QC.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    max_models = None if args.all_models else args.max_models

    summary = build_conformation_descriptors(
        qc_path=args.qc_path,
        results_root=args.results_root,
        output_torsions=args.output_torsions,
        output_rings=args.output_rings,
        output_summary=args.output_summary,
        max_models=max_models,
        include_fragile=args.include_fragile,
    )

    print(f"[OK] wrote {args.output_torsions} ({summary['n_torsion_rows']} rows)")
    print(f"[OK] wrote {args.output_rings} ({summary['n_ring_rows']} rows)")
    print(f"[OK] wrote {args.output_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
