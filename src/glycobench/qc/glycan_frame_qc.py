from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import statistics


TOOL_NAMES = ("glycoshield", "glycoshape")


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def safe_round(value: Any, ndigits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(round(float(value), ndigits))
    except (TypeError, ValueError):
        return ""


def join_warnings(warnings: list[str] | None, limit: int | None = None) -> str:
    if not warnings:
        return ""
    selected = warnings if limit is None else warnings[:limit]
    suffix = "" if limit is None or len(warnings) <= limit else f" | ... +{len(warnings) - limit} more"
    return " | ".join(selected) + suffix


def rmsd_std_from_models(models: list[dict[str, Any]]) -> str:
    values = [float(model["centroid_rmsd"]) for model in models if "centroid_rmsd" in model]
    if len(values) < 2:
        return ""
    return str(round(statistics.pstdev(values), 6))


def infer_alignment_quality(tool_report: dict[str, Any]) -> str:
    models = tool_report.get("models") or []
    qualities = {model.get("alignment_quality") for model in models if model.get("alignment_quality")}
    if "fragile_two_centroids" in qualities:
        return "fragile_two_centroids"
    if "robust_3d" in qualities:
        return "robust_3d"
    return ""


def missing_or_extra_note(mapping_warnings: list[str]) -> str:
    has_missing = any("no source child found" in warning for warning in mapping_warnings)
    has_extra = any("extra source child not mapped" in warning for warning in mapping_warnings)
    has_subtree = any("subtree mismatch" in warning for warning in mapping_warnings)

    parts: list[str] = []
    if has_missing:
        parts.append("reference branch absent in source")
    if has_extra:
        parts.append("extra source branch not mapped")
    if has_subtree:
        parts.append("local subtree differs but common edge/name was usable")
    return "; ".join(parts)


def build_qc_note(
    row: dict[str, str],
    tool_name: str,
    tool_report: dict[str, Any],
    report_exists: bool,
) -> str:
    if not report_exists:
        return "missing glycan-frame alignment report"

    if tool_report.get("status") == "not_available":
        return "tool not available for this site"

    notes: list[str] = []
    is_secondary = as_bool(row.get("use_for_glycoshield_secondary")) and not as_bool(row.get("use_for_triple_benchmark"))
    n_mapped = tool_report.get("n_mapped_residues")
    n_ref = tool_report.get("n_reference_residues")
    mapping_status = tool_report.get("mapping_status")
    tool_status = tool_report.get("status")
    mapping_warnings = tool_report.get("mapping_warnings") or []

    if is_secondary and tool_name == "glycoshield":
        notes.append("GlycoShield-only secondary site")

    if mapping_status == "partial" or tool_status == "partial":
        if n_mapped is not None and n_ref is not None:
            notes.append(f"partial residue mapping ({n_mapped}/{n_ref}); RMSD uses common scaffold only")
        else:
            notes.append("partial residue mapping; RMSD uses common scaffold only")
        detail = missing_or_extra_note(mapping_warnings)
        if detail:
            notes.append(detail)

    if tool_status == "fragile" or infer_alignment_quality(tool_report) == "fragile_two_centroids":
        notes.append("fragile alignment: only two mapped centroids, so rotation around the inter-centroid axis is underdetermined")

    if tool_status == "failed":
        notes.append("alignment failed")

    if not notes and as_bool(row.get("use_for_triple_benchmark")):
        notes.append("strict triple-core glycan-frame comparison")

    return "; ".join(notes)


def recommended_use(row: dict[str, str], tool_report: dict[str, Any], report_exists: bool) -> str:
    if not report_exists or tool_report.get("status") in {"failed", "skipped", "not_available"}:
        return "exclude_missing_or_failed"

    status = tool_report.get("status")
    n_mapped = tool_report.get("n_mapped_residues") or 0
    is_secondary = as_bool(row.get("use_for_glycoshield_secondary")) and not as_bool(row.get("use_for_triple_benchmark"))

    if status == "fragile" or n_mapped < 3:
        return "exploratory_fragile_exclude_strict"
    if status == "partial":
        return "exploratory_common_scaffold"
    if is_secondary:
        return "secondary_glycoshield_exploratory"
    return "strict"


def row_scope(row: dict[str, str]) -> str:
    if as_bool(row.get("use_for_triple_benchmark")):
        return "triple_core"
    if as_bool(row.get("use_for_glycoshield_secondary")):
        return "glycoshield_secondary"
    return row.get("analysis_group") or "other"


def expected_tools_for_row(row: dict[str, str], include_secondary: bool) -> list[str]:
    if as_bool(row.get("use_for_triple_benchmark")):
        return ["glycoshield", "glycoshape"]
    if include_secondary and as_bool(row.get("use_for_glycoshield_secondary")):
        return ["glycoshield"]
    return []


def build_qc_rows(
    manifest_path: str | Path = "results/analysis_manifest.csv",
    results_root: str | Path = "results",
    include_secondary: bool = False,
) -> list[dict[str, str]]:
    results_root = Path(results_root)
    manifest_rows = load_csv_rows(manifest_path)
    qc_rows: list[dict[str, str]] = []

    for row in manifest_rows:
        tools = expected_tools_for_row(row, include_secondary=include_secondary)
        if not tools:
            continue

        pdb_id = row["pdb_id"]
        site = row["site"]
        report_path = results_root / pdb_id / "aligned" / site / "glycan_frame" / "glycan_frame_alignment_report.json"
        report_exists = report_path.exists()
        report = load_json(report_path) if report_exists else {}

        for tool_name in tools:
            tool_available = bool(row.get(f"{tool_name}_glycan_ensemble_pdb"))
            tool_report = (report.get("tools") or {}).get(tool_name)
            if tool_report is None:
                tool_report = {
                    "tool": tool_name,
                    "status": "not_available" if not tool_available else "missing_report",
                    "warnings": [],
                    "mapping_warnings": [],
                }

            models = tool_report.get("models") or []
            qc_rows.append(
                {
                    "pdb_id": pdb_id,
                    "site": site,
                    "scope": row_scope(row),
                    "analysis_group": row.get("analysis_group", ""),
                    "glycan_class": row.get("glycan_class", ""),
                    "glycan_name": row.get("glycan_name", ""),
                    "glytoucan": row.get("glytoucan", ""),
                    "tool": tool_name,
                    "tool_available": str(tool_available),
                    "site_status": report.get("status", "missing_report" if not report_exists else ""),
                    "tool_status": tool_report.get("status", ""),
                    "mapping_status": tool_report.get("mapping_status", ""),
                    "alignment_quality": infer_alignment_quality(tool_report),
                    "n_reference_residues": str(tool_report.get("n_reference_residues", "")),
                    "n_source_residues": str(tool_report.get("n_source_residues", "")),
                    "n_mapped_residues": str(tool_report.get("n_mapped_residues", "")),
                    "mapping_fraction": safe_round(
                        (float(tool_report.get("n_mapped_residues")) / float(tool_report.get("n_reference_residues")))
                        if tool_report.get("n_mapped_residues") is not None and tool_report.get("n_reference_residues")
                        else None,
                        ndigits=4,
                    ),
                    "n_models_input": str(tool_report.get("n_models_input", "")),
                    "n_models_aligned": str(tool_report.get("n_models_aligned", "")),
                    "mean_centroid_rmsd": safe_round(tool_report.get("mean_centroid_rmsd")),
                    "median_centroid_rmsd": safe_round(tool_report.get("median_centroid_rmsd")),
                    "min_centroid_rmsd": safe_round(tool_report.get("min_centroid_rmsd")),
                    "max_centroid_rmsd": safe_round(tool_report.get("max_centroid_rmsd")),
                    "std_centroid_rmsd": rmsd_std_from_models(models),
                    "recommended_use": recommended_use(row, tool_report, report_exists=report_exists),
                    "qc_note": build_qc_note(row, tool_name, tool_report, report_exists=report_exists),
                    "n_tool_warnings": str(len(tool_report.get("warnings") or [])),
                    "n_mapping_warnings": str(len(tool_report.get("mapping_warnings") or [])),
                    "tool_warnings": join_warnings(tool_report.get("warnings") or [], limit=3),
                    "mapping_warnings": join_warnings(tool_report.get("mapping_warnings") or [], limit=3),
                    "report_path": str(report_path),
                }
            )

    return qc_rows


def write_qc_csv(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pdb_id",
        "site",
        "scope",
        "analysis_group",
        "glycan_class",
        "glycan_name",
        "glytoucan",
        "tool",
        "tool_available",
        "site_status",
        "tool_status",
        "mapping_status",
        "alignment_quality",
        "n_reference_residues",
        "n_source_residues",
        "n_mapped_residues",
        "mapping_fraction",
        "n_models_input",
        "n_models_aligned",
        "mean_centroid_rmsd",
        "median_centroid_rmsd",
        "min_centroid_rmsd",
        "max_centroid_rmsd",
        "std_centroid_rmsd",
        "recommended_use",
        "qc_note",
        "n_tool_warnings",
        "n_mapping_warnings",
        "tool_warnings",
        "mapping_warnings",
        "report_path",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def summarize_qc_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "by_scope": {},
        "by_tool_status": {},
        "by_recommended_use": {},
    }

    for row in rows:
        scope = row["scope"]
        tool_status_key = f"{row['tool']}:{row['tool_status']}"
        use = row["recommended_use"]
        summary["by_scope"][scope] = summary["by_scope"].get(scope, 0) + 1
        summary["by_tool_status"][tool_status_key] = summary["by_tool_status"].get(tool_status_key, 0) + 1
        summary["by_recommended_use"][use] = summary["by_recommended_use"].get(use, 0) + 1

    return summary
