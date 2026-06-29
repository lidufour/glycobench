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

from glycobench.mapping.glycan_residue_mapping import ResidueKey, group_atoms_by_residue
from glycobench.mapping.protein_frame_alignment import Atom, parse_models, read_text_maybe_zip


DEFAULT_RECOMMENDED_USES = ("strict", "exploratory_common_scaffold")
TOOL_NAMES = ("reference", "glycoshield", "glycoshape")

# Minimal residue-name normalization needed before calling GlyContact.
# GlyContact 0.3.4 parses either ATOM-only glycan PDB files or HETATM
# glycan records in a protein-containing PDB. Our glycan-frame files are
# glycan-only, and experimental references are HETATM-only. GlycoShield also
# uses several CHARMM-like residue names that are not all accepted by
# GlyContact's PDB parser. These substitutions are applied only to temporary
# files used by the adapter; original GlycoBench outputs are never modified.
GLYCONTACT_RESNAME_MAP: dict[str, str] = {
    # GlcNAc-like
    "BGL": "NAG",
    "BGN": "NAG",
    "BGC": "GLC",
    # Mannose-like
    "AMA": "MAN",
    "BMA": "BMA",
    # Fucose-like
    "AFU": "FUC",
    "AFL": "FUC",
    # Galactose / GalNAc-like
    "BGA": "GAL",
    "AGA": "A2G",
}


@dataclass(frozen=True)
class GlycanFramePathSet:
    pdb_id: str
    site: str
    tool: str
    recommended_use: str
    scope: str
    analysis_group: str
    glycan_class: str
    glycan_name: str
    glytoucan: str
    pdb_path: Path
    reference_pdb_path: Path
    source_qc_row: dict[str, str]


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("")
        return output_path

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_json(data: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2)
    return output_path


def as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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


def join_warnings(warnings: Iterable[str], limit: int = 8) -> str:
    values = [str(item) for item in warnings if str(item)]
    if not values:
        return ""
    selected = values[:limit]
    suffix = "" if len(values) <= limit else f" | ... +{len(values) - limit} more"
    return " | ".join(selected) + suffix


def import_glycontact() -> Any:
    try:
        import glycontact  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "GlyContact is not importable in this environment. "
            "Run this adapter with `uv run ...` inside the glycobench environment."
        ) from exc
    return glycontact


def glycan_frame_dir_from_row(row: dict[str, str], results_root: str | Path) -> Path:
    report_path = row.get("report_path") or ""
    if report_path:
        return Path(report_path).parent
    return Path(results_root) / row["pdb_id"] / "aligned" / row["site"] / "glycan_frame"


def resolve_tool_pdb_path(row: dict[str, str], results_root: str | Path) -> Path:
    frame_dir = glycan_frame_dir_from_row(row, results_root=results_root)
    tool = row["tool"]
    if tool == "reference":
        return frame_dir / "reference_glycan.pdb"
    return frame_dir / f"{tool}_glycan_ensemble.pdb"


def resolve_reference_pdb_path(row: dict[str, str], results_root: str | Path) -> Path:
    frame_dir = glycan_frame_dir_from_row(row, results_root=results_root)
    return frame_dir / "reference_glycan.pdb"


def should_keep_qc_row(
    row: dict[str, str],
    recommended_uses: set[str],
    include_fragile: bool = False,
) -> bool:
    if not as_bool(row.get("tool_available")):
        return False

    use = row.get("recommended_use", "")
    if use in recommended_uses:
        return True

    if include_fragile and use == "exploratory_fragile_exclude_strict":
        return True

    return False


def resolve_glycan_frame_paths(
    qc_rows: list[dict[str, str]],
    results_root: str | Path = "results",
    recommended_uses: Iterable[str] = DEFAULT_RECOMMENDED_USES,
    include_fragile: bool = False,
    include_reference: bool = True,
) -> list[GlycanFramePathSet]:
    """Resolve aligned glycan-frame PDB files to probe with GlyContact.

    QC rows contain one row per tool. This function also adds one reference row
    per selected site, because the experimental glycan is the anchor for all
    subsequent comparisons.
    """

    recommended_use_set = set(recommended_uses)
    selected_rows = [
        row
        for row in qc_rows
        if should_keep_qc_row(row, recommended_use_set, include_fragile=include_fragile)
    ]

    path_sets: list[GlycanFramePathSet] = []
    reference_seen: set[tuple[str, str]] = set()

    for row in selected_rows:
        pdb_id = row["pdb_id"]
        site = row["site"]
        reference_pdb = resolve_reference_pdb_path(row, results_root=results_root)

        if include_reference and (pdb_id, site) not in reference_seen:
            reference_seen.add((pdb_id, site))
            path_sets.append(
                GlycanFramePathSet(
                    pdb_id=pdb_id,
                    site=site,
                    tool="reference",
                    recommended_use="reference_for_selected_site",
                    scope=row.get("scope", ""),
                    analysis_group=row.get("analysis_group", ""),
                    glycan_class=row.get("glycan_class", ""),
                    glycan_name=row.get("glycan_name", ""),
                    glytoucan=row.get("glytoucan", ""),
                    pdb_path=reference_pdb,
                    reference_pdb_path=reference_pdb,
                    source_qc_row=row,
                )
            )

        path_sets.append(
            GlycanFramePathSet(
                pdb_id=pdb_id,
                site=site,
                tool=row["tool"],
                recommended_use=row.get("recommended_use", ""),
                scope=row.get("scope", ""),
                analysis_group=row.get("analysis_group", ""),
                glycan_class=row.get("glycan_class", ""),
                glycan_name=row.get("glycan_name", ""),
                glytoucan=row.get("glytoucan", ""),
                pdb_path=resolve_tool_pdb_path(row, results_root=results_root),
                reference_pdb_path=reference_pdb,
                source_qc_row=row,
            )
        )

    return path_sets


def count_models_and_first_model_atoms(pdb_path: str | Path) -> dict[str, Any]:
    text = read_text_maybe_zip(pdb_path)
    models = parse_models(text)

    if not models:
        return {
            "n_models": 0,
            "n_atoms_first_model": 0,
            "n_residues_first_model": 0,
        }

    first_model = sorted(models)[0]
    atoms = models[first_model]
    residues = group_atoms_by_residue(atoms)

    return {
        "n_models": len(models),
        "n_atoms_first_model": len(atoms),
        "n_residues_first_model": len(residues),
    }


def _model_blocks_from_pdb_text(text: str) -> list[list[str]]:
    lines = text.splitlines()
    has_model_records = any(line.startswith("MODEL") for line in lines)

    if not has_model_records:
        return [lines]

    blocks: list[list[str]] = []
    current: list[str] = []
    inside_model = False

    for line in lines:
        if line.startswith("MODEL"):
            if inside_model and current:
                blocks.append(current)
            current = []
            inside_model = True
            continue

        if line.startswith("ENDMDL"):
            if inside_model:
                blocks.append(current)
                current = []
                inside_model = False
            continue

        if inside_model:
            current.append(line)

    if inside_model and current:
        blocks.append(current)

    return blocks


def _normalize_pdb_line_for_glycontact(line: str) -> str:
    """Return a PDB line compatible with GlyContact's local PDB parser.

    The adapter only writes normalized *temporary* files. The scientific source
    files generated by previous GlycoBench layers are left unchanged.
    """

    if not line.startswith(("ATOM", "HETATM")):
        return line.rstrip()

    out = line.rstrip()

    # GlyContact 0.3.4 reads ATOM records when a file does not contain explicit
    # protein markers. Our isolated experimental glycans are HETATM-only, so we
    # convert them to ATOM in the temporary copy.
    if out.startswith("HETATM"):
        out = "ATOM  " + out[6:]

    if len(out) >= 20:
        resname = out[17:20].strip().upper()
        normalized = GLYCONTACT_RESNAME_MAP.get(resname)
        if normalized:
            out = f"{out[:17]}{normalized:>3s}{out[20:]}"

    return out


def split_pdb_models_to_temp_files(
    pdb_path: str | Path,
    temp_dir: str | Path,
    prefix: str,
    max_models: int | None = 1,
) -> list[Path]:
    """Write one GlyContact-compatible temporary PDB file per MODEL block.

    GlyContact functions are safer on single-model PDB files. This helper keeps
    the first ``max_models`` blocks and writes them as plain PDB files without
    MODEL/ENDMDL wrappers. During this temporary write, HETATM-only glycans are
    converted to ATOM records and a few GlycoShield residue names are translated
    to residue names accepted by GlyContact.
    """

    text = read_text_maybe_zip(pdb_path)
    blocks = _model_blocks_from_pdb_text(text)

    if max_models is not None:
        blocks = blocks[:max_models]

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    out_paths: list[Path] = []
    safe_prefix = prefix.replace("/", "_").replace(" ", "_")

    for index, block in enumerate(blocks, start=1):
        out_path = temp_dir / f"{safe_prefix}_model_{index:04d}.pdb"
        cleaned_lines = [
            _normalize_pdb_line_for_glycontact(line)
            for line in block
            if line.rstrip()
        ]
        if not cleaned_lines or cleaned_lines[-1] != "END":
            cleaned_lines.append("END")
        out_path.write_text("\n".join(cleaned_lines) + "\n")
        out_paths.append(out_path)

    return out_paths


def status_from_exception_free_results(ok_count: int, empty_count: int, failed_count: int) -> str:
    if ok_count > 0 and failed_count == 0 and empty_count == 0:
        return "ok"
    if ok_count > 0:
        return "partial"
    if empty_count > 0 and failed_count == 0:
        return "empty"
    if failed_count > 0:
        return "failed"
    return "not_run"


def call_get_sequences(glycontact: Any, pdb_path: Path) -> tuple[str, int, list[str], list[str]]:
    warnings: list[str] = []
    sequences: list[str] = []

    try:
        raw = glycontact.get_glycan_sequences_from_pdb(str(pdb_path))
    except Exception as exc:
        return "failed", 0, [], [f"get_glycan_sequences_from_pdb failed: {type(exc).__name__}: {exc}"]

    if raw is None:
        return "empty", 0, [], ["get_glycan_sequences_from_pdb returned None"]

    if isinstance(raw, str):
        sequences = [raw]
    else:
        try:
            sequences = [str(item) for item in raw]
        except TypeError:
            sequences = [str(raw)]

    sequences = [seq for seq in sequences if seq and seq.lower() != "none"]
    status = "ok" if sequences else "empty"
    if not sequences:
        warnings.append("no IUPAC sequence detected by GlyContact")

    return status, len(sequences), sequences, warnings


def _iupac_label_from_resname(glycontact: Any, resname: Any) -> str:
    clean = str(resname or "").strip().upper()
    raw = getattr(glycontact, "map_dict", {}).get(clean, clean)
    label = str(raw).strip()
    # GlyContact's internal map stores strings such as "GlcNAc(b". The
    # puckering code only needs broad labels to detect sialic acids/furanoses.
    return label.replace("(?", "").replace("(a", "").replace("(b", "").strip() or clean


def ensure_iupac_column(glycontact: Any, df: Any) -> Any:
    """Return a copy of ``df`` with the IUPAC column expected by GlyContact.

    ``extract_3D_coordinates`` in GlyContact 0.3.4 returns a column named
    ``monosaccharide`` but ``calculate_ring_pucker`` indexes ``IUPAC``. This
    small compatibility shim keeps the adapter robust without modifying
    GlyContact itself.
    """

    if df is None or not hasattr(df, "columns") or "IUPAC" in df.columns:
        return df

    if "monosaccharide" not in df.columns:
        return df

    out = df.copy()
    out["IUPAC"] = out["monosaccharide"].map(lambda name: _iupac_label_from_resname(glycontact, name))
    return out


def call_extract_coordinates(glycontact: Any, pdb_path: Path) -> tuple[str, Any | None, int, str, list[str]]:
    try:
        df = glycontact.extract_3D_coordinates(str(pdb_path))
    except Exception as exc:
        return "failed", None, 0, "", [f"extract_3D_coordinates failed: {type(exc).__name__}: {exc}"]

    if df is None:
        return "empty", None, 0, "", ["extract_3D_coordinates returned None"]

    df = ensure_iupac_column(glycontact, df)

    try:
        n_rows = len(df)
    except TypeError:
        n_rows = 0

    columns = ""
    if hasattr(df, "columns"):
        columns = ";".join(str(col) for col in list(df.columns))

    if n_rows == 0:
        return "empty", df, 0, columns, ["extract_3D_coordinates returned an empty table"]

    return "ok", df, n_rows, columns, []


def call_ring_conformations(glycontact: Any, coordinates_df: Any | None) -> tuple[str, int, str, list[str]]:
    if coordinates_df is None:
        return "not_run", 0, "", ["ring conformations skipped because coordinates are missing"]

    try:
        ring_df = glycontact.get_ring_conformations(coordinates_df)
    except Exception as exc:
        return "failed", 0, "", [f"get_ring_conformations failed: {type(exc).__name__}: {exc}"]

    if ring_df is None:
        return "empty", 0, "", ["get_ring_conformations returned None"]

    try:
        n_rows = len(ring_df)
    except TypeError:
        n_rows = 0

    columns = ""
    if hasattr(ring_df, "columns"):
        columns = ";".join(str(col) for col in list(ring_df.columns))

    if n_rows == 0:
        return "empty", 0, columns, ["get_ring_conformations returned an empty table"]

    return "ok", n_rows, columns, []


def call_superimpose(
    glycontact: Any,
    reference_model_pdb: Path,
    mobile_model_pdb: Path,
    fast: bool = True,
) -> tuple[str, float | None, list[str]]:
    try:
        result = glycontact.superimpose_glycans(
            str(reference_model_pdb),
            str(mobile_model_pdb),
            fast=fast,
        )
    except Exception as exc:
        return "failed", None, [f"superimpose_glycans failed: {type(exc).__name__}: {exc}"]

    if not isinstance(result, dict):
        return "failed", None, [f"superimpose_glycans returned unexpected object: {type(result).__name__}"]

    rmsd = safe_float(result.get("rmsd"))
    if rmsd is None:
        return "failed", None, ["superimpose_glycans did not return a numeric rmsd"]

    return "ok", rmsd, []


def probe_glycontact_on_model(
    glycontact: Any,
    model_pdb: Path,
) -> dict[str, Any]:
    warnings: list[str] = []

    sequence_status, n_sequences, sequences, seq_warnings = call_get_sequences(glycontact, model_pdb)
    warnings.extend(seq_warnings)

    coordinate_status, coordinates_df, n_coordinate_rows, coordinate_columns, coord_warnings = call_extract_coordinates(
        glycontact,
        model_pdb,
    )
    warnings.extend(coord_warnings)

    ring_status, n_ring_rows, ring_columns, ring_warnings = call_ring_conformations(
        glycontact,
        coordinates_df,
    )
    warnings.extend(ring_warnings)

    return {
        "model_pdb": str(model_pdb),
        "sequence_status": sequence_status,
        "n_sequences": n_sequences,
        "sequences": sequences,
        "coordinate_status": coordinate_status,
        "n_coordinate_rows": n_coordinate_rows,
        "coordinate_columns": coordinate_columns,
        "ring_status": ring_status,
        "n_ring_rows": n_ring_rows,
        "ring_columns": ring_columns,
        "warnings": warnings,
    }


def summarize_model_probes(model_reports: list[dict[str, Any]], field: str) -> str:
    ok_count = sum(1 for report in model_reports if report.get(field) == "ok")
    empty_count = sum(1 for report in model_reports if report.get(field) == "empty")
    failed_count = sum(1 for report in model_reports if report.get(field) == "failed")
    return status_from_exception_free_results(ok_count, empty_count, failed_count)


def unique_sequences(model_reports: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for report in model_reports:
        for sequence in report.get("sequences") or []:
            if sequence not in seen:
                seen.add(sequence)
                out.append(sequence)
    return out


def summarize_superimposition(superimpose_reports: list[dict[str, Any]]) -> dict[str, str]:
    if not superimpose_reports:
        return {
            "superimpose_to_reference_status": "not_run",
            "n_superimpose_ok": "0",
            "n_superimpose_failed": "0",
            "min_rmsd_to_reference": "",
            "mean_rmsd_to_reference": "",
            "median_rmsd_to_reference": "",
        }

    rmsds = [report["rmsd"] for report in superimpose_reports if report.get("status") == "ok" and report.get("rmsd") is not None]
    failed_count = sum(1 for report in superimpose_reports if report.get("status") == "failed")

    if rmsds and failed_count == 0:
        status = "ok"
    elif rmsds:
        status = "partial"
    else:
        status = "failed"

    return {
        "superimpose_to_reference_status": status,
        "n_superimpose_ok": str(len(rmsds)),
        "n_superimpose_failed": str(failed_count),
        "min_rmsd_to_reference": fmt_float(min(rmsds) if rmsds else None),
        "mean_rmsd_to_reference": fmt_float(statistics.mean(rmsds) if rmsds else None),
        "median_rmsd_to_reference": fmt_float(statistics.median(rmsds) if rmsds else None),
    }


def probe_site_tool(
    path_set: GlycanFramePathSet,
    glycontact: Any | None = None,
    max_models: int | None = 1,
    superimpose_fast: bool = True,
) -> dict[str, Any]:
    """Probe one reference/tool glycan-frame PDB with GlyContact."""

    if glycontact is None:
        glycontact = import_glycontact()

    base_row: dict[str, Any] = {
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
        "pdb_exists": str(path_set.pdb_path.exists()),
        "reference_pdb_path": str(path_set.reference_pdb_path),
        "reference_pdb_exists": str(path_set.reference_pdb_path.exists()),
        "n_models_total": "",
        "n_models_tested": "0",
        "n_atoms_first_model": "",
        "n_residues_first_model": "",
        "glycontact_version": str(getattr(glycontact, "__version__", "unknown")),
        "sequence_status": "not_run",
        "n_unique_sequences": "0",
        "first_sequence": "",
        "coordinate_status": "not_run",
        "n_coordinate_rows_first_tested_model": "",
        "ring_status": "not_run",
        "n_ring_rows_first_tested_model": "",
        "coordinate_columns_first_tested_model": "",
        "ring_columns_first_tested_model": "",
        "superimpose_to_reference_status": "not_run",
        "n_superimpose_ok": "0",
        "n_superimpose_failed": "0",
        "min_rmsd_to_reference": "",
        "mean_rmsd_to_reference": "",
        "median_rmsd_to_reference": "",
        "adapter_status": "not_run",
        "warnings": "",
    }

    warnings: list[str] = []

    if not path_set.pdb_path.exists():
        base_row["adapter_status"] = "failed"
        base_row["warnings"] = f"missing PDB file: {path_set.pdb_path}"
        return base_row

    if not path_set.reference_pdb_path.exists():
        warnings.append(f"missing reference PDB file: {path_set.reference_pdb_path}")

    try:
        counts = count_models_and_first_model_atoms(path_set.pdb_path)
        base_row["n_models_total"] = str(counts["n_models"])
        base_row["n_atoms_first_model"] = str(counts["n_atoms_first_model"])
        base_row["n_residues_first_model"] = str(counts["n_residues_first_model"])
    except Exception as exc:
        base_row["adapter_status"] = "failed"
        base_row["warnings"] = f"internal PDB parser failed: {type(exc).__name__}: {exc}"
        return base_row

    with TemporaryDirectory(prefix="glycobench_glycontact_") as tmp:
        tmp_dir = Path(tmp)
        model_paths = split_pdb_models_to_temp_files(
            path_set.pdb_path,
            tmp_dir,
            prefix=f"{path_set.pdb_id}_{path_set.site}_{path_set.tool}",
            max_models=max_models,
        )
        base_row["n_models_tested"] = str(len(model_paths))

        if not model_paths:
            base_row["adapter_status"] = "failed"
            base_row["warnings"] = "no model file could be created from PDB"
            return base_row

        model_reports = [probe_glycontact_on_model(glycontact, model_pdb) for model_pdb in model_paths]

        model_warnings: list[str] = []
        for index, report in enumerate(model_reports, start=1):
            for warning in report.get("warnings") or []:
                model_warnings.append(f"model {index}: {warning}")
        warnings.extend(model_warnings)

        sequences = unique_sequences(model_reports)
        first_report = model_reports[0]

        base_row["sequence_status"] = summarize_model_probes(model_reports, "sequence_status")
        base_row["n_unique_sequences"] = str(len(sequences))
        base_row["first_sequence"] = sequences[0] if sequences else ""
        base_row["coordinate_status"] = summarize_model_probes(model_reports, "coordinate_status")
        base_row["n_coordinate_rows_first_tested_model"] = str(first_report.get("n_coordinate_rows", ""))
        base_row["coordinate_columns_first_tested_model"] = str(first_report.get("coordinate_columns", ""))
        base_row["ring_status"] = summarize_model_probes(model_reports, "ring_status")
        base_row["n_ring_rows_first_tested_model"] = str(first_report.get("n_ring_rows", ""))
        base_row["ring_columns_first_tested_model"] = str(first_report.get("ring_columns", ""))

        superimpose_reports: list[dict[str, Any]] = []
        if path_set.tool != "reference" and path_set.reference_pdb_path.exists():
            reference_model_paths = split_pdb_models_to_temp_files(
                path_set.reference_pdb_path,
                tmp_dir,
                prefix=f"{path_set.pdb_id}_{path_set.site}_reference",
                max_models=1,
            )
            if reference_model_paths:
                reference_model_pdb = reference_model_paths[0]
                for index, model_pdb in enumerate(model_paths, start=1):
                    status, rmsd, rmsd_warnings = call_superimpose(
                        glycontact,
                        reference_model_pdb=reference_model_pdb,
                        mobile_model_pdb=model_pdb,
                        fast=superimpose_fast,
                    )
                    superimpose_reports.append({"model": index, "status": status, "rmsd": rmsd})
                    warnings.extend([f"model {index}: {warning}" for warning in rmsd_warnings])
            else:
                warnings.append("reference model split produced no temporary file")

        base_row.update(summarize_superimposition(superimpose_reports))

    # The adapter is considered usable if GlyContact can at least extract atoms.
    # Ring pucker and sequence recovery are useful diagnostics, but they are not
    # required for the adapter itself to be usable by the next descriptor layer.
    if base_row["coordinate_status"] == "ok":
        base_row["adapter_status"] = "ok"
    elif base_row["coordinate_status"] == "partial":
        base_row["adapter_status"] = "partial"
    else:
        base_row["adapter_status"] = "failed"

    base_row["warnings"] = join_warnings(warnings)
    return base_row


def summarize_adapter_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "by_tool": {},
        "by_adapter_status": {},
        "by_recommended_use": {},
        "by_coordinate_status": {},
        "by_ring_status": {},
        "by_sequence_status": {},
        "by_superimpose_status": {},
    }

    for row in rows:
        for field, key in [
            ("tool", "by_tool"),
            ("adapter_status", "by_adapter_status"),
            ("recommended_use", "by_recommended_use"),
            ("coordinate_status", "by_coordinate_status"),
            ("ring_status", "by_ring_status"),
            ("sequence_status", "by_sequence_status"),
            ("superimpose_to_reference_status", "by_superimpose_status"),
        ]:
            value = str(row.get(field, ""))
            summary[key][value] = summary[key].get(value, 0) + 1

    return summary


def build_glycontact_adapter_qc(
    glycan_frame_qc_path: str | Path = "results/glycan_frame_qc.csv",
    results_root: str | Path = "results",
    recommended_uses: Iterable[str] = DEFAULT_RECOMMENDED_USES,
    include_fragile: bool = False,
    include_reference: bool = True,
    max_models: int | None = 1,
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

    return [
        probe_site_tool(
            path_set,
            glycontact=glycontact,
            max_models=max_models,
            superimpose_fast=superimpose_fast,
        )
        for path_set in path_sets
    ]


def write_glycontact_adapter_qc(
    rows: list[dict[str, Any]],
    output_csv: str | Path = "results/glycontact_adapter_qc.csv",
    output_summary_json: str | Path = "results/glycontact_adapter_qc_summary.json",
) -> tuple[Path, Path]:
    csv_path = write_csv_rows(rows, output_csv)
    summary_path = write_json(summarize_adapter_rows(rows), output_summary_json)
    return csv_path, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe aligned glycan-frame PDB files with GlyContact and write an adapter QC table.",
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
        default="results/glycontact_adapter_qc.csv",
        help="Output adapter QC CSV.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="results/glycontact_adapter_qc_summary.json",
        help="Output adapter QC summary JSON.",
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
        help="Number of models to probe per ensemble PDB. Use 0 with --all-models instead of this option.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Probe all models in each ensemble PDB.",
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

    rows = build_glycontact_adapter_qc(
        glycan_frame_qc_path=args.glycan_frame_qc,
        results_root=args.results_root,
        recommended_uses=recommended_uses,
        include_fragile=args.include_fragile,
        include_reference=not args.no_reference,
        max_models=max_models,
        superimpose_fast=not args.slow_superimpose,
    )
    csv_path, summary_path = write_glycontact_adapter_qc(
        rows,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )

    summary = summarize_adapter_rows(rows)
    print(f"[OK] wrote {csv_path} ({len(rows)} rows)")
    print(f"[OK] wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
