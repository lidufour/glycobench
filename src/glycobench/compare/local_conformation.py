from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from glycobench.compare.common import (
    TOOL_REFERENCE,
    circular_distance_deg,
    circular_mean_deg,
    first_existing_column,
    optional_column,
    read_csv,
    weighted_mean,
    weighted_median,
    write_csv,
)


OK_STATUSES = {"ok", "self_reference"}


LINKAGE_RE = re.compile(
    r".*?:?(?P<child_num>\d+)_(?P<child_name>[A-Za-z0-9]+)-"
    r"(?P<parent_num>\d+)_(?P<parent_name>[A-Za-z0-9]+)"
)


def _json_field(raw: object, field: str) -> object:
    if pd.isna(raw):
        return np.nan

    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return np.nan

    return data.get(field, np.nan)


def _linkage_match_key(row: pd.Series, linkage_col: str) -> str:
    linkage = str(row[linkage_col])
    match = LINKAGE_RE.match(linkage)

    if match is None:
        return linkage

    child_name = match.group("child_name")
    parent_name = match.group("parent_name")

    anomeric = row.get("_anomeric_form", np.nan)
    position = row.get("_linkage_position", np.nan)

    anomeric_part = "na" if pd.isna(anomeric) else str(anomeric)
    position_part = "na" if pd.isna(position) else str(int(float(position)))

    return f"{child_name}-{parent_name}_{anomeric_part}_{position_part}"


def _add_torsion_match_key(
    df: pd.DataFrame,
    cols: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = df.copy()

    raw_col = optional_column(df, ["raw_json"])

    if raw_col is not None:
        df["_anomeric_form"] = df[raw_col].apply(
            lambda value: _json_field(value, "anomeric_form")
        )
        df["_linkage_position"] = df[raw_col].apply(
            lambda value: _json_field(value, "position")
        )
    else:
        df["_anomeric_form"] = np.nan
        df["_linkage_position"] = np.nan

    df["_linkage_match_key"] = df.apply(
        lambda row: _linkage_match_key(row, cols["linkage"]),
        axis=1,
    )

    cols = cols.copy()
    cols["linkage_original"] = cols["linkage"]
    cols["linkage"] = "_linkage_match_key"

    return df, cols




def _case_columns(cols: dict[str, str | None]) -> list[str]:
    case_cols = [cols["pdb"]]

    if cols["site"] is not None:
        case_cols.append(cols["site"])

    if cols["glycan_class"] is not None:
        case_cols.append(cols["glycan_class"])

    return case_cols


def _clean_status(df: pd.DataFrame, status_col: str | None) -> pd.DataFrame:
    if status_col is None:
        return df.copy()
    return df[df[status_col].isin(OK_STATUSES)].copy()


def _mode(values: pd.Series) -> object:
    values = values.dropna()
    if values.empty:
        return np.nan
    return values.value_counts().index[0]


def _weighted_fraction_equal(
    values: pd.Series,
    expected: object,
    weights: pd.Series | None = None,
) -> float:
    values = values.dropna()

    if values.empty or pd.isna(expected):
        return np.nan

    is_equal = values == expected

    if weights is None:
        return float(is_equal.mean())

    w = pd.to_numeric(weights.loc[values.index], errors="coerce").fillna(0.0)
    if w.sum() == 0:
        return np.nan

    return float(np.average(is_equal.astype(float), weights=w))


def _detect_torsion_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "pdb": first_existing_column(df, ["pdb_id", "pdb", "pdb_code"]),
        "tool": first_existing_column(df, ["tool", "source"]),
        "site": optional_column(
            df,
            ["site_id", "glycosite_id", "site", "site_label", "glycosylation_site"],
        ),
        "glycan_class": optional_column(
            df,
            ["glycan_class", "glycan_type", "class", "glycoform_class"],
        ),
        "status": optional_column(
            df,
            ["torsion_status", "descriptor_status", "conformation_status", "status"],
        ),
        "weight": optional_column(
            df,
            ["weight", "frame_weight", "model_weight", "cluster_weight", "population"],
        ),
        "model_index": optional_column(df, ["model_index", "frame", "frame_index"]),
        "linkage": first_existing_column(
            df,
            [
                "linkage_id",
                "linkage_key",
                "link_id",
                "bond_id",
                "glycosidic_linkage",
                "residue_pair",
            ],
        ),
        "torsion_name": optional_column(
            df,
            ["torsion_name", "angle_name", "dihedral_name"],
        ),
        "torsion_angle": optional_column(
            df,
            ["torsion_angle_deg", "angle_deg", "dihedral_angle_deg"],
        ),
        "phi": optional_column(df, ["phi", "phi_deg", "phi_angle", "phi_angle_deg"]),
        "psi": optional_column(df, ["psi", "psi_deg", "psi_angle", "psi_angle_deg"]),
        "omega": optional_column(
            df,
            ["omega", "omega_deg", "omega_angle", "omega_angle_deg"],
        ),
    }

def _prepare_torsion_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    cols = _detect_torsion_columns(df)
    df = _clean_status(df, cols["status"]).copy()
    df, cols = _add_torsion_match_key(df, cols)

    wide_angle_cols = [cols[name] for name in ["phi", "psi", "omega"] if cols[name] is not None]

    if wide_angle_cols:
        for angle_col in wide_angle_cols:
            df[angle_col] = pd.to_numeric(df[angle_col], errors="coerce")
        return df, cols

    if cols["torsion_name"] is None or cols["torsion_angle"] is None:
        raise KeyError(
            "No torsion angle columns found. Expected either phi/psi/omega "
            "or a long format with torsion_name + torsion_angle_deg."
        )

    df[cols["torsion_angle"]] = pd.to_numeric(df[cols["torsion_angle"]], errors="coerce")
    df[cols["torsion_name"]] = df[cols["torsion_name"]].astype(str).str.lower()

    index_cols = _case_columns(cols) + [cols["tool"], cols["linkage"]]

    if cols["model_index"] is not None:
        index_cols.append(cols["model_index"])

    if cols["weight"] is not None:
        df[cols["weight"]] = pd.to_numeric(df[cols["weight"]], errors="coerce").fillna(0.0)
        index_cols.append(cols["weight"])

    wide = (
        df.pivot_table(
            index=index_cols,
            columns=cols["torsion_name"],
            values=cols["torsion_angle"],
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for angle_name in ["phi", "psi", "omega"]:
        if angle_name not in wide.columns:
            wide[angle_name] = np.nan

    cols["phi"] = "phi"
    cols["psi"] = "psi"
    cols["omega"] = "omega"

    return wide, cols


def _detect_ring_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "pdb": first_existing_column(df, ["pdb_id", "pdb", "pdb_code"]),
        "tool": first_existing_column(df, ["tool", "source"]),
        "site": optional_column(
            df,
            ["site_id", "glycosite_id", "site", "site_label", "glycosylation_site"],
        ),
        "glycan_class": optional_column(
            df,
            ["glycan_class", "glycan_type", "class", "glycoform_class"],
        ),
        "status": optional_column(
            df,
            ["ring_status", "descriptor_status", "puckering_status", "status"],
        ),
        "weight": optional_column(
            df,
            ["weight", "frame_weight", "model_weight", "cluster_weight", "population"],
        ),
        "model_index": optional_column(df, ["model_index", "frame", "frame_index"]),
        "residue_index": optional_column(
            df,
            ["residue_index", "ring_id", "residue_id", "residue_key", "monosaccharide_id"],
        ),
        "residue_name": optional_column(
            df,
            ["residue_name", "monosaccharide", "monosaccharide_name", "sugar_name"],
        ),
        "pucker_class": first_existing_column(
            df,
            [
                "ring_class",
                "pucker_class",
                "cremer_pople_class",
                "puckering_class",
                "class_pucker",
            ],
        ),
    }


def _prepare_ring_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    cols = _detect_ring_columns(df)
    df = _clean_status(df, cols["status"]).copy()

    if cols["weight"] is not None:
        df[cols["weight"]] = pd.to_numeric(df[cols["weight"]], errors="coerce").fillna(0.0)

    base_cols = _case_columns(cols) + [cols["tool"]]

    if cols["model_index"] is not None:
        base_cols.append(cols["model_index"])

    residue_index_col = cols["residue_index"]
    residue_name_col = cols["residue_name"]

    use_existing_index = (
        residue_index_col is not None
        and df[residue_index_col].notna().any()
    )

    if use_existing_index:
        index_part = df[residue_index_col].astype("Int64").astype(str)
    else:
        index_part = (
            df.groupby(base_cols, dropna=False)
            .cumcount()
            .add(1)
            .astype(str)
        )

    if residue_name_col is not None:
        name_part = df[residue_name_col].fillna("UNK").astype(str)
    else:
        name_part = "UNK"

    df["_ring_key"] = index_part + "_" + name_part
    cols["ring"] = "_ring_key"

    return df, cols


def build_torsion_comparison(df: pd.DataFrame) -> pd.DataFrame:
    df, cols = _prepare_torsion_table(df)

    angle_cols = {
        angle_name: cols[angle_name]
        for angle_name in ["phi", "psi", "omega"]
        if cols[angle_name] is not None
    }

    for angle_col in angle_cols.values():
        df[angle_col] = pd.to_numeric(df[angle_col], errors="coerce")

    case_cols = _case_columns(cols)
    group_cols = case_cols + [cols["linkage"]]
    tool_col = cols["tool"]
    weight_col = cols["weight"]

    rows: list[dict[str, object]] = []

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_cols, group_key, strict=True))

        ref_df = group_df[group_df[tool_col] == TOOL_REFERENCE]
        if ref_df.empty:
            continue

        ref_values = {}
        for angle_name, angle_col in angle_cols.items():
            ref_series = pd.to_numeric(ref_df[angle_col], errors="coerce").dropna()
            ref_values[angle_name] = float(ref_series.iloc[0]) if not ref_series.empty else np.nan

        for tool, tool_df in group_df[group_df[tool_col] != TOOL_REFERENCE].groupby(tool_col, dropna=False):
            weights = tool_df[weight_col] if weight_col is not None else None

            row = {
                **group_info,
                "tool": tool,
                "n_models": int(len(tool_df)),
            }

            for angle_name, angle_col in angle_cols.items():
                ref_angle = ref_values[angle_name]
                values = pd.to_numeric(tool_df[angle_col], errors="coerce")
                distances = values.apply(lambda value: circular_distance_deg(value, ref_angle))

                row[f"reference_{angle_name}"] = ref_angle
                row[f"tool_circular_mean_{angle_name}"] = circular_mean_deg(values, weights)
                row[f"min_circular_distance_{angle_name}"] = (
                    float(distances.min()) if distances.notna().any() else np.nan
                )
                row[f"median_circular_distance_{angle_name}"] = weighted_median(distances, weights)
                row[f"mean_circular_distance_{angle_name}"] = weighted_mean(distances, weights)

            rows.append(row)

    return pd.DataFrame(rows)


def build_puckering_comparison(df: pd.DataFrame) -> pd.DataFrame:
    df, cols = _prepare_ring_table(df)

    case_cols = _case_columns(cols)
    group_cols = case_cols + [cols["ring"]]
    tool_col = cols["tool"]
    weight_col = cols["weight"]

    rows: list[dict[str, object]] = []

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_cols, group_key, strict=True))

        ref_df = group_df[group_df[tool_col] == TOOL_REFERENCE]
        if ref_df.empty:
            continue

        ref_classes = ref_df[cols["pucker_class"]].dropna()
        if ref_classes.empty:
            continue

        reference_class = ref_classes.iloc[0]

        for tool, tool_df in group_df[group_df[tool_col] != TOOL_REFERENCE].groupby(tool_col, dropna=False):
            weights = tool_df[weight_col] if weight_col is not None else None
            tool_classes = tool_df[cols["pucker_class"]]

            rows.append(
                {
                    **group_info,
                    "tool": tool,
                    "n_models": int(len(tool_df)),
                    "reference_pucker_class": reference_class,
                    "tool_majority_pucker_class": _mode(tool_classes),
                    "pucker_agreement_fraction": _weighted_fraction_equal(
                        tool_classes,
                        reference_class,
                        weights,
                    ),
                    "n_distinct_tool_pucker_classes": int(tool_classes.dropna().nunique()),
                }
            )

    return pd.DataFrame(rows)


def build_local_conformation_by_class(
    torsions: pd.DataFrame,
    puckering: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    use_class = "glycan_class" in torsions.columns or "glycan_class" in puckering.columns

    if not torsions.empty:
        group_cols = ["glycan_class", "tool"] if use_class and "glycan_class" in torsions.columns else ["tool"]

        for key, group in torsions.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)

            row = dict(zip(group_cols, key, strict=True))
            row["section"] = "torsions"
            row["n_rows"] = int(len(group))

            for angle_name in ["phi", "psi", "omega"]:
                col = f"median_circular_distance_{angle_name}"
                if col in group.columns:
                    row[f"median_of_median_circular_distance_{angle_name}"] = float(group[col].median())

            rows.append(row)

    if not puckering.empty:
        group_cols = ["glycan_class", "tool"] if use_class and "glycan_class" in puckering.columns else ["tool"]

        for key, group in puckering.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)

            row = dict(zip(group_cols, key, strict=True))
            row["section"] = "puckering"
            row["n_rows"] = int(len(group))
            row["median_pucker_agreement_fraction"] = float(
                group["pucker_agreement_fraction"].median()
            )
            row["mean_pucker_agreement_fraction"] = float(
                group["pucker_agreement_fraction"].mean()
            )

            rows.append(row)

    return pd.DataFrame(rows)


def run(
    torsions_csv: str | Path = "results/glycan_conformation_torsions.csv",
    rings_csv: str | Path = "results/glycan_conformation_rings.csv",
    output_torsions_csv: str | Path = "results/compare_local_torsions.csv",
    output_puckering_csv: str | Path = "results/compare_local_puckering.csv",
    output_by_class_csv: str | Path = "results/compare_local_conformation_by_class.csv",
) -> None:
    torsions_df = read_csv(torsions_csv)
    rings_df = read_csv(rings_csv)

    torsions = build_torsion_comparison(torsions_df)
    puckering = build_puckering_comparison(rings_df)
    by_class = build_local_conformation_by_class(torsions, puckering)

    write_csv(torsions, output_torsions_csv)
    write_csv(puckering, output_puckering_csv)
    write_csv(by_class, output_by_class_csv)

    print("[SUMMARY] torsion rows by tool")
    print(torsions["tool"].value_counts().to_string() if not torsions.empty else "none")

    print("[SUMMARY] puckering rows by tool")
    print(puckering["tool"].value_counts().to_string() if not puckering.empty else "none")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local glycan conformation against reference structures.",
    )
    parser.add_argument(
        "--torsions",
        default="results/glycan_conformation_torsions.csv",
        help="Input torsion table from layer 3.",
    )
    parser.add_argument(
        "--rings",
        default="results/glycan_conformation_rings.csv",
        help="Input ring puckering table from layer 3.",
    )
    parser.add_argument(
        "--output-torsions",
        default="results/compare_local_torsions.csv",
        help="Detailed torsion comparison table.",
    )
    parser.add_argument(
        "--output-puckering",
        default="results/compare_local_puckering.csv",
        help="Detailed puckering comparison table.",
    )
    parser.add_argument(
        "--output-by-class",
        default="results/compare_local_conformation_by_class.csv",
        help="Summary table stratified by glycan class and tool.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        torsions_csv=args.torsions,
        rings_csv=args.rings,
        output_torsions_csv=args.output_torsions,
        output_puckering_csv=args.output_puckering,
        output_by_class_csv=args.output_by_class,
    )


if __name__ == "__main__":
    main()
