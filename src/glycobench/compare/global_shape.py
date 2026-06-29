from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glycobench.compare.common import (
    TOOL_REFERENCE,
    first_existing_column,
    optional_column,
    percentile_of_value,
    read_csv,
    weighted_mean,
    weighted_median,
    write_csv,
)


def _detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
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
        "recommended_use": optional_column(df, ["recommended_use", "scope"]),
        "status": optional_column(
            df,
            ["descriptor_status", "shape_status", "status"],
        ),
        "weight": optional_column(
            df,
            ["weight", "frame_weight", "model_weight", "cluster_weight", "population"],
        ),
        "rmsd": first_existing_column(
            df,
            ["rmsd_to_reference", "glycan_rmsd_to_reference", "rmsd_ref", "rmsd"],
        ),
        "rg": first_existing_column(
            df,
            ["rg", "radius_gyration", "radius_of_gyration", "rg_angstrom", "rg_A"],
        ),
        "span": first_existing_column(
            df,
            ["span", "glycan_span", "max_span", "span_angstrom", "span_A"],
        ),
    }


def _case_columns(cols: dict[str, str | None]) -> list[str]:
    case_cols = [cols["pdb"]]

    if cols["site"] is not None:
        case_cols.append(cols["site"])

    if cols["glycan_class"] is not None:
        case_cols.append(cols["glycan_class"])

    return case_cols


def _clean_input(df: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    out = df.copy()

    if cols["status"] is not None:
        out = out[out[cols["status"]].isin(["ok", "self_reference"])].copy()

    for col_name in ["rmsd", "rg", "span"]:
        out[cols[col_name]] = pd.to_numeric(out[cols[col_name]], errors="coerce")

    if cols["weight"] is not None:
        out[cols["weight"]] = pd.to_numeric(out[cols["weight"]], errors="coerce")
        out[cols["weight"]] = out[cols["weight"]].fillna(0.0)

    return out


def _get_single_reference_value(
    ref_rows: pd.DataFrame,
    metric_col: str,
) -> float:
    values = pd.to_numeric(ref_rows[metric_col], errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[0])


def build_global_shape_comparison(df: pd.DataFrame) -> pd.DataFrame:
    cols = _detect_columns(df)
    df = _clean_input(df, cols)

    case_cols = _case_columns(cols)
    tool_col = cols["tool"]
    weight_col = cols["weight"]

    rows: list[dict[str, object]] = []

    for case_key, case_df in df.groupby(case_cols, dropna=False):
        if not isinstance(case_key, tuple):
            case_key = (case_key,)

        case_info = dict(zip(case_cols, case_key, strict=True))

        ref_df = case_df[case_df[tool_col] == TOOL_REFERENCE]
        if ref_df.empty:
            continue

        ref_rg = _get_single_reference_value(ref_df, cols["rg"])
        ref_span = _get_single_reference_value(ref_df, cols["span"])

        for tool, tool_df in case_df.groupby(tool_col, dropna=False):
            if tool == TOOL_REFERENCE:
                continue

            weights = tool_df[weight_col] if weight_col is not None else None

            rmsd_values = pd.to_numeric(tool_df[cols["rmsd"]], errors="coerce")
            rg_values = pd.to_numeric(tool_df[cols["rg"]], errors="coerce")
            span_values = pd.to_numeric(tool_df[cols["span"]], errors="coerce")

            row = {
                **case_info,
                "tool": tool,
                "n_models": int(len(tool_df)),
                "n_rmsd_valid": int(rmsd_values.notna().sum()),
                "min_rmsd_to_reference": float(rmsd_values.min())
                if rmsd_values.notna().any()
                else np.nan,
                "median_rmsd_to_reference": weighted_median(rmsd_values, weights),
                "mean_rmsd_to_reference": weighted_mean(rmsd_values, weights),
                "reference_rg": ref_rg,
                "tool_median_rg": weighted_median(rg_values, weights),
                "tool_mean_rg": weighted_mean(rg_values, weights),
                "reference_rg_percentile_in_tool": percentile_of_value(
                    rg_values,
                    ref_rg,
                ),
                "delta_median_rg_vs_reference": weighted_median(rg_values, weights)
                - ref_rg
                if not pd.isna(ref_rg)
                else np.nan,
                "reference_span": ref_span,
                "tool_median_span": weighted_median(span_values, weights),
                "tool_mean_span": weighted_mean(span_values, weights),
                "reference_span_percentile_in_tool": percentile_of_value(
                    span_values,
                    ref_span,
                ),
                "delta_median_span_vs_reference": weighted_median(span_values, weights)
                - ref_span
                if not pd.isna(ref_span)
                else np.nan,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def build_global_shape_by_class(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison.copy()

    group_cols = []

    if "glycan_class" in comparison.columns:
        group_cols.append("glycan_class")

    group_cols.append("tool")

    rows = []

    for key, group in comparison.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(group_cols, key, strict=True))

        row.update(
            {
                "n_cases": int(len(group)),
                "median_min_rmsd_to_reference": float(
                    group["min_rmsd_to_reference"].median()
                ),
                "mean_min_rmsd_to_reference": float(
                    group["min_rmsd_to_reference"].mean()
                ),
                "median_reference_rg_percentile": float(
                    group["reference_rg_percentile_in_tool"].median()
                ),
                "median_reference_span_percentile": float(
                    group["reference_span_percentile_in_tool"].median()
                ),
                "median_delta_rg_vs_reference": float(
                    group["delta_median_rg_vs_reference"].median()
                ),
                "median_delta_span_vs_reference": float(
                    group["delta_median_span_vs_reference"].median()
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def run(
    input_csv: str | Path = "results/glycan_shape_descriptors.csv",
    output_csv: str | Path = "results/compare_global_shape.csv",
    output_by_class_csv: str | Path = "results/compare_global_shape_by_class.csv",
) -> None:
    df = read_csv(input_csv)

    comparison = build_global_shape_comparison(df)
    by_class = build_global_shape_by_class(comparison)

    write_csv(comparison, output_csv)
    write_csv(by_class, output_by_class_csv)

    if comparison.empty:
        print("[WARN] No comparison row was produced.")
        print("[WARN] Check that reference rows and tool rows share the same case keys.")
        return

    print("[SUMMARY] rows by tool")
    print(comparison["tool"].value_counts().to_string())

    if "glycan_class" in comparison.columns:
        print("[SUMMARY] rows by glycan class")
        print(comparison["glycan_class"].value_counts().to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare global glycan shape descriptors against reference structures.",
    )
    parser.add_argument(
        "--input",
        default="results/glycan_shape_descriptors.csv",
        help="Input table from layer 3.",
    )
    parser.add_argument(
        "--output",
        default="results/compare_global_shape.csv",
        help="Detailed output table.",
    )
    parser.add_argument(
        "--output-by-class",
        default="results/compare_global_shape_by_class.csv",
        help="Summary output table stratified by glycan class and tool.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        input_csv=args.input,
        output_csv=args.output,
        output_by_class_csv=args.output_by_class,
    )


if __name__ == "__main__":
    main()
