from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from glycobench.compare.common import (
    TOOL_GLYCOSHAPE,
    TOOL_GLYCOSHIELD,
    TOOL_REFERENCE,
    first_existing_column,
    optional_column,
    read_csv,
    safe_pearson,
    safe_spearman,
    write_csv,
)


OK_STATUSES = {"ok", "self_reference"}


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


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def _rmse(x: pd.Series, y: pd.Series) -> float:
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.empty:
        return np.nan
    diff = xy.iloc[:, 0] - xy.iloc[:, 1]
    return float(np.sqrt(np.mean(diff**2)))


def _mae(x: pd.Series, y: pd.Series) -> float:
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.empty:
        return np.nan
    return float((xy.iloc[:, 0] - xy.iloc[:, 1]).abs().mean())


def _jaccard(a: pd.Series, b: pd.Series) -> float:
    a = _as_bool(a)
    b = _as_bool(b)

    union = (a | b).sum()
    if union == 0:
        return np.nan

    return float((a & b).sum() / union)


def _ordered_pairs(tools: set[str]) -> list[tuple[str, str]]:
    preferred = [
        (TOOL_REFERENCE, TOOL_GLYCOSHIELD),
        (TOOL_REFERENCE, TOOL_GLYCOSHAPE),
        (TOOL_GLYCOSHIELD, TOOL_GLYCOSHAPE),
    ]

    return [pair for pair in preferred if pair[0] in tools and pair[1] in tools]


def _detect_profile_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "pdb": first_existing_column(df, ["pdb_id", "pdb", "pdb_code"]),
        "site": optional_column(
            df,
            ["site", "site_id", "glycosite_id", "site_label", "glycosylation_site"],
        ),
        "glycan_class": optional_column(
            df,
            ["glycan_class", "glycan_type", "class", "glycoform_class"],
        ),
        "tool": first_existing_column(df, ["tool", "source"]),
        "residue_key": first_existing_column(
            df,
            ["residue_key", "protein_residue_key", "residue_id"],
        ),
        "masking_frequency": first_existing_column(
            df,
            ["masking_frequency", "masked_frequency", "mask_frequency"],
        ),
        "mean_delta": first_existing_column(
            df,
            ["mean_delta_sasa_a2", "mean_delta_sasa", "delta_sasa_mean"],
        ),
        "max_delta": first_existing_column(
            df,
            ["max_delta_sasa_a2", "max_delta_sasa", "delta_sasa_max"],
        ),
        "n_masked_models": optional_column(
            df,
            ["n_masked_models", "masked_models"],
        ),
        "n_models": optional_column(df, ["n_models", "model_count"]),
    }


def build_shielding_profile_comparison(df: pd.DataFrame) -> pd.DataFrame:
    cols = _detect_profile_columns(df)
    df = df.copy()

    for col_name in ["masking_frequency", "mean_delta", "max_delta"]:
        df[cols[col_name]] = pd.to_numeric(df[cols[col_name]], errors="coerce")

    case_cols = _case_columns(cols)
    tool_col = cols["tool"]
    residue_col = cols["residue_key"]

    rows: list[dict[str, object]] = []

    for case_key, case_df in df.groupby(case_cols, dropna=False):
        if not isinstance(case_key, tuple):
            case_key = (case_key,)

        case_info = dict(zip(case_cols, case_key, strict=True))
        tools = set(case_df[tool_col].dropna().astype(str).unique())

        for tool_a, tool_b in _ordered_pairs(tools):
            a = case_df[case_df[tool_col] == tool_a].copy()
            b = case_df[case_df[tool_col] == tool_b].copy()

            merged = a.merge(
                b,
                on=residue_col,
                suffixes=("_a", "_b"),
                how="inner",
            )

            if merged.empty:
                continue

            mean_a = merged[f"{cols['mean_delta']}_a"]
            mean_b = merged[f"{cols['mean_delta']}_b"]
            freq_a = merged[f"{cols['masking_frequency']}_a"]
            freq_b = merged[f"{cols['masking_frequency']}_b"]

            masked_a = freq_a > 0
            masked_b = freq_b > 0

            rows.append(
                {
                    **case_info,
                    "tool_a": tool_a,
                    "tool_b": tool_b,
                    "comparison": f"{tool_a}_vs_{tool_b}",
                    "n_common_residues": int(len(merged)),
                    "n_residues_tool_a": int(a[residue_col].nunique()),
                    "n_residues_tool_b": int(b[residue_col].nunique()),
                    "coverage_tool_a": float(len(merged) / a[residue_col].nunique())
                    if a[residue_col].nunique() else np.nan,
                    "coverage_tool_b": float(len(merged) / b[residue_col].nunique())
                    if b[residue_col].nunique() else np.nan,
                    "pearson_mean_delta_sasa": safe_pearson(mean_a, mean_b),
                    "spearman_mean_delta_sasa": safe_spearman(mean_a, mean_b),
                    "mae_mean_delta_sasa_a2": _mae(mean_a, mean_b),
                    "rmse_mean_delta_sasa_a2": _rmse(mean_a, mean_b),
                    "sum_mean_delta_sasa_tool_a": float(mean_a.sum()),
                    "sum_mean_delta_sasa_tool_b": float(mean_b.sum()),
                    "delta_sum_mean_delta_sasa_b_minus_a": float(mean_b.sum() - mean_a.sum()),
                    "pearson_masking_frequency": safe_pearson(freq_a, freq_b),
                    "spearman_masking_frequency": safe_spearman(freq_a, freq_b),
                    "mae_masking_frequency": _mae(freq_a, freq_b),
                    "masked_residues_tool_a": int(masked_a.sum()),
                    "masked_residues_tool_b": int(masked_b.sum()),
                    "masked_residue_overlap": int((masked_a & masked_b).sum()),
                    "masked_residue_union": int((masked_a | masked_b).sum()),
                    "masked_residue_jaccard": _jaccard(masked_a, masked_b),
                }
            )

    return pd.DataFrame(rows)


def _detect_residue_sasa_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "pdb": first_existing_column(df, ["pdb_id", "pdb", "pdb_code"]),
        "site": optional_column(
            df,
            ["site", "site_id", "glycosite_id", "site_label", "glycosylation_site"],
        ),
        "glycan_class": optional_column(
            df,
            ["glycan_class", "glycan_type", "class", "glycoform_class"],
        ),
        "tool": first_existing_column(df, ["tool", "source"]),
        "model_index": first_existing_column(df, ["model_index", "frame", "frame_index"]),
        "status": optional_column(df, ["shielding_status", "descriptor_status", "status"]),
        "residue_key": first_existing_column(
            df,
            ["residue_key", "protein_residue_key", "residue_id"],
        ),
        "delta": first_existing_column(
            df,
            ["delta_sasa_a2", "delta_sasa", "sasa_delta"],
        ),
        "masked": first_existing_column(df, ["masked", "is_masked"]),
    }


def build_model_shielding_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = _detect_residue_sasa_columns(df)
    df = _clean_status(df, cols["status"]).copy()

    df[cols["delta"]] = pd.to_numeric(df[cols["delta"]], errors="coerce").fillna(0.0)
    df["_positive_delta_sasa_a2"] = df[cols["delta"]].clip(lower=0.0)
    df["_masked_bool"] = _as_bool(df[cols["masked"]])

    group_cols = _case_columns(cols) + [cols["tool"], cols["model_index"]]

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_residues=(cols["residue_key"], "nunique"),
            total_delta_sasa_a2=(cols["delta"], "sum"),
            total_positive_delta_sasa_a2=("_positive_delta_sasa_a2", "sum"),
            n_masked_residues=("_masked_bool", "sum"),
        )
        .reset_index()
    )

    out["masked_residue_fraction"] = out["n_masked_residues"] / out["n_residues"]

    return out


def build_shielding_by_class(
    profiles: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not model_summary.empty:
        group_cols = ["tool"]
        if "glycan_class" in model_summary.columns:
            group_cols = ["glycan_class", "tool"]

        for key, group in model_summary.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)

            row = dict(zip(group_cols, key, strict=True))
            row.update(
                {
                    "section": "model_totals",
                    "n_rows": int(len(group)),
                    "n_cases": int(group[["pdb_id", "site"]].drop_duplicates().shape[0])
                    if "site" in group.columns
                    else int(group["pdb_id"].nunique()),
                    "median_total_positive_delta_sasa_a2": float(
                        group["total_positive_delta_sasa_a2"].median()
                    ),
                    "mean_total_positive_delta_sasa_a2": float(
                        group["total_positive_delta_sasa_a2"].mean()
                    ),
                    "median_n_masked_residues": float(
                        group["n_masked_residues"].median()
                    ),
                    "mean_masked_residue_fraction": float(
                        group["masked_residue_fraction"].mean()
                    ),
                }
            )
            rows.append(row)

    if not profiles.empty:
        group_cols = ["comparison"]
        if "glycan_class" in profiles.columns:
            group_cols = ["glycan_class", "comparison"]

        for key, group in profiles.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)

            row = dict(zip(group_cols, key, strict=True))
            row.update(
                {
                    "section": "profile_comparison",
                    "n_rows": int(len(group)),
                    "median_pearson_mean_delta_sasa": float(
                        group["pearson_mean_delta_sasa"].median()
                    ),
                    "median_spearman_mean_delta_sasa": float(
                        group["spearman_mean_delta_sasa"].median()
                    ),
                    "median_mae_mean_delta_sasa_a2": float(
                        group["mae_mean_delta_sasa_a2"].median()
                    ),
                    "median_masked_residue_jaccard": float(
                        group["masked_residue_jaccard"].median()
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def run(
    residue_sasa_csv: str | Path = "results/glycan_shielding_residue_sasa.csv",
    summary_csv: str | Path = "results/glycan_shielding_summary.csv",
    output_profiles_csv: str | Path = "results/compare_shielding_profiles.csv",
    output_summary_csv: str | Path = "results/compare_shielding_summary.csv",
    output_by_class_csv: str | Path = "results/compare_shielding_by_class.csv",
) -> None:
    summary_df = read_csv(summary_csv)
    residue_df = read_csv(residue_sasa_csv)

    profiles = build_shielding_profile_comparison(summary_df)
    model_summary = build_model_shielding_summary(residue_df)
    by_class = build_shielding_by_class(profiles, model_summary)

    write_csv(profiles, output_profiles_csv)
    write_csv(model_summary, output_summary_csv)
    write_csv(by_class, output_by_class_csv)

    print("[SUMMARY] profile comparisons")
    print(profiles["comparison"].value_counts().to_string() if not profiles.empty else "none")

    print("[SUMMARY] model totals by tool")
    print(model_summary["tool"].value_counts().to_string() if not model_summary.empty else "none")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare glycan shielding profiles from layer 3 tables.",
    )
    parser.add_argument(
        "--residue-sasa",
        default="results/glycan_shielding_residue_sasa.csv",
        help="Per-residue, per-model shielding table from layer 3.",
    )
    parser.add_argument(
        "--summary",
        default="results/glycan_shielding_summary.csv",
        help="Per-residue shielding summary table from layer 3.",
    )
    parser.add_argument(
        "--output-profiles",
        default="results/compare_shielding_profiles.csv",
        help="Pairwise profile comparison table.",
    )
    parser.add_argument(
        "--output-summary",
        default="results/compare_shielding_summary.csv",
        help="Per-model total shielding summary.",
    )
    parser.add_argument(
        "--output-by-class",
        default="results/compare_shielding_by_class.csv",
        help="Summary stratified by glycan class.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        residue_sasa_csv=args.residue_sasa,
        summary_csv=args.summary,
        output_profiles_csv=args.output_profiles,
        output_summary_csv=args.output_summary,
        output_by_class_csv=args.output_by_class,
    )


if __name__ == "__main__":
    main()
