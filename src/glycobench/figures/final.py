from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TOOL_ORDER = ["glycoshield", "glycoshape"]
MODEL_TOOL_ORDER = ["reference", "glycoshield", "glycoshape"]
COMPARISON_ORDER = [
    "reference_vs_glycoshield",
    "reference_vs_glycoshape",
    "glycoshield_vs_glycoshape",
]
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42
SITE_KEY_COL = "site_key"
LOW_N_CI_THRESHOLD = 3
POINT_SIZE = 34


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, low_memory=False)


def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    fig_dir = outdir / "figures"
    table_dir = outdir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def save_figure(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[OK] wrote {png}")
    print(f"[OK] wrote {pdf}")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalise_tool_name(value: object) -> str:
    text = str(value).lower().replace("-", "").replace("_", "")
    if "reference" in text or text == "ref":
        return "reference"
    if "glycoshield" in text:
        return "glycoshield"
    if "glycoshape" in text or "reglyco" in text:
        return "glycoshape"
    return text


def _is_present(value: object) -> bool:
    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "false", "0"}:
        return False

    try:
        return float(text) > 0
    except ValueError:
        return True


def _join_unique(values: pd.Series) -> str:
    clean = values.dropna().astype(str).map(str.strip)
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    return "; ".join(sorted(clean.unique()))


def _add_site_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable site key used to keep comparisons paired by PDB and glycosylation site."""
    if SITE_KEY_COL in df.columns:
        return df.copy()

    required = {"pdb_id", "site"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot build paired triple-site figures. "
            f"Missing columns for site key: {sorted(missing)}"
        )

    out = df.copy()
    out[SITE_KEY_COL] = (
        out["pdb_id"].astype(str).str.strip()
        + "|"
        + out["site"].astype(str).str.strip()
    )
    return out


def _availability_from_manifest(group: pd.DataFrame, prefix: str) -> bool:
    candidates = [
        f"{prefix}_n_models",
        f"{prefix}_n_residues",
        f"{prefix}_glycan_pdb",
        f"{prefix}_glycan_ensemble_pdb",
    ]
    for col in candidates:
        if col in group.columns:
            return group[col].map(_is_present).any()
    return False


def load_triple_site_manifest(results_dir: Path, table_dir: Path) -> pd.DataFrame:
    """Return only sites where reference, GlycoShield and GlycoShape are all available."""
    manifest_path = results_dir / "analysis_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "analysis_manifest.csv is required to restrict figures to paired triple sites"
        )

    manifest = _add_site_key(read_csv(manifest_path))

    required = {"pdb_id", "site", "glycan_class"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Cannot identify triple sites. Missing columns in manifest: {sorted(missing)}"
        )

    scope_col = _first_existing_column(
        manifest,
        ["scope", "analysis_scope", "dataset_scope", "comparison_scope", "recommended_use", "status"],
    )

    rows: list[dict[str, object]] = []
    for site_key, group in manifest.groupby(SITE_KEY_COL, sort=True):
        scope_is_triple = False
        if scope_col is not None:
            scope_values = group[scope_col].dropna().astype(str).str.lower()
            scope_is_triple = scope_values.str.contains("triple").any()

        has_reference = _availability_from_manifest(group, "reference")
        has_glycoshield = _availability_from_manifest(group, "glycoshield")
        has_glycoshape = _availability_from_manifest(group, "glycoshape")
        availability_is_triple = has_reference and has_glycoshield and has_glycoshape
        is_triple = scope_is_triple or availability_is_triple

        if not is_triple:
            continue

        rows.append(
            {
                SITE_KEY_COL: site_key,
                "pdb_id": group["pdb_id"].dropna().astype(str).iloc[0],
                "site": group["site"].dropna().astype(str).iloc[0],
                "glycan_class": group["glycan_class"].dropna().astype(str).iloc[0],
            }
        )

    triple = pd.DataFrame(rows)
    if triple.empty:
        raise ValueError("No paired triple site found in analysis_manifest.csv")

    counts = (
        triple.groupby("glycan_class", dropna=False)[SITE_KEY_COL]
        .nunique()
        .rename("n_triple_sites")
        .reset_index()
        .sort_values("glycan_class")
    )
    counts_out = table_dir / "table_triple_site_counts.csv"
    counts.to_csv(counts_out, index=False)
    print(f"[OK] wrote {counts_out}")
    print("[INFO] paired triple sites by class")
    for _, row in counts.iterrows():
        print(f"  {row['glycan_class']}: n={row['n_triple_sites']}")

    return triple


def restrict_to_triple_sites(df: pd.DataFrame, triple_sites: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep the same paired triple sites in every figure-generating table."""
    data = _add_site_key(df)
    before_sites = data[SITE_KEY_COL].nunique()
    keep = set(triple_sites[SITE_KEY_COL])
    data = data[data[SITE_KEY_COL].isin(keep)].copy()
    after_sites = data[SITE_KEY_COL].nunique()

    if data.empty:
        raise ValueError(f"{label}: no row left after paired triple-site filtering")

    print(
        f"[INFO] {label}: restricted to paired triple sites "
        f"({after_sites}/{before_sites} available sites kept)"
    )
    return data


def site_level_table(
    df: pd.DataFrame,
    *,
    hue: str,
    metrics: list[str],
    aggfunc: str = "median",
) -> pd.DataFrame:
    """Collapse model/linkage/ring-level rows into one value per site and hue.

    This prevents n from becoming model count or linkage count. The figure n is therefore
    the number of paired glycosylation sites contributing to each class/tool bar.
    """
    available_metrics = [metric for metric in metrics if metric in df.columns]
    if not available_metrics:
        return pd.DataFrame()

    required = {SITE_KEY_COL, "pdb_id", "site", "glycan_class", hue}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot build site-level table. Missing columns: {sorted(missing)}")

    data = df[[SITE_KEY_COL, "pdb_id", "site", "glycan_class", hue, *available_metrics]].copy()
    data = data.dropna(subset=[hue])

    grouped = data.groupby([SITE_KEY_COL, "pdb_id", "site", "glycan_class", hue], dropna=False)
    if aggfunc == "mean":
        out = grouped[available_metrics].mean().reset_index()
    elif aggfunc == "median":
        out = grouped[available_metrics].median().reset_index()
    else:
        raise ValueError(f"Unsupported site-level aggregation: {aggfunc}")

    return out



def folded_typicality_from_percentile(values: pd.Series | np.ndarray) -> pd.Series:
    """Convert a raw percentile into folded typicality.

    The input comparison tables store the percentile position of the experimental
    reference inside the tool ensemble. This raw percentile is directional: values
    near 0 or 100 both mean that the reference lies at the edge of the generated
    distribution. The folded typicality score removes that directionality:

        typicality = 2 * min(p, 1 - p)

    with p expressed between 0 and 1. The output is therefore 1 for a central
    reference and 0 for a reference at either distribution edge.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series.dropna()
    if finite.empty:
        return series

    # Layer-4 tables have historically used 0-100 percentiles. This also accepts
    # already-normalized 0-1 percentiles to keep the script robust.
    if finite.max() > 1.0 or finite.min() < 0.0:
        p = series / 100.0
    else:
        p = series

    p = p.clip(lower=0.0, upper=1.0)
    return 2.0 * np.minimum(p, 1.0 - p)

def bootstrap_ci(
    values: pd.Series | np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Median and percentile bootstrap 95% confidence interval."""
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)

    median = float(np.median(arr))
    if arr.size == 1:
        return (median, median, median)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    medians = np.median(arr[idx], axis=1)
    ci_low, ci_high = np.percentile(medians, [2.5, 97.5])
    return median, float(ci_low), float(ci_high)


def summarise_bootstrap(
    df: pd.DataFrame,
    *,
    x: str,
    hue: str,
    y: str,
) -> pd.DataFrame:
    data = df[[x, hue, y, SITE_KEY_COL]].dropna(subset=[x, hue, y]).copy()
    rows = []
    for (x_value, hue_value), group in data.groupby([x, hue], dropna=False, sort=True):
        median, ci_low, ci_high = bootstrap_ci(group[y])
        rows.append(
            {
                x: x_value,
                hue: hue_value,
                "metric": y,
                "n_sites": group[SITE_KEY_COL].nunique(),
                "median": median,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def write_paired_bootstrap_differences(
    df: pd.DataFrame,
    *,
    x: str,
    hue: str,
    y: str,
    table_dir: Path,
    name: str,
    hue_pairs: list[tuple[str, str]],
) -> None:
    """Write paired bootstrap CIs for median differences between bars.

    Positive difference means second hue minus first hue.
    """
    required = {x, hue, y, SITE_KEY_COL}
    missing = required - set(df.columns)
    if missing:
        print(f"[SKIP] paired bootstrap {name}: missing {sorted(missing)}")
        return

    data = df[[x, hue, y, SITE_KEY_COL]].dropna().copy()
    rows = []

    for x_value, group in data.groupby(x, dropna=False, sort=True):
        pivot = group.pivot_table(index=SITE_KEY_COL, columns=hue, values=y, aggfunc="median")
        for first, second in hue_pairs:
            if first not in pivot.columns or second not in pivot.columns:
                continue
            paired = pivot[[first, second]].dropna()
            if paired.empty:
                continue

            diffs = paired[second] - paired[first]
            median, ci_low, ci_high = bootstrap_ci(diffs)
            rows.append(
                {
                    x: x_value,
                    "metric": y,
                    "comparison": f"{second}_minus_{first}",
                    "n_paired_sites": int(len(diffs)),
                    "median_difference": median,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

    if not rows:
        print(f"[SKIP] paired bootstrap {name}: no paired data")
        return

    out = table_dir / f"table_{name}_paired_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {out}")


def draw_grouped_site_distribution(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x: str,
    hue: str,
    y: str,
    title: str,
    ylabel: str,
    hue_order: list[str] | None = None,
    y_reference_line: float | None = None,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Draw one point per paired site, with a median tick and bootstrap CI.

    Low-n groups are intentionally greyed and their bootstrap CI is not drawn,
    because an error bar over one or two sites gives a misleading impression of
    statistical support.
    """
    required = {x, hue, y, SITE_KEY_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")

    data = df[[x, hue, y, SITE_KEY_COL]].dropna(subset=[x, hue, y]).copy()
    if data.empty:
        return pd.DataFrame(), False

    data[x] = data[x].astype(str)
    data[hue] = data[hue].astype(str)
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[y])
    if data.empty:
        return pd.DataFrame(), False

    if hue_order is None:
        hue_values = sorted(data[hue].dropna().unique())
    else:
        present_hues = set(data[hue].dropna().unique())
        hue_values = [str(value) for value in hue_order if str(value) in present_hues]

    x_values = sorted(data[x].dropna().unique())
    stats = summarise_bootstrap(data, x=x, hue=hue, y=y)
    if stats.empty:
        return stats, False

    n_x = len(x_values)
    n_hue = max(1, len(hue_values))
    width = min(0.8 / n_hue, 0.32)
    x_positions = np.arange(n_x)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    finite_values: list[float] = []
    low_n_any = False

    for i, hue_value in enumerate(hue_values):
        color = colors[i % len(colors)] if colors else None
        offset = (i - (n_hue - 1) / 2) * width

        for x_idx, x_value in enumerate(x_values):
            subset = data[(data[x] == x_value) & (data[hue] == str(hue_value))]
            if subset.empty:
                continue

            # One value per paired site. This keeps the visual layer consistent
            # with the n shown on the figure and with the bootstrap unit.
            per_site = (
                subset.groupby(SITE_KEY_COL, dropna=False)[y]
                .median()
                .dropna()
                .astype(float)
                .sort_values()
            )
            values = per_site.to_numpy()
            n_sites = int(per_site.shape[0])
            if n_sites == 0:
                continue

            stat_row = stats[
                (stats[x].astype(str) == x_value)
                & (stats[hue].astype(str) == str(hue_value))
            ]
            if stat_row.empty:
                continue

            median = float(stat_row["median"].iloc[0])
            ci_low = float(stat_row["ci95_low"].iloc[0])
            ci_high = float(stat_row["ci95_high"].iloc[0])
            base_x = x_positions[x_idx] + offset
            is_low_n = n_sites < LOW_N_CI_THRESHOLD
            low_n_any = low_n_any or is_low_n
            plot_color = "0.55" if is_low_n else color
            alpha = 0.65 if is_low_n else 0.85

            if n_sites == 1:
                jitter = np.array([0.0])
            else:
                jitter = np.linspace(-0.23 * width, 0.23 * width, n_sites)

            ax.scatter(
                np.full(n_sites, base_x) + jitter,
                values,
                s=POINT_SIZE,
                color=plot_color,
                alpha=alpha,
                zorder=3,
            )

            # Median tick. Bootstrap CI is shown only when there are enough sites
            # to avoid implying robustness from n=1 or n=2 groups.
            if is_low_n:
                ax.plot(
                    [base_x - 0.24 * width, base_x + 0.24 * width],
                    [median, median],
                    color="0.35",
                    linewidth=1.6,
                    zorder=4,
                )
                label = f"n={n_sites}*"
            else:
                yerr_low = max(0.0, median - ci_low)
                yerr_high = max(0.0, ci_high - median)
                ax.errorbar(
                    [base_x],
                    [median],
                    yerr=[[yerr_low], [yerr_high]],
                    fmt="_",
                    markersize=13,
                    markeredgewidth=2.0,
                    capsize=4,
                    color=plot_color,
                    linewidth=1.2,
                    zorder=4,
                )
                label = f"n={n_sites}"

            label_y = float(np.nanmax(values))
            if np.isfinite(ci_high) and not is_low_n:
                label_y = max(label_y, ci_high)
            ax.annotate(
                label,
                xy=(base_x, label_y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

            finite_values.extend(values[np.isfinite(values)].tolist())
            if np.isfinite(ci_low):
                finite_values.append(ci_low)
            if np.isfinite(ci_high):
                finite_values.append(ci_high)

    if y_reference_line is not None:
        ax.axhline(y_reference_line, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values, rotation=35, ha="right")

    if show_legend and hue_values:
        handles = []
        for i, hue_value in enumerate(hue_values):
            color = colors[i % len(colors)] if colors else None
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=color,
                    label=str(hue_value),
                    markersize=6,
                )
            )
        if low_n_any:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color="0.55",
                    label=f"n<{LOW_N_CI_THRESHOLD}, indicative",
                    markersize=6,
                )
            )
        ax.legend(handles=handles, title=hue, frameon=False)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif finite_values:
        finite = np.array([value for value in finite_values if np.isfinite(value)], dtype=float)
        if finite.size:
            ymin = min(0.0, float(np.nanmin(finite)))
            ymax = float(np.nanmax(finite))
            padding = 0.22 * (ymax - ymin if ymax > ymin else max(abs(ymax), 1.0))
            ax.set_ylim(ymin, ymax + padding)

    return stats, low_n_any


def grouped_barplot(
    df: pd.DataFrame,
    *,
    x: str,
    hue: str,
    y: str,
    title: str,
    ylabel: str,
    fig_dir: Path,
    table_dir: Path,
    name: str,
    hue_order: list[str] | None = None,
    y_reference_line: float | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Grouped strip plot with one point per paired site.

    The function name is kept for backward compatibility with the rest of the
    script, but the visual representation is no longer a barplot. This is more
    honest for classes with only 6-8 paired sites and prevents n=2 omega groups
    from looking more robust than they are.
    """
    required = {x, hue, y, SITE_KEY_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    try:
        stats, low_n_any = draw_grouped_site_distribution(
            ax,
            df,
            x=x,
            hue=hue,
            y=y,
            title=title,
            ylabel=ylabel,
            hue_order=hue_order,
            y_reference_line=y_reference_line,
            y_limits=y_limits,
        )
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc

    if stats.empty:
        plt.close(fig)
        print(f"[SKIP] {name}: no data")
        return

    summary_out = table_dir / f"table_{name}_summary.csv"
    stats.to_csv(summary_out, index=False)
    print(f"[OK] wrote {summary_out}")

    note = (
        f"Triple paired sites only. Points=sites; tick=median; CI95 shown only if n≥{LOW_N_CI_THRESHOLD}."
    )
    if low_n_any:
        note += f" Grey/n*: n<{LOW_N_CI_THRESHOLD}, indicative; CI hidden."

    fig.text(
        0.01,
        0.01,
        note,
        ha="left",
        va="bottom",
        fontsize=7,
    )

    save_figure(fig, fig_dir, name)

def write_dataset_overview_outputs(results_dir: Path, fig_dir: Path, table_dir: Path) -> None:
    manifest_path = results_dir / "analysis_manifest.csv"

    if not manifest_path.exists():
        print(f"[SKIP] dataset overview: missing {manifest_path}")
        return

    df = read_csv(manifest_path)

    required = {"pdb_id", "glycan_class", "site"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot build dataset overview table. Missing columns: {sorted(missing)}"
        )

    rows = []

    for pdb_id, group in df.groupby("pdb_id", sort=True):
        has_reference = _availability_from_manifest(group, "reference")
        has_glycoshield = _availability_from_manifest(group, "glycoshield")
        has_glycoshape = _availability_from_manifest(group, "glycoshape")

        if has_reference and has_glycoshield and has_glycoshape:
            status = "triple"
        elif has_reference and has_glycoshield:
            status = "reference + GlycoShield"
        elif has_reference and has_glycoshape:
            status = "reference + GlycoShape"
        elif has_reference:
            status = "reference only"
        else:
            status = "excluded / no reference"

        rows.append(
            {
                "PDB": pdb_id,
                "glycan_class": _join_unique(group["glycan_class"]),
                "sites": _join_unique(group["site"]),
                "n_sites": group["site"].dropna().astype(str).nunique(),
                "reference": "yes" if has_reference else "no",
                "GlycoShield": "yes" if has_glycoshield else "no",
                "GlycoShape": "yes" if has_glycoshape else "no",
                "status": status,
            }
        )

    overview = pd.DataFrame(rows)

    out = table_dir / "table_dataset_overview.csv"
    overview.to_csv(out, index=False)
    print(f"[OK] wrote {out}")

    figure_table = overview.copy()
    for col in ["reference", "GlycoShield", "GlycoShape"]:
        figure_table[col] = figure_table[col].map({"yes": "✓", "no": "–"})

    fig_height = max(5.0, 0.34 * len(figure_table) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=figure_table.values,
        colLabels=figure_table.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)

    ax.set_title("Dataset overview", pad=16)

    save_figure(fig, fig_dir, "fig_dataset_overview_table")


def write_global_shape_outputs(
    results_dir: Path,
    fig_dir: Path,
    table_dir: Path,
    triple_sites: pd.DataFrame,
) -> None:
    df = restrict_to_triple_sites(
        read_csv(results_dir / "compare_global_shape.csv"),
        triple_sites,
        "global shape",
    )
    df = site_level_table(
        df,
        hue="tool",
        metrics=[
            "min_rmsd_to_reference",
            "reference_rg_percentile_in_tool",
            "reference_span_percentile_in_tool",
            "delta_median_rg_vs_reference",
            "delta_median_span_vs_reference",
        ],
    )

    if df.empty:
        print("[SKIP] global shape: no available metric")
        return

    if "reference_rg_percentile_in_tool" in df.columns:
        df["reference_rg_typicality_in_tool"] = folded_typicality_from_percentile(
            df["reference_rg_percentile_in_tool"]
        )
    if "reference_span_percentile_in_tool" in df.columns:
        df["reference_span_typicality_in_tool"] = folded_typicality_from_percentile(
            df["reference_span_percentile_in_tool"]
        )

    summary = (
        df.groupby(["glycan_class", "tool"], dropna=False)
        .agg(
            n_sites=(SITE_KEY_COL, "nunique"),
            median_min_rmsd_to_reference=("min_rmsd_to_reference", "median"),
            median_reference_rg_percentile=("reference_rg_percentile_in_tool", "median"),
            median_reference_span_percentile=("reference_span_percentile_in_tool", "median"),
            median_reference_rg_typicality=("reference_rg_typicality_in_tool", "median"),
            median_reference_span_typicality=("reference_span_typicality_in_tool", "median"),
            median_delta_rg=("delta_median_rg_vs_reference", "median"),
            median_delta_span=("delta_median_span_vs_reference", "median"),
        )
        .reset_index()
    )

    out = table_dir / "table_global_shape_summary.csv"
    summary.to_csv(out, index=False)
    print(f"[OK] wrote {out}")

    grouped_barplot(
        df,
        x="glycan_class",
        hue="tool",
        y="min_rmsd_to_reference",
        title="Best agreement to experimental glycan structure",
        ylabel="Median minimal RMSD to reference (Å)",
        fig_dir=fig_dir,
        table_dir=table_dir,
        name="fig_global_min_rmsd_by_class",
        hue_order=TOOL_ORDER,
    )
    write_paired_bootstrap_differences(
        df,
        x="glycan_class",
        hue="tool",
        y="min_rmsd_to_reference",
        table_dir=table_dir,
        name="fig_global_min_rmsd_by_class",
        hue_pairs=[("glycoshield", "glycoshape")],
    )

    grouped_barplot(
        df,
        x="glycan_class",
        hue="tool",
        y="reference_rg_typicality_in_tool",
        title="Reference Rg typicality in tool ensemble",
        ylabel="Folded typicality score (0=edge, 1=central)",
        fig_dir=fig_dir,
        table_dir=table_dir,
        name="fig_global_rg_typicity_by_class",
        hue_order=TOOL_ORDER,
        y_limits=(0.0, 1.05),
    )
    write_paired_bootstrap_differences(
        df,
        x="glycan_class",
        hue="tool",
        y="reference_rg_typicality_in_tool",
        table_dir=table_dir,
        name="fig_global_rg_typicity_by_class",
        hue_pairs=[("glycoshield", "glycoshape")],
    )

    grouped_barplot(
        df,
        x="glycan_class",
        hue="tool",
        y="reference_span_typicality_in_tool",
        title="Reference span typicality in tool ensemble",
        ylabel="Folded typicality score (0=edge, 1=central)",
        fig_dir=fig_dir,
        table_dir=table_dir,
        name="fig_global_span_typicity_by_class",
        hue_order=TOOL_ORDER,
        y_limits=(0.0, 1.05),
    )
    write_paired_bootstrap_differences(
        df,
        x="glycan_class",
        hue="tool",
        y="reference_span_typicality_in_tool",
        table_dir=table_dir,
        name="fig_global_span_typicity_by_class",
        hue_pairs=[("glycoshield", "glycoshape")],
    )


def write_local_conformation_outputs(
    results_dir: Path,
    fig_dir: Path,
    table_dir: Path,
    triple_sites: pd.DataFrame,
) -> None:
    torsions = restrict_to_triple_sites(
        read_csv(results_dir / "compare_local_torsions.csv"),
        triple_sites,
        "local torsions",
    )
    puckering = restrict_to_triple_sites(
        read_csv(results_dir / "compare_local_puckering.csv"),
        triple_sites,
        "local puckering",
    )

    torsion_metrics = [
        "median_circular_distance_phi",
        "median_circular_distance_psi",
        "median_circular_distance_omega",
    ]
    available_torsion_metrics = [c for c in torsion_metrics if c in torsions.columns]

    if available_torsion_metrics:
        torsion_site = site_level_table(
            torsions,
            hue="tool",
            metrics=available_torsion_metrics,
            aggfunc="median",
        )
        torsion_summary = (
            torsion_site.groupby(["glycan_class", "tool"], dropna=False)
            .agg(
                n_sites=(SITE_KEY_COL, "nunique"),
                **{f"median_{c}": (c, "median") for c in available_torsion_metrics},
            )
            .reset_index()
        )

        out = table_dir / "table_local_torsion_summary.csv"
        torsion_summary.to_csv(out, index=False)
        print(f"[OK] wrote {out}")

        for metric in available_torsion_metrics:
            grouped_barplot(
                torsion_site,
                x="glycan_class",
                hue="tool",
                y=metric,
                title=f"Local torsion agreement: {metric.replace('median_circular_distance_', '')}",
                ylabel="Median circular distance to reference (degrees)",
                fig_dir=fig_dir,
                table_dir=table_dir,
                name=f"fig_local_{metric}_by_class",
                hue_order=TOOL_ORDER,
            )
            write_paired_bootstrap_differences(
                torsion_site,
                x="glycan_class",
                hue="tool",
                y=metric,
                table_dir=table_dir,
                name=f"fig_local_{metric}_by_class",
                hue_pairs=[("glycoshield", "glycoshape")],
            )

    if "pucker_agreement_fraction" in puckering.columns:
        puckering = puckering.copy()
        puckering["pucker_disagreement_fraction"] = 1.0 - puckering["pucker_agreement_fraction"]
        pucker_site = site_level_table(
            puckering,
            hue="tool",
            metrics=["pucker_agreement_fraction", "pucker_disagreement_fraction", "n_distinct_tool_pucker_classes"],
            aggfunc="mean",
        )

        pucker_summary = (
            pucker_site.groupby(["glycan_class", "tool"], dropna=False)
            .agg(
                n_sites=(SITE_KEY_COL, "nunique"),
                median_pucker_agreement_fraction=("pucker_agreement_fraction", "median"),
                mean_pucker_agreement_fraction=("pucker_agreement_fraction", "mean"),
                mean_pucker_disagreement_fraction=("pucker_disagreement_fraction", "mean"),
                median_n_distinct_tool_pucker_classes=("n_distinct_tool_pucker_classes", "median"),
            )
            .reset_index()
        )

        out = table_dir / "table_local_puckering_summary.csv"
        pucker_summary.to_csv(out, index=False)
        print(f"[OK] wrote {out}")

        grouped_barplot(
            pucker_site,
            x="glycan_class",
            hue="tool",
            y="pucker_disagreement_fraction",
            title="Ring puckering disagreement with reference",
            ylabel="Mean disagreement fraction per site",
            fig_dir=fig_dir,
            table_dir=table_dir,
            name="fig_local_pucker_disagreement_by_class",
            hue_order=TOOL_ORDER,
        )
        write_paired_bootstrap_differences(
            pucker_site,
            x="glycan_class",
            hue="tool",
            y="pucker_disagreement_fraction",
            table_dir=table_dir,
            name="fig_local_pucker_disagreement_by_class",
            hue_pairs=[("glycoshield", "glycoshape")],
        )


def write_shielding_outputs(
    results_dir: Path,
    fig_dir: Path,
    table_dir: Path,
    triple_sites: pd.DataFrame,
) -> None:
    profiles = restrict_to_triple_sites(
        read_csv(results_dir / "compare_shielding_profiles.csv"),
        triple_sites,
        "shielding profiles",
    )
    summary = restrict_to_triple_sites(
        read_csv(results_dir / "compare_shielding_summary.csv"),
        triple_sites,
        "shielding model summary",
    )

    profile_metrics = [
        "pearson_mean_delta_sasa",
        "spearman_mean_delta_sasa",
        "mae_mean_delta_sasa_a2",
        "masked_residue_jaccard",
    ]
    profile_site = site_level_table(
        profiles,
        hue="comparison",
        metrics=profile_metrics,
        aggfunc="median",
    )

    if not profile_site.empty:
        profile_summary = (
            profile_site.groupby(["glycan_class", "comparison"], dropna=False)
            .agg(
                n_sites=(SITE_KEY_COL, "nunique"),
                median_pearson_delta_sasa=("pearson_mean_delta_sasa", "median"),
                median_spearman_delta_sasa=("spearman_mean_delta_sasa", "median"),
                median_mae_delta_sasa_a2=("mae_mean_delta_sasa_a2", "median"),
                median_masked_residue_jaccard=("masked_residue_jaccard", "median"),
            )
            .reset_index()
        )

        out = table_dir / "table_shielding_profile_summary.csv"
        profile_summary.to_csv(out, index=False)
        print(f"[OK] wrote {out}")

        grouped_barplot(
            profile_site,
            x="glycan_class",
            hue="comparison",
            y="pearson_mean_delta_sasa",
            title="Similarity of shielding profiles",
            ylabel="Median Pearson correlation of mean ΔSASA",
            fig_dir=fig_dir,
            table_dir=table_dir,
            name="fig_shielding_pearson_delta_sasa_by_class",
            hue_order=COMPARISON_ORDER,
        )

        grouped_barplot(
            profile_site,
            x="glycan_class",
            hue="comparison",
            y="masked_residue_jaccard",
            title="Overlap of masked residues",
            ylabel="Median Jaccard index",
            fig_dir=fig_dir,
            table_dir=table_dir,
            name="fig_shielding_masked_residue_jaccard_by_class",
            hue_order=COMPARISON_ORDER,
        )

    if {"glycan_class", "tool", "total_delta_sasa_a2", "masked_residue_fraction"}.issubset(summary.columns):
        shielding_site = site_level_table(
            summary,
            hue="tool",
            metrics=["total_delta_sasa_a2", "masked_residue_fraction", "n_masked_residues"],
            aggfunc="median",
        )

        shielding_model_summary = (
            shielding_site.groupby(["glycan_class", "tool"], dropna=False)
            .agg(
                n_sites=(SITE_KEY_COL, "nunique"),
                median_total_delta_sasa_a2=("total_delta_sasa_a2", "median"),
                median_masked_residue_fraction=("masked_residue_fraction", "median"),
                median_n_masked_residues=("n_masked_residues", "median"),
            )
            .reset_index()
        )

        out = table_dir / "table_shielding_model_summary.csv"
        shielding_model_summary.to_csv(out, index=False)
        print(f"[OK] wrote {out}")

        grouped_barplot(
            shielding_site,
            x="glycan_class",
            hue="tool",
            y="total_delta_sasa_a2",
            title="Total protein surface shielding",
            ylabel="Median total ΔSASA (Å²)",
            fig_dir=fig_dir,
            table_dir=table_dir,
            name="fig_shielding_total_delta_sasa_by_class",
            hue_order=MODEL_TOOL_ORDER,
        )
        write_paired_bootstrap_differences(
            shielding_site,
            x="glycan_class",
            hue="tool",
            y="total_delta_sasa_a2",
            table_dir=table_dir,
            name="fig_shielding_total_delta_sasa_by_class",
            hue_pairs=[("glycoshield", "glycoshape"), ("reference", "glycoshield"), ("reference", "glycoshape")],
        )


def write_index(outdir: Path) -> None:
    rows = []
    for subdir in ["figures", "tables"]:
        for path in sorted((outdir / subdir).glob("*")):
            rows.append(
                {
                    "type": subdir[:-1],
                    "path": str(path),
                    "filename": path.name,
                }
            )

    index = pd.DataFrame(rows)
    out = outdir / "deliverables_index.csv"
    index.to_csv(out, index=False)
    print(f"[OK] wrote {out}")


def build_final_figures(results_dir: Path, outdir: Path) -> None:
    fig_dir, table_dir = ensure_dirs(outdir)

    write_dataset_overview_outputs(results_dir, fig_dir, table_dir)
    triple_sites = load_triple_site_manifest(results_dir, table_dir)
    write_global_shape_outputs(results_dir, fig_dir, table_dir, triple_sites)
    write_local_conformation_outputs(results_dir, fig_dir, table_dir, triple_sites)
    write_shielding_outputs(results_dir, fig_dir, table_dir, triple_sites)
    write_index(outdir)

    print("[DONE] final figures and tables generated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final GlycoBench figures and summary tables from layer 4 comparison CSV files."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory containing compare_*.csv files.",
    )
    parser.add_argument(
        "--outdir",
        default="deliverables",
        type=Path,
        help="Output directory for final figures and tables.",
    )

    args = parser.parse_args()
    build_final_figures(args.results_dir, args.outdir)


if __name__ == "__main__":
    main()
