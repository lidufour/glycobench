# GlycoBench

Comparative benchmark of two computational re-glycosylation tools —
**GlycoSHIELD** and **GlycoShape / Re-Glyco** — against experimental
glycosylated structures (the **DIONYSUS** set), along three axes: overall glycan
shape, internal conformation (O-glycosidic torsions + ring puckering), and
protein-surface shielding.

Scientific computation relies on **GlyContact** (≥ 0.3.4) as much as possible;
in-house code is limited to ingesting the heterogeneous input formats,
residue/atom mapping, alignment, cross-tool comparison, and figure generation.

> The detailed design is frozen in [`ARCHITECTURE.md`](ARCHITECTURE.md). This
> README describes the pipeline as it runs today. The figure layer has evolved
> since the original spec (see [§ Figures and tables](#6-figures-and-tables)).

---

## 1. Principle

For each reference structure, we compare what each re-glycosylation tool produces
against the experimental conformer, then the two tools against each other.
Results are **always stratified by glycan type** (small per-class counts), never
pooled globally.

Two readings are reported against the single experimental conformer:

- **Best agreement** — minimum RMSD over the ensemble: can the tool *reach* the
  experimental conformation?
- **Typicality** — does the experimental conformer fall within the region
  populated by the tool: is it typical or rare?

---

## 2. Dataset

23 reference PDB structures, partitioned by comparability:

| Category | n | PDB |
|---|---|---|
| Triple (Shield + Shape + ref.) | 17 | 1OYH, 2J1G, 2PE4, 2QL1, 2QWB, 3AVE, 3DGC, 4B7I, 4C54, 5IW3, 5L2J, 5XWD, 6CMG, 6F9T, 6FB3, 6X3X, 6Z2P |
| GlycoSHIELD-only + ref. | 4 | 1BHG, 3WSR, 3WV0, 6WXU |
| Excluded (glycosylated-residue extraction failure) | 2 | 3AYA, 5A58 |

Stratification by glycan type (`*` = GlycoSHIELD-only case):

- **High-mannose**: 1BHG\*, 2QWB, 4B7I, 6FB3, 6WXU\*, 6X3X
- **Paucimannose**: 1OYH, 2J1G, 2PE4, 3DGC, 5L2J, 6CMG
- **Complex**: 2QL1, 3AVE, 4C54, 5IW3, 5XWD, 6F9T
- **Usable O-glycan**: 3WSR\*, 3WV0\*, 6Z2P

The full partition, annotated sites, GlyTouCan IDs, and input paths live in
[`config/dataset.yaml`](config/dataset.yaml) — the **single source of truth**
(layer 0). Nothing is hard-coded in the pipeline.

---

## 3. Installation

The project uses [`uv`](https://docs.astral.sh/uv/). Python ≥ 3.10.

```bash
git clone https://github.com/lidufour/glycobench.git
cd glycobench
uv sync
```

Main dependencies: `glycontact ≥ 0.3.4`, `biopython`, `mdanalysis`, `numpy`,
`pandas`, `pyyaml`. See [`pyproject.toml`](pyproject.toml).

---

## 4. Architecture (6 layers)

Flow: 3 heterogeneous inputs → unified internal representation → GlyContact
descriptors (3 axes) → comparison → figures.

| Layer | Role | Module |
|---|---|---|
| 0 — Manifest | Source of truth (partition, sites, paths, statuses) | `config/dataset.yaml` |
| 1 — Ingestion + QC | One reader per format → standardized bundle; quality control | `ingest/`, `qc/` |
| 2 — Mapping + alignment | Residue/atom mapping (anchored on the reducing end + tree topology); protein superposition | `mapping/` |
| 3 — Descriptors | Shape, conformation, shielding via GlyContact | `descriptors/` |
| 4 — Comparison + stats | Distances between distributions (circular for torsions), pucker-class agreement, RMSD to experimental, shielding-profile correlation | `compare/` |
| 5 — Figures + tables | Final per-class figures + summary tables | `figures/final.py` |

### GlyContact / in-house boundary

- **GlyContact provides**: graph + connectivity + universal nomenclature; φ/ψ/ω
  torsions; Cremer-Pople; Shrake-Rupley SASA (**1.4 Å** probe);
  `superimpose_glycans` / Kabsch RMSD; contact maps; flexibility.
- **In-house code**: ingestion of the 3 formats + QC; protein mapping and
  alignment; weight and ensemble handling; cross-tool comparison + circular
  statistics; figures; manifest.

---

## 5. Running the pipeline

Orchestration does not go through `stages/` scripts but through **console
entry-points** declared in `pyproject.toml` (`[project.scripts]`). Each step
reads and writes standard paths (by default under `results/`), overridable via
options.

```bash
# 1 · Ingestion — standardized bundles in results/<pdb>/<tool>/
uv run glycobench-read-reference
uv run glycobench-read-glycoshield
uv run glycobench-read-glycoshape
uv run glycobench-build-ingestion-qc

# 2 · Mapping + alignment
uv run glycobench-map-glycan-residues
uv run glycobench-align-glycan-frame
uv run glycobench-align-protein-frame
uv run glycobench-build-glycan-frame-qc

# Analysis manifest (consolidates layers 1+2) + GlyContact-adapter QC
uv run glycobench-build-analysis-manifest        # → results/analysis_manifest.csv
uv run glycobench-build-glycontact-adapter-qc

# 3 · GlyContact descriptors (3 axes)
uv run glycobench-build-glycan-shape-descriptors
uv run glycobench-build-glycan-conformation-descriptors
uv run glycobench-build-glycan-shielding-descriptors

# 4 · Cross-tool comparison → results/compare_*.csv
uv run glycobench-compare-global-shape
uv run glycobench-compare-local-conformation
uv run glycobench-compare-shielding

# 5 · Final figures + tables → deliverables/
uv run glycobench-build-final-figures
```

`glycobench-build-final-figures` accepts `--results-dir` (default `results`) and
`--outdir` (default `deliverables`).

---

## 6. Figures and tables

> **Major change vs `ARCHITECTURE.md`.** The original spec called for
> Ramachandran-style overlays, pucker distributions, RMSD bar charts, on-protein
> shielding heatmaps, and PyMOL sessions. The figure layer has been reworked
> into a single module [`figures/final.py`](src/glycobench/figures/final.py) that
> starts directly from the layer-4 comparison CSVs and produces a set of
> **per-class** figures, paired and annotated with confidence intervals.

Principles of the new generation:

- **Paired triple-site subset.** All figures are restricted to a strictly paired
  subset, identified by a stable site key (`site_key = pdb_id|site`). Comparisons
  stay paired by PDB **and** by glycosylation site.
- **One point per site.** Distributions show one point per paired site, a median
  tick, and a **95 % bootstrap confidence interval** (`BOOTSTRAP_N = 5000`,
  seed `42`).
- **Low-n guardrail.** Low-count groups (n ≤ 3) are greyed out and their CI is
  not drawn — an error bar over one or two sites would give a false impression of
  statistical support.
- **Paired-bootstrap tables.** For each relevant figure, a table reports the
  median difference between tools and its 95 % CI (positive difference = second
  tool minus first).
- **Paired inference.** Per-tool individual CIs can overlap and mislead; it is
  the paired bootstrap of the *difference* that decides.
- Outputs as **PNG (300 dpi) + PDF**, plus a `deliverables_index.csv`.

### Figures produced (`deliverables/figures/`)

| Axis | Figure |
|---|---|
| Overview | `fig_dataset_overview_table` |
| Global shape | `fig_global_min_rmsd_by_class` |
| Global shape | `fig_global_rg_typicity_by_class` |
| Global shape | `fig_global_span_typicity_by_class` |
| Conformation | `fig_local_median_circular_distance_{phi,psi,omega}_by_class` |
| Conformation | `fig_local_pucker_{agreement,disagreement}_by_class` |
| Shielding | `fig_shielding_pearson_delta_sasa_by_class` |
| Shielding | `fig_shielding_masked_residue_jaccard_by_class` |
| Shielding | `fig_shielding_total_delta_sasa_by_class` |

### Tables produced (`deliverables/tables/`)

For each figure: a `*_summary.csv` table (medians + n per bar) and, where
applicable, a `*_paired_bootstrap.csv` table (median difference + 95 % CI). These
are complemented by the global summary tables (`table_global_shape_summary`,
`table_local_torsion_summary`, `table_local_puckering_summary`,
`table_shielding_model_summary`, `table_shielding_profile_summary`,
`table_dataset_overview`, `table_triple_site_counts`).

---

## 7. Methodological decisions (locked)

1. **Double alignment** — superposition on the whole protein (where the glycan is
   placed → relevant for shielding) **and** local fit on the reducing end via
   `superimpose_glycans` (intrinsic tree shape).
2. **Ensembles** — comparisons on normalized distributions (unit area, insensitive
   to ensemble size), with GlycoShape cluster weights honored.
3. **SASA recomputed in-house** — a single engine (GlyContact / Shrake-Rupley),
   1.4 Å probe, applied identically to all three sources. Tool-provided SASA is
   used only for cross-checking.
4. **Canonical GlycoSHIELD ensemble** — the 30 models of `merged_traj_pdb.pdb` for
   the 21 cases with an output, ensuring parity with GlycoShape's 30 conformers.
   The full XTC, where it exists, feeds only an optional flexibility metric.
5. **The experimental conformer is not ground truth** — it is a single conformer,
   potentially affected by crystal packing.

### Acknowledged limitations

- Per-class conclusions rest on n = 6–8 sites; after pairing, some comparisons
  (notably ω) drop to very low n — hence the low-n guardrail on the figures.
- Both global-shape metrics (minimum RMSD and typicality) reward ensemble
  dispersion: they are not independent.

---

## 8. Repository layout

```text
glycobench/
├── pyproject.toml              # uv ; dependencies + console entry-points
├── ARCHITECTURE.md             # frozen design spec
├── README.md                   # this document
├── config/
│   └── dataset.yaml            # layer 0 — manifest
├── src/glycobench/
│   ├── cli/                    # entry-points: readers, mapping, alignment, QC, manifest
│   ├── ingest/                 # 1 · per-format readers
│   ├── qc/                     # 1 · frame quality control
│   ├── mapping/                # 2 · mapping + alignment
│   ├── descriptors/            # 3 · GlyContact adapters (shape, conformation, shielding)
│   ├── compare/                # 4 · cross-tool stats
│   └── figures/final.py        # 5 · final figures + tables
├── results/                    # intermediates: per-PDB bundles + descriptor/comparison CSVs
└── deliverables/               # final figures + tables (+ deliverables_index.csv)
```

---

## 9. Data availability

The per-residue SASA results CSV (`results/glycan_shielding_residue_sasa.csv`)
**is not hosted on GitHub**: its size exceeds GitHub's file-size limit, so it is
excluded via `.gitignore`. It is a generated intermediate and can be regenerated
locally by running the shielding-descriptor step:

```bash
uv run glycobench-build-glycan-shielding-descriptors
```

All other intermediates and the final deliverables are versioned, so figures and
tables can be rebuilt without it once the shielding descriptors are regenerated.

---

## 10. Reproducibility

- Single versioned manifest (`config/dataset.yaml`); no paths hard-coded in the
  code.
- Fixed-seed bootstrap (`42`) for reproducible CIs.
- Environment locked by `uv.lock`.
- The intermediate CSVs in `results/` and the outputs in `deliverables/` allow
  figures and tables to be regenerated without rerunning the full upstream
  computation (with the exception noted in § 9).
