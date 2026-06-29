from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict, deque
import itertools
import json
import math
import zipfile

from glycobench.ingest.reference_reader import (
    ResidueKey,
    STANDARD_AA,
    site_from_manifest,
)


EXCLUDED_NON_GLYCAN = {
    "HOH", "WAT", "SOL", "NA", "CL", "K", "CA", "MG", "ZN", "MN",
    "SO4", "PO4", "EDO", "GOL", "PEG", "ACT", "DMS", "DOD",
}


@dataclass
class ModelAtom:
    model: int
    line: str
    atom_name: str
    element: str
    residue: ResidueKey
    x: float
    y: float
    z: float

    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def parse_atom_line(line: str, model: int) -> ModelAtom:
    atom_name = line[12:16].strip()
    resname = line[17:20].strip()
    chain = line[21].strip()
    resnum = int(line[22:26])
    icode = line[26].strip()
    element = line[76:78].strip() if len(line) >= 78 else ""

    if not element:
        element = atom_name[0]

    return ModelAtom(
        model=model,
        line=line.rstrip("\n"),
        atom_name=atom_name,
        element=element.upper(),
        residue=ResidueKey(
            resname=resname,
            chain=chain,
            resnum=resnum,
            icode=icode,
        ),
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
    )


def parse_multimodel_pdb(lines: list[str]) -> dict[int, list[ModelAtom]]:
    models: dict[int, list[ModelAtom]] = defaultdict(list)
    current_model = 1
    saw_model_record = False

    for line in lines:
        record = line[:6].strip()

        if record == "MODEL":
            saw_model_record = True
            current_model = int(line.split()[1])
            continue

        if record in {"ATOM", "HETATM"}:
            model = current_model if saw_model_record else 1
            models[model].append(parse_atom_line(line, model=model))

    return dict(sorted(models.items()))


def uniquify_repeated_residue_blocks(
    models: dict[int, list[ModelAtom]],
) -> dict[int, list[ModelAtom]]:
    """Split repeated PDB residue labels into distinct physical residue instances.

    Some GlycoShield merged PDB files reuse the same chain/resnum/resname labels
    for several copies of the same glycan. If we keep only resname+chain+resnum,
    these copies are artificially merged. We therefore add an artificial iCode
    occurrence tag only when the same residue label appears in several
    non-contiguous blocks in one model.
    """
    fixed_models: dict[int, list[ModelAtom]] = {}

    for model_id, atoms in models.items():
        blocks: list[tuple[ResidueKey, list[ModelAtom]]] = []

        current_key: ResidueKey | None = None
        current_atoms: list[ModelAtom] = []

        for atom in atoms:
            key = atom.residue

            if current_key is None:
                current_key = key
                current_atoms = [atom]
                continue

            if key == current_key:
                current_atoms.append(atom)
                continue

            blocks.append((current_key, current_atoms))
            current_key = key
            current_atoms = [atom]

        if current_key is not None:
            blocks.append((current_key, current_atoms))

        total_occurrences: dict[ResidueKey, int] = defaultdict(int)

        for key, _ in blocks:
            total_occurrences[key] += 1

        seen_occurrences: dict[ResidueKey, int] = defaultdict(int)
        fixed_atoms: list[ModelAtom] = []

        for key, block_atoms in blocks:
            seen_occurrences[key] += 1

            if total_occurrences[key] > 1:
                new_key = ResidueKey(
                    resname=key.resname,
                    chain=key.chain,
                    resnum=key.resnum,
                    icode=f"occ{seen_occurrences[key]}",
                )
            else:
                new_key = key

            for atom in block_atoms:
                fixed_atoms.append(
                    ModelAtom(
                        model=atom.model,
                        line=atom.line,
                        atom_name=atom.atom_name,
                        element=atom.element,
                        residue=new_key,
                        x=atom.x,
                        y=atom.y,
                        z=atom.z,
                    )
                )

        fixed_models[model_id] = fixed_atoms

    return fixed_models


def is_hydrogen(atom: ModelAtom) -> bool:
    return atom.element == "H" or atom.atom_name.upper().startswith("H")


def is_glycan_residue(residue: ResidueKey) -> bool:
    return (
        residue.resname not in STANDARD_AA
        and residue.resname not in EXCLUDED_NON_GLYCAN
    )


def heavy_atoms(atoms: list[ModelAtom]) -> list[ModelAtom]:
    return [atom for atom in atoms if not is_hydrogen(atom)]


def distance(atom_a: ModelAtom, atom_b: ModelAtom) -> float:
    ax, ay, az = atom_a.coord()
    bx, by, bz = atom_b.coord()
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def min_distance(atoms_a: list[ModelAtom], atoms_b: list[ModelAtom]) -> float:
    return min(distance(atom_a, atom_b) for atom_a in atoms_a for atom_b in atoms_b)


def residue_atoms_by_key(atoms: list[ModelAtom]) -> dict[ResidueKey, list[ModelAtom]]:
    grouped: dict[ResidueKey, list[ModelAtom]] = defaultdict(list)

    for atom in atoms:
        grouped[atom.residue].append(atom)

    return dict(grouped)


def find_site_atoms(model_atoms: list[ModelAtom], site: ResidueKey) -> list[ModelAtom]:
    atoms = [atom for atom in model_atoms if atom.residue == site]

    if atoms:
        return atoms

    same_resnum = sorted(
        {
            atom.residue.label()
            for atom in model_atoms
            if atom.residue.resname == site.resname
            and atom.residue.resnum == site.resnum
        }
    )

    raise ValueError(
        f"Site {site.label()} introuvable dans le PDB GlycoShield. "
        f"Résidus avec même nom/numéro disponibles: {same_resnum}"
    )


def build_glycan_components(
    model_atoms: list[ModelAtom],
    bond_cutoff: float = 1.75,
) -> list[set[ResidueKey]]:
    glycan_atoms = [
        atom
        for atom in heavy_atoms(model_atoms)
        if is_glycan_residue(atom.residue)
    ]

    by_residue = residue_atoms_by_key(glycan_atoms)
    residues = sorted(
        by_residue,
        key=lambda residue: (
            residue.chain,
            residue.resnum,
            residue.resname,
            residue.icode,
        ),
    )

    graph: dict[ResidueKey, set[ResidueKey]] = {
        residue: set()
        for residue in residues
    }

    for residue_a, residue_b in itertools.combinations(residues, 2):
        if min_distance(by_residue[residue_a], by_residue[residue_b]) <= bond_cutoff:
            graph[residue_a].add(residue_b)
            graph[residue_b].add(residue_a)

    components: list[set[ResidueKey]] = []
    seen: set[ResidueKey] = set()

    for residue in residues:
        if residue in seen:
            continue

        component: set[ResidueKey] = set()
        queue: deque[ResidueKey] = deque([residue])
        seen.add(residue)

        while queue:
            current = queue.popleft()
            component.add(current)

            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def assign_components_to_sites(
    model_atoms: list[ModelAtom],
    sites: list[ResidueKey],
    components: list[set[ResidueKey]],
    max_anchor_distance: float = 7.0,
) -> dict[ResidueKey, tuple[set[ResidueKey], ResidueKey, float]]:
    by_residue = residue_atoms_by_key(heavy_atoms(model_atoms))
    site_atoms_by_site: dict[ResidueKey, list[ModelAtom]] = {}

    for site in sites:
        site_atoms = heavy_atoms(find_site_atoms(model_atoms, site))

        if not site_atoms:
            raise ValueError(f"Aucun atome lourd trouvé pour le site {site.label()}.")

        site_atoms_by_site[site] = site_atoms

    candidates: list[tuple[float, int, ResidueKey, ResidueKey]] = []

    for component_id, component in enumerate(components):
        for site in sites:
            residue_distances = [
                (
                    min_distance(site_atoms_by_site[site], by_residue[residue]),
                    residue,
                )
                for residue in component
            ]

            best_distance, anchor_residue = min(residue_distances, key=lambda x: x[0])
            candidates.append((best_distance, component_id, site, anchor_residue))

    candidates.sort(key=lambda x: x[0])

    final_assignments: dict[ResidueKey, tuple[set[ResidueKey], ResidueKey, float]] = {}
    used_components: set[int] = set()
    used_sites: set[ResidueKey] = set()

    for anchor_distance, component_id, site, anchor_residue in candidates:
        if component_id in used_components:
            continue

        if site in used_sites:
            continue

        if anchor_distance > max_anchor_distance:
            continue

        final_assignments[site] = (
            components[component_id],
            anchor_residue,
            anchor_distance,
        )
        used_components.add(component_id)
        used_sites.add(site)

    for site in sites:
        if site not in final_assignments:
            site_candidates = [
                (distance, component_id, anchor)
                for distance, component_id, candidate_site, anchor in candidates
                if candidate_site == site
            ]

            if site_candidates:
                best_distance, _, best_anchor = min(site_candidates, key=lambda x: x[0])
                print(
                    f"[WARN] {site.label()} non assigné dans GlycoShield "
                    f"(glycane le plus proche: {best_anchor.label()}, "
                    f"d={best_distance:.3f} Å)."
                )
            else:
                print(f"[WARN] {site.label()} non assigné dans GlycoShield.")

    return final_assignments


def read_glycoshield_pdb_lines(entry: dict, manifest: dict) -> tuple[list[str], str]:
    root = Path(manifest["roots"]["glycoshield"])
    glycoshield = entry["glycoshield"]
    canonical = glycoshield["canonical_pdb"]

    if glycoshield.get("form") == "complete":
        pdb_path = root / glycoshield["dir"] / canonical

        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB canonique GlycoShield introuvable: {pdb_path}")

        text = pdb_path.read_text(encoding="utf-8", errors="replace")
        return text.splitlines(), str(pdb_path)

    zip_path = root / glycoshield["zip"]

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip GlycoShield introuvable: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        member = canonical if canonical in names else None

        if member is None:
            matches = [
                name for name in names
                if name.endswith(Path(canonical).name)
            ]

            if len(matches) == 1:
                member = matches[0]
            else:
                raise FileNotFoundError(
                    f"{canonical} introuvable dans {zip_path}. "
                    f"Candidats: {matches}"
                )

        content = archive.read(member).decode("utf-8", errors="replace")
        return content.splitlines(), f"{zip_path}!{member}"


def write_site_ensemble(
    models: dict[int, list[ModelAtom]],
    component: set[ResidueKey],
    output_path: Path,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atoms_per_model: list[int] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for model_id, atoms in models.items():
            selected = [
                atom for atom in atoms
                if atom.residue in component
            ]
            atoms_per_model.append(len(selected))

            handle.write(f"MODEL     {model_id:4d}\n")

            for atom in selected:
                handle.write(atom.line + "\n")

            handle.write("ENDMDL\n")

        handle.write("END\n")

    unique_counts = set(atoms_per_model)

    if len(unique_counts) != 1:
        raise ValueError(
            f"Nombre d'atomes variable selon les modèles: {sorted(unique_counts)}"
        )

    return len(models), atoms_per_model[0] if atoms_per_model else 0


def read_glycoshield_entry(
    entry: dict,
    manifest: dict,
    results_root: Path,
) -> list[dict]:
    if not entry.get("glycoshield", {}).get("available", False):
        return []

    pdb_id = entry["pdb_id"]

    lines, source = read_glycoshield_pdb_lines(entry, manifest)
    models = parse_multimodel_pdb(lines)
    models = uniquify_repeated_residue_blocks(models)

    if not models:
        raise ValueError(f"Aucun modèle lu dans {source}")

    first_model_id = sorted(models)[0]
    first_model_atoms = models[first_model_id]

    sites = [
        site_from_manifest(site_dict)
        for site_dict in entry.get("sites", [])
    ]

    components = build_glycan_components(first_model_atoms)
    assignments = assign_components_to_sites(
        first_model_atoms,
        sites,
        components,
    )

    reports: list[dict] = []

    for site, (component, anchor_residue, anchor_distance) in assignments.items():
        site_dir = results_root / pdb_id / "glycoshield" / site.label()
        ensemble_pdb = site_dir / "glycan_ensemble.pdb"

        n_models, atoms_per_model = write_site_ensemble(
            models=models,
            component=component,
            output_path=ensemble_pdb,
        )

        report = {
            "pdb_id": pdb_id,
            "site": asdict(site),
            "source_pdb": source,
            "n_models": n_models,
            "n_glycan_residues": len(component),
            "n_glycan_atoms_per_model": atoms_per_model,
            "anchor_residue_nearest_site": asdict(anchor_residue),
            "anchor_distance_angstrom": round(anchor_distance, 3),
            "glycan_residues": [
                asdict(residue)
                for residue in sorted(
                    component,
                    key=lambda r: (r.chain, r.resnum, r.resname, r.icode),
                )
            ],
            "output_glycan_ensemble_pdb": str(ensemble_pdb),
        }

        with (site_dir / "glycoshield_bundle.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        reports.append(report)

    return reports
