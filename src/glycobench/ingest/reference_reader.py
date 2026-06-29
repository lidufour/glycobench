from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import json
import yaml


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


@dataclass(frozen=True)
class ResidueKey:
    resname: str
    chain: str
    resnum: int
    icode: str = ""

    def label(self) -> str:
        suffix = self.icode if self.icode else ""
        return f"{self.resname}{self.resnum}{suffix}_{self.chain}"


@dataclass
class AtomRecord:
    line: str
    residue: ResidueKey


@dataclass
class LinkRecord:
    atom1: str
    residue1: ResidueKey
    atom2: str
    residue2: ResidueKey


def load_manifest(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_atom_line(line: str) -> AtomRecord:
    resname = line[17:20].strip()
    chain = line[21].strip()
    resnum = int(line[22:26])
    icode = line[26].strip()

    residue = ResidueKey(
        resname=resname,
        chain=chain,
        resnum=resnum,
        icode=icode,
    )
    return AtomRecord(line=line.rstrip("\n"), residue=residue)


def parse_link_line(line: str) -> LinkRecord:
    atom1 = line[12:16].strip()
    resname1 = line[17:20].strip()
    chain1 = line[21].strip()
    resnum1 = int(line[22:26])
    icode1 = line[26].strip()

    atom2 = line[42:46].strip()
    resname2 = line[47:50].strip()
    chain2 = line[51].strip()
    resnum2 = int(line[52:56])
    icode2 = line[56].strip()

    return LinkRecord(
        atom1=atom1,
        residue1=ResidueKey(resname1, chain1, resnum1, icode1),
        atom2=atom2,
        residue2=ResidueKey(resname2, chain2, resnum2, icode2),
    )


def read_pdb(pdb_path: Path) -> tuple[list[AtomRecord], list[LinkRecord]]:
    atoms: list[AtomRecord] = []
    links: list[LinkRecord] = []

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"}:
                atoms.append(parse_atom_line(line))

            elif record == "LINK":
                links.append(parse_link_line(line))

    return atoms, links


def site_from_manifest(site_dict: dict) -> ResidueKey:
    return ResidueKey(
        resname=site_dict["resname"],
        chain=str(site_dict["chain"]),
        resnum=int(site_dict["resnum"]),
        icode=str(site_dict.get("icode", "") or ""),
    )


def is_protein_residue(residue: ResidueKey) -> bool:
    return residue.resname in STANDARD_AA


def find_glycan_root(site: ResidueKey, links: list[LinkRecord]) -> ResidueKey:
    candidates: list[ResidueKey] = []

    for link in links:
        if link.residue1 == site and not is_protein_residue(link.residue2):
            candidates.append(link.residue2)

        if link.residue2 == site and not is_protein_residue(link.residue1):
            candidates.append(link.residue1)

    unique_candidates = sorted(set(candidates), key=lambda r: (r.chain, r.resnum, r.resname))

    if len(unique_candidates) == 1:
        return unique_candidates[0]

    if not unique_candidates:
        nearby = [
            link for link in links
            if link.residue1.chain == site.chain and link.residue1.resnum == site.resnum
            or link.residue2.chain == site.chain and link.residue2.resnum == site.resnum
        ]
        details = [f"{l.residue1.label()}-{l.atom1} <-> {l.residue2.label()}-{l.atom2}" for l in nearby]
        raise ValueError(
            f"Aucun sucre racine trouvé pour le site {site.label()}. "
            f"LINK proches trouvés: {details}"
        )

    raise ValueError(
        f"Plusieurs sucres racines trouvés pour le site {site.label()}: "
        f"{[r.label() for r in unique_candidates]}"
    )


def build_link_graph(links: list[LinkRecord]) -> dict[ResidueKey, set[ResidueKey]]:
    graph: dict[ResidueKey, set[ResidueKey]] = defaultdict(set)

    for link in links:
        graph[link.residue1].add(link.residue2)
        graph[link.residue2].add(link.residue1)

    return graph


EXCLUDED_NON_GLYCAN = {
    "HOH", "WAT",
    "NA", "K", "CL", "MG", "MN", "CA", "ZN", "FE", "CU", "CO", "NI",
    "SO4", "PO4", "ACT", "EDO", "GOL",
}


def collect_connected_glycan_residues(
    root: ResidueKey,
    site: ResidueKey,
    links: list[LinkRecord],
) -> set[ResidueKey]:
    graph = build_link_graph(links)

    visited: set[ResidueKey] = set()
    queue: list[ResidueKey] = [root]

    while queue:
        residue = queue.pop(0)

        if residue in visited:
            continue

        if residue == site:
            continue

        if is_protein_residue(residue):
            continue

        if residue.resname.upper() in EXCLUDED_NON_GLYCAN:
            continue

        visited.add(residue)

        for neighbor in graph.get(residue, set()):
            if neighbor in visited:
                continue

            if neighbor == site:
                continue

            if is_protein_residue(neighbor):
                continue

            if neighbor.resname.upper() in EXCLUDED_NON_GLYCAN:
                continue

            queue.append(neighbor)

    return visited


def write_glycan_pdb(
    atoms: list[AtomRecord],
    residues: set[ResidueKey],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_lines = [atom.line for atom in atoms if atom.residue in residues]

    with output_path.open("w", encoding="utf-8") as handle:
        for line in selected_lines:
            handle.write(line + "\n")
        handle.write("END\n")

    return len(selected_lines)


def resolve_reference_path(entry: dict, manifest: dict) -> Path:
    root = Path(manifest["roots"]["reference"])
    return root / entry["reference"]["glyco"]


def resolve_reference_apo_path(entry: dict, manifest: dict) -> Path:
    root = Path(manifest["roots"]["reference"])
    return root / entry["reference"]["apo"]


def read_reference_entry(entry: dict, manifest: dict, results_root: Path) -> list[dict]:
    pdb_id = entry["pdb_id"]
    pdb_path = resolve_reference_path(entry, manifest)
    apo_path = resolve_reference_apo_path(entry, manifest)

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB de référence introuvable: {pdb_path}")

    atoms, links = read_pdb(pdb_path)

    reports: list[dict] = []

    for site_dict in entry.get("sites", []):
        site = site_from_manifest(site_dict)
        root = find_glycan_root(site, links)
        glycan_residues = collect_connected_glycan_residues(root, site, links)

        site_dir = results_root / pdb_id / "reference" / site.label()
        glycan_pdb = site_dir / "glycan.pdb"
        atom_count = write_glycan_pdb(atoms, glycan_residues, glycan_pdb)

        report = {
            "pdb_id": pdb_id,
            "site": asdict(site),
            "glycan_root": asdict(root),
            "n_glycan_residues": len(glycan_residues),
            "n_glycan_atoms": atom_count,
            "glycan_residues": [asdict(r) for r in sorted(glycan_residues, key=lambda x: (x.chain, x.resnum, x.resname))],
            "reference_pdb": str(pdb_path),
            "reference_apo_pdb": str(apo_path),
            "output_glycan_pdb": str(glycan_pdb),
        }

        with (site_dir / "reference_bundle.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        reports.append(report)

    return reports
