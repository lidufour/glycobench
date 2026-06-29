from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict
from typing import Any

import numpy as np

from glycobench.mapping.protein_frame_alignment import Atom, parse_models, read_text_maybe_zip


@dataclass(frozen=True, order=True)
class ResidueKey:
    """Stable identifier for one monosaccharide residue in one PDB file."""

    chain: str
    resnum: int
    icode: str
    resname: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResidueKey":
        return cls(
            chain=str(data.get("chain", "")),
            resnum=int(data["resnum"]),
            icode=str(data.get("icode", "")),
            resname=str(data["resname"]),
        )

    @classmethod
    def from_atom(cls, atom: Atom) -> "ResidueKey":
        return cls(
            chain=atom.chain,
            resnum=atom.resnum,
            icode=atom.icode,
            resname=atom.resname,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "resname": self.resname,
            "chain": self.chain,
            "resnum": self.resnum,
            "icode": self.icode,
        }

    def short(self) -> str:
        icode = self.icode if self.icode else ""
        return f"{self.resname}{self.resnum}{icode}_{self.chain}"


@dataclass(frozen=True)
class ResidueNode:
    key: ResidueKey
    index: int
    normalized_name: str
    n_atoms: int
    n_heavy_atoms: int
    centroid: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "key": self.key.as_dict(),
            "normalized_name": self.normalized_name,
            "n_atoms": self.n_atoms,
            "n_heavy_atoms": self.n_heavy_atoms,
            "centroid": [round(x, 3) for x in self.centroid],
        }


@dataclass(frozen=True)
class GraphEdge:
    residue_a: ResidueKey
    residue_b: ResidueKey
    atom_a: str
    atom_b: str
    distance: float

    def atom_on(self, key: ResidueKey) -> str:
        if key == self.residue_a:
            return self.atom_a
        if key == self.residue_b:
            return self.atom_b
        raise KeyError(key)

    def other(self, key: ResidueKey) -> ResidueKey:
        if key == self.residue_a:
            return self.residue_b
        if key == self.residue_b:
            return self.residue_a
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "residue_a": self.residue_a.as_dict(),
            "residue_b": self.residue_b.as_dict(),
            "atom_a": self.atom_a,
            "atom_b": self.atom_b,
            "distance": round(self.distance, 3),
        }


# Equivalences minimales observées dans les sorties référence, GlycoShield et GlycoShape.
# Le but est de comparer des classes de monosaccharides, pas des noms PDB stricts.
MONOSACCHARIDE_NAME_MAP: dict[str, str] = {
    # N-acetyl-glucosamine-like
    "NAG": "GlcNAc",
    "NDG": "GlcNAc",
    "BGL": "GlcNAc",
    "BGC": "GlcNAc",
    "BGN": "GlcNAc",
    # N-acetyl-galactosamine-like, frequent O-glycan root
    "A2G": "GalNAc",
    "AGA": "GalNAc",
    "GALNAC": "GalNAc",
    # Mannose-like
    "MAN": "Man",
    "BMA": "Man",
    "AMA": "Man",
    # Galactose-like
    "GAL": "Gal",
    "BGA": "Gal",
    "AGA2": "Gal",
    # Fucose-like
    "FUC": "Fuc",
    "FUL": "Fuc",
    "AFU": "Fuc",
    # Sialic acids
    "SIA": "Sia",
    "NAN": "Sia",
    "NANA": "Sia",
    "SLB": "Sia",
    "ANE": "Sia",
}


def normalize_resname(resname: str) -> str:
    """Return a coarse monosaccharide class used for cross-tool matching."""

    clean = resname.strip().upper()
    return MONOSACCHARIDE_NAME_MAP.get(clean, clean)


def is_hydrogen(atom: Atom) -> bool:
    atom_name = atom.atom_name.strip().upper()
    if atom_name.startswith("H"):
        return True

    # PDB element field is not parsed in Atom, but it remains in the raw line.
    if len(atom.line) >= 78 and atom.line[76:78].strip().upper() == "H":
        return True

    return False


def atom_xyz(atom: Atom) -> np.ndarray:
    return np.array([atom.x, atom.y, atom.z], dtype=float)


def load_first_model_atoms(pdb_path: str | Path) -> list[Atom]:
    text = read_text_maybe_zip(pdb_path)
    models = parse_models(text)

    if not models:
        return []

    first_model = sorted(models)[0]
    return models[first_model]


def group_atoms_by_residue(atoms: list[Atom]) -> dict[ResidueKey, list[Atom]]:
    grouped: dict[ResidueKey, list[Atom]] = defaultdict(list)

    for atom in atoms:
        grouped[ResidueKey.from_atom(atom)].append(atom)

    return dict(grouped)


def ordered_keys_from_bundle(bundle: dict[str, Any]) -> list[ResidueKey]:
    return [ResidueKey.from_dict(item) for item in bundle.get("glycan_residues", [])]


def resolve_bundle_key(
    key: ResidueKey,
    atoms_by_residue: dict[ResidueKey, list[Atom]],
) -> tuple[ResidueKey | None, str | None]:
    """Resolve a bundle residue key to the corresponding key parsed from the PDB.

    Some reader steps use pseudo insertion codes such as ``occ1`` to separate
    physical GlycoShield copies. These labels are useful in JSON metadata but
    cannot be stored faithfully in the one-character PDB insertion-code field.
    Therefore, the aligned PDB may contain the same residue with an empty icode.
    """

    if key in atoms_by_residue:
        return key, None

    candidates = [
        pdb_key
        for pdb_key in atoms_by_residue
        if (
            pdb_key.chain == key.chain
            and pdb_key.resnum == key.resnum
            and pdb_key.resname == key.resname
        )
    ]

    if len(candidates) == 1:
        # Expected for pseudo insertion codes such as occ1/occ2 stored in JSON metadata.
        return candidates[0], None

    if len(candidates) > 1:
        return None, f"ambiguous PDB residue candidates for bundle residue {key.short()}"

    return None, f"bundle residue absent from PDB atoms: {key.short()}"


def choose_root_key(bundle: dict[str, Any], atoms_by_residue: dict[ResidueKey, list[Atom]]) -> tuple[ResidueKey | None, str, list[str]]:
    """Choose the reducing-end root.

    Reference bundles store an explicit glycan_root. Tool bundles currently do not.
    In the generated bundles, the first glycan_residue is the reducing-end residue.
    The anchor_residue_nearest_site is deliberately *not* used as a root because it
    can be a residue close to the protein surface rather than the covalent root.
    """

    warnings: list[str] = []

    if "glycan_root" in bundle:
        root = ResidueKey.from_dict(bundle["glycan_root"])
        resolved_root, warning = resolve_bundle_key(root, atoms_by_residue)
        if resolved_root is not None:
            if warning:
                warnings.append(warning)
            return resolved_root, "bundle.glycan_root", warnings
        if warning:
            warnings.append(warning)

    ordered_keys = ordered_keys_from_bundle(bundle)

    if ordered_keys:
        root = ordered_keys[0]
        resolved_root, warning = resolve_bundle_key(root, atoms_by_residue)
        if resolved_root is not None:
            if warning:
                warnings.append(warning)
            return resolved_root, "first_bundle_glycan_residue", warnings
        if warning:
            warnings.append(warning)

    if atoms_by_residue:
        root = sorted(atoms_by_residue)[0]
        warnings.append(f"fallback root selected from PDB order: {root.short()}")
        return root, "first_pdb_residue", warnings

    return None, "missing", ["no residue found in PDB"]


def residue_centroid(atoms: list[Atom], heavy_only: bool = True) -> tuple[float, float, float]:
    selected = [atom for atom in atoms if not heavy_only or not is_hydrogen(atom)]

    if not selected:
        selected = atoms

    xyz = np.array([[atom.x, atom.y, atom.z] for atom in selected], dtype=float)
    centroid = xyz.mean(axis=0)
    return float(centroid[0]), float(centroid[1]), float(centroid[2])


def build_residue_nodes(
    bundle: dict[str, Any],
    atoms_by_residue: dict[ResidueKey, list[Atom]],
) -> tuple[dict[ResidueKey, ResidueNode], list[str]]:
    warnings: list[str] = []
    nodes: dict[ResidueKey, ResidueNode] = {}

    ordered_keys = ordered_keys_from_bundle(bundle)
    seen: set[ResidueKey] = set()

    for index, key in enumerate(ordered_keys, start=1):
        resolved_key, warning = resolve_bundle_key(key, atoms_by_residue)

        if resolved_key is None:
            if warning:
                warnings.append(warning)
            continue

        if warning:
            warnings.append(warning)

        atoms = atoms_by_residue[resolved_key]
        heavy_atoms = [atom for atom in atoms if not is_hydrogen(atom)]
        nodes[resolved_key] = ResidueNode(
            key=resolved_key,
            index=index,
            normalized_name=normalize_resname(resolved_key.resname),
            n_atoms=len(atoms),
            n_heavy_atoms=len(heavy_atoms),
            centroid=residue_centroid(atoms, heavy_only=True),
        )
        seen.add(resolved_key)

    # Keep unexpected glycan residues if present, but put them after bundle residues.
    next_index = len(ordered_keys) + 1
    for key in sorted(atoms_by_residue):
        if key in seen:
            continue

        atoms = atoms_by_residue[key]
        heavy_atoms = [atom for atom in atoms if not is_hydrogen(atom)]
        nodes[key] = ResidueNode(
            key=key,
            index=next_index,
            normalized_name=normalize_resname(key.resname),
            n_atoms=len(atoms),
            n_heavy_atoms=len(heavy_atoms),
            centroid=residue_centroid(atoms, heavy_only=True),
        )
        warnings.append(f"PDB residue absent from bundle, kept anyway: {key.short()}")
        next_index += 1

    return nodes, warnings


def min_heavy_atom_distance(atoms_a: list[Atom], atoms_b: list[Atom]) -> tuple[float, str, str]:
    heavy_a = [atom for atom in atoms_a if not is_hydrogen(atom)]
    heavy_b = [atom for atom in atoms_b if not is_hydrogen(atom)]

    best_distance = float("inf")
    best_pair = ("", "")

    for atom_a in heavy_a:
        xyz_a = atom_xyz(atom_a)
        for atom_b in heavy_b:
            distance = float(np.linalg.norm(xyz_a - atom_xyz(atom_b)))
            if distance < best_distance:
                best_distance = distance
                best_pair = (atom_a.atom_name, atom_b.atom_name)

    return best_distance, best_pair[0], best_pair[1]


def build_residue_graph(
    atoms_by_residue: dict[ResidueKey, list[Atom]],
    residue_keys: list[ResidueKey],
    bond_cutoff: float = 1.85,
) -> tuple[dict[ResidueKey, list[GraphEdge]], list[GraphEdge], list[str]]:
    """Infer covalent glycosidic links from short inter-residue heavy-atom distances."""

    warnings: list[str] = []
    edges: list[GraphEdge] = []
    adjacency: dict[ResidueKey, list[GraphEdge]] = {key: [] for key in residue_keys}

    keys = [key for key in residue_keys if key in atoms_by_residue]

    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            distance, atom_a, atom_b = min_heavy_atom_distance(
                atoms_by_residue[key_a],
                atoms_by_residue[key_b],
            )

            if distance <= bond_cutoff:
                edge = GraphEdge(
                    residue_a=key_a,
                    residue_b=key_b,
                    atom_a=atom_a,
                    atom_b=atom_b,
                    distance=distance,
                )
                edges.append(edge)
                adjacency[key_a].append(edge)
                adjacency[key_b].append(edge)

    if len(keys) > 0 and len(edges) != len(keys) - 1:
        warnings.append(
            "unexpected glycan graph edge count: "
            f"n_residues={len(keys)}, n_edges={len(edges)}, expected_tree_edges={len(keys) - 1}"
        )

    return adjacency, edges, warnings


def child_edges(
    adjacency: dict[ResidueKey, list[GraphEdge]],
    parent: ResidueKey,
    previous: ResidueKey | None,
) -> list[tuple[str, str, float, ResidueKey, GraphEdge]]:
    out: list[tuple[str, str, float, ResidueKey, GraphEdge]] = []

    for edge in adjacency.get(parent, []):
        child = edge.other(parent)
        if child == previous:
            continue

        out.append((edge.atom_on(parent), edge.atom_on(child), edge.distance, child, edge))

    return sorted(out, key=lambda item: (item[0], normalize_resname(item[3].resname), item[3].resnum, item[3].chain))


def subtree_signature(
    root: ResidueKey,
    parent: ResidueKey | None,
    adjacency: dict[ResidueKey, list[GraphEdge]],
) -> tuple[Any, ...]:
    children = []

    for parent_atom, child_atom, _distance, child, _edge in child_edges(adjacency, root, parent):
        children.append(
            (
                parent_atom,
                child_atom,
                normalize_resname(child.resname),
                subtree_signature(child, root, adjacency),
            )
        )

    return (normalize_resname(root.resname), tuple(sorted(children)))


def topology_summary(
    root: ResidueKey | None,
    adjacency: dict[ResidueKey, list[GraphEdge]],
) -> list[dict[str, Any]]:
    if root is None:
        return []

    rows: list[dict[str, Any]] = []
    visited: set[ResidueKey] = set()

    def visit(node: ResidueKey, parent: ResidueKey | None, depth: int) -> None:
        visited.add(node)
        for parent_atom, child_atom, distance, child, _edge in child_edges(adjacency, node, parent):
            rows.append(
                {
                    "parent": node.as_dict(),
                    "child": child.as_dict(),
                    "parent_atom": parent_atom,
                    "child_atom": child_atom,
                    "distance": round(distance, 3),
                    "depth": depth + 1,
                }
            )
            if child not in visited:
                visit(child, node, depth + 1)

    visit(root, None, 0)
    return rows


def compatible_names(reference: ResidueKey, source: ResidueKey) -> bool:
    return normalize_resname(reference.resname) == normalize_resname(source.resname)


def map_rooted_trees(
    reference_root: ResidueKey,
    source_root: ResidueKey,
    reference_adjacency: dict[ResidueKey, list[GraphEdge]],
    source_adjacency: dict[ResidueKey, list[GraphEdge]],
) -> tuple[dict[ResidueKey, ResidueKey], list[str]]:
    """Map residues from reference to source using rooted topology and sugar class."""

    mapping: dict[ResidueKey, ResidueKey] = {}
    warnings: list[str] = []

    if not compatible_names(reference_root, source_root):
        warnings.append(
            "root monosaccharide class mismatch: "
            f"reference={reference_root.short()}({normalize_resname(reference_root.resname)}), "
            f"source={source_root.short()}({normalize_resname(source_root.resname)})"
        )

    def recurse(ref_node: ResidueKey, src_node: ResidueKey, ref_parent: ResidueKey | None, src_parent: ResidueKey | None) -> None:
        if ref_node in mapping:
            if mapping[ref_node] != src_node:
                warnings.append(
                    f"conflicting mapping for {ref_node.short()}: "
                    f"{mapping[ref_node].short()} vs {src_node.short()}"
                )
            return

        mapping[ref_node] = src_node

        ref_children = child_edges(reference_adjacency, ref_node, ref_parent)
        src_children = child_edges(source_adjacency, src_node, src_parent)
        unused_source_children = {item[3] for item in src_children}

        for ref_parent_atom, _ref_child_atom, _ref_distance, ref_child, _ref_edge in ref_children:
            ref_class = normalize_resname(ref_child.resname)
            ref_sig = subtree_signature(ref_child, ref_node, reference_adjacency)

            edge_and_subtree_candidates = []
            subtree_candidates = []
            edge_candidates = []
            class_candidates = []

            for src_parent_atom, _src_child_atom, _src_distance, src_child, _src_edge in src_children:
                if src_child not in unused_source_children:
                    continue

                if normalize_resname(src_child.resname) != ref_class:
                    continue

                src_sig = subtree_signature(src_child, src_node, source_adjacency)
                class_candidates.append(src_child)

                if src_parent_atom == ref_parent_atom and src_sig == ref_sig:
                    edge_and_subtree_candidates.append(src_child)
                elif src_sig == ref_sig:
                    subtree_candidates.append(src_child)
                elif src_parent_atom == ref_parent_atom:
                    edge_candidates.append(src_child)

            if len(edge_and_subtree_candidates) == 1:
                selected = edge_and_subtree_candidates[0]
            elif len(edge_and_subtree_candidates) > 1:
                selected = sorted(edge_and_subtree_candidates)[0]
                warnings.append(
                    f"ambiguous child mapping for {ref_child.short()} below {ref_node.short()} "
                    f"using edge {ref_parent_atom} and subtree; selected {selected.short()}"
                )
            elif len(subtree_candidates) == 1:
                selected = subtree_candidates[0]
                warnings.append(
                    f"edge label mismatch but subtree/name match for {ref_child.short()} below {ref_node.short()}; "
                    f"selected {selected.short()}"
                )
            elif len(subtree_candidates) > 1:
                selected = sorted(subtree_candidates)[0]
                warnings.append(
                    f"ambiguous subtree-only mapping for {ref_child.short()} below {ref_node.short()}; "
                    f"selected {selected.short()}"
                )
            elif len(edge_candidates) == 1:
                selected = edge_candidates[0]
                warnings.append(
                    f"subtree mismatch but edge/name match for {ref_child.short()} below {ref_node.short()}; "
                    f"selected {selected.short()}"
                )
            elif len(edge_candidates) > 1:
                selected = sorted(edge_candidates)[0]
                warnings.append(
                    f"ambiguous edge-only child mapping for {ref_child.short()} below {ref_node.short()}; "
                    f"selected {selected.short()}"
                )
            elif len(class_candidates) == 1:
                selected = class_candidates[0]
                warnings.append(
                    f"edge/subtree mismatch for {ref_child.short()} below {ref_node.short()}; "
                    f"selected {selected.short()} by sugar class only"
                )
            elif len(class_candidates) > 1:
                selected = sorted(class_candidates)[0]
                warnings.append(
                    f"ambiguous class-only mapping for {ref_child.short()} below {ref_node.short()}; "
                    f"selected {selected.short()}"
                )
            else:
                warnings.append(
                    f"no source child found for {ref_child.short()} below {ref_node.short()} "
                    f"with edge {ref_parent_atom} and class {ref_class}"
                )
                continue

            unused_source_children.remove(selected)
            recurse(ref_child, selected, ref_node, src_node)

        for extra_source_child in sorted(unused_source_children):
            warnings.append(
                f"extra source child not mapped below {src_node.short()}: {extra_source_child.short()}"
            )

    recurse(reference_root, source_root, None, None)
    return mapping, warnings


def load_structure_for_mapping(
    bundle_path: str | Path,
    pdb_path: str | Path,
    bond_cutoff: float = 1.85,
) -> dict[str, Any]:
    bundle_path = Path(bundle_path)
    pdb_path = Path(pdb_path)

    with bundle_path.open() as handle:
        bundle = json.load(handle)

    atoms = load_first_model_atoms(pdb_path)
    atoms_by_residue = group_atoms_by_residue(atoms)
    nodes, node_warnings = build_residue_nodes(bundle, atoms_by_residue)
    ordered_keys = list(nodes.keys())
    adjacency, edges, graph_warnings = build_residue_graph(atoms_by_residue, ordered_keys, bond_cutoff=bond_cutoff)
    root, root_strategy, root_warnings = choose_root_key(bundle, atoms_by_residue)

    warnings = []
    warnings.extend(node_warnings)
    warnings.extend(graph_warnings)
    warnings.extend(root_warnings)

    return {
        "bundle_path": str(bundle_path),
        "pdb_path": str(pdb_path),
        "bundle": bundle,
        "atoms_by_residue": atoms_by_residue,
        "nodes": nodes,
        "adjacency": adjacency,
        "edges": edges,
        "root": root,
        "root_strategy": root_strategy,
        "warnings": warnings,
    }


def mapping_rows(mapping: dict[ResidueKey, ResidueKey]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for reference_key, source_key in sorted(mapping.items(), key=lambda item: (item[0].chain, item[0].resnum, item[0].icode)):
        rows.append(
            {
                "reference": reference_key.as_dict(),
                "source": source_key.as_dict(),
                "reference_class": normalize_resname(reference_key.resname),
                "source_class": normalize_resname(source_key.resname),
            }
        )

    return rows


def build_pair_mapping_report(
    reference_bundle_path: str | Path,
    reference_pdb_path: str | Path,
    source_bundle_path: str | Path,
    source_pdb_path: str | Path,
    source_name: str,
    bond_cutoff: float = 1.85,
) -> dict[str, Any]:
    reference = load_structure_for_mapping(reference_bundle_path, reference_pdb_path, bond_cutoff=bond_cutoff)
    source = load_structure_for_mapping(source_bundle_path, source_pdb_path, bond_cutoff=bond_cutoff)

    warnings: list[str] = []
    warnings.extend([f"reference: {warning}" for warning in reference["warnings"]])
    warnings.extend([f"{source_name}: {warning}" for warning in source["warnings"]])

    reference_root = reference["root"]
    source_root = source["root"]

    if reference_root is None or source_root is None:
        mapping: dict[ResidueKey, ResidueKey] = {}
        warnings.append("mapping skipped because one root is missing")
    else:
        mapping, map_warnings = map_rooted_trees(
            reference_root=reference_root,
            source_root=source_root,
            reference_adjacency=reference["adjacency"],
            source_adjacency=source["adjacency"],
        )
        warnings.extend([f"mapping: {warning}" for warning in map_warnings])

    n_reference = len(reference["nodes"])
    n_source = len(source["nodes"])
    n_mapped = len(mapping)

    status = "ok"
    if n_mapped == 0:
        status = "failed"
    elif n_mapped < min(n_reference, n_source):
        status = "partial"

    return {
        "source_name": source_name,
        "status": status,
        "method": "rooted_topology_heavy_atom_bonds",
        "bond_cutoff_angstrom": bond_cutoff,
        "reference_bundle_path": str(reference_bundle_path),
        "reference_pdb_path": str(reference_pdb_path),
        "source_bundle_path": str(source_bundle_path),
        "source_pdb_path": str(source_pdb_path),
        "reference_root": reference_root.as_dict() if reference_root else None,
        "reference_root_strategy": reference["root_strategy"],
        "source_root": source_root.as_dict() if source_root else None,
        "source_root_strategy": source["root_strategy"],
        "n_reference_residues": n_reference,
        "n_source_residues": n_source,
        "n_mapped_residues": n_mapped,
        "reference_residues": [node.as_dict() for node in sorted(reference["nodes"].values(), key=lambda item: item.index)],
        "source_residues": [node.as_dict() for node in sorted(source["nodes"].values(), key=lambda item: item.index)],
        "reference_edges": [edge.as_dict() for edge in reference["edges"]],
        "source_edges": [edge.as_dict() for edge in source["edges"]],
        "reference_topology": topology_summary(reference_root, reference["adjacency"]),
        "source_topology": topology_summary(source_root, source["adjacency"]),
        "mapping": mapping_rows(mapping),
        "warnings": warnings,
    }


def build_site_mapping_report(
    pdb_id: str,
    site: str,
    results_root: str | Path = Path("results"),
    include_glycoshape: bool = True,
    include_glycoshield: bool = True,
    bond_cutoff: float = 1.85,
) -> dict[str, Any]:
    results_root = Path(results_root)

    reference_bundle = results_root / pdb_id / "reference" / site / "reference_bundle.json"
    protein_frame = results_root / pdb_id / "aligned" / site / "protein_frame"
    reference_pdb = protein_frame / "reference_glycan.pdb"

    report: dict[str, Any] = {
        "pdb_id": pdb_id,
        "site": site,
        "status": "ok",
        "reference_bundle_path": str(reference_bundle),
        "reference_pdb_path": str(reference_pdb),
        "tools": {},
        "warnings": [],
    }

    if not reference_bundle.exists():
        report["status"] = "failed"
        report["warnings"].append(f"missing reference bundle: {reference_bundle}")
        return report

    if not reference_pdb.exists():
        report["status"] = "failed"
        report["warnings"].append(f"missing protein-frame reference glycan: {reference_pdb}")
        return report

    tool_specs = []

    if include_glycoshield:
        tool_specs.append(
            (
                "glycoshield",
                results_root / pdb_id / "glycoshield" / site / "glycoshield_bundle.json",
                protein_frame / "glycoshield_glycan_ensemble.pdb",
            )
        )

    if include_glycoshape:
        tool_specs.append(
            (
                "glycoshape",
                results_root / pdb_id / "glycoshape" / site / "glycoshape_bundle.json",
                protein_frame / "glycoshape_glycan_ensemble.pdb",
            )
        )

    for tool_name, tool_bundle, tool_pdb in tool_specs:
        if not tool_bundle.exists() or not tool_pdb.exists():
            report["warnings"].append(f"skipped {tool_name}: missing bundle or protein-frame PDB")
            continue

        pair_report = build_pair_mapping_report(
            reference_bundle_path=reference_bundle,
            reference_pdb_path=reference_pdb,
            source_bundle_path=tool_bundle,
            source_pdb_path=tool_pdb,
            source_name=tool_name,
            bond_cutoff=bond_cutoff,
        )
        report["tools"][tool_name] = pair_report

    tool_statuses = [tool_report["status"] for tool_report in report["tools"].values()]

    if not tool_statuses:
        report["status"] = "failed"
    elif any(status == "failed" for status in tool_statuses):
        report["status"] = "partial"
    elif any(status == "partial" for status in tool_statuses):
        report["status"] = "partial"

    return report


def write_site_mapping_report(
    report: dict[str, Any],
    results_root: str | Path = Path("results"),
) -> Path:
    results_root = Path(results_root)
    output_dir = results_root / report["pdb_id"] / "aligned" / report["site"] / "glycan_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "glycan_residue_mapping_report.json"

    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return output_path
