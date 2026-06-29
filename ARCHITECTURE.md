# Benchmark GlycoShield / GlycoShape — architecture du pipeline

Spec de conception figée. Benchmark comparatif de **GlycoShield** et
**GlycoShape / Re-Glyco** contre des structures expérimentales glycosylées
(jeu DIONYSUS), sur trois axes : forme générale du glycane, conformation
(torsions O-glycosidiques + ring puckering), shielding de la surface
protéique. Le calcul scientifique repose au maximum sur **GlyContact** ; le
code maison se limite à l'ingestion, au mapping, à la comparaison inter-outils
et aux figures.

---

## 1. Objectif

Pour chaque structure de référence, comparer ce que produit chaque outil de
re-glycosylation à la structure expérimentale, puis comparer les deux outils
entre eux. Sortie attendue : tableaux et figures de comparaison, stratifiés par
type de glycane.

---

## 2. Jeu de données et partition

23 PDB de référence. Comparabilité (issue des deux fichiers `.ods` de mapping) :

| Catégorie | n | PDB |
|---|---|---|
| Triple (Shield + Shape + réf.) | 17 | 1OYH, 2J1G, 2PE4, 2QL1, 2QWB, 3AVE, 3DGC, 4B7I, 4C54, 5IW3, 5L2J, 5XWD, 6CMG, 6F9T, 6FB3, 6X3X, 6Z2P |
| GlycoShield-seul + réf. | 4 | 1BHG, 3WSR, 3WV0, 6WXU |
| Exclus (échec extraction du résidu glycosylé) | 2 | 3AYA, 5A58 |

### Stratification des résultats par type de glycane

Les résultats sont **toujours présentés par type**, jamais regroupés (petit
effectif par classe). L'astérisque marque les cas GlycoShield-seul.

- **High-mannose** (6) : 1BHG\*, 2QWB, 4B7I, 6FB3, 6WXU\*, 6X3X
- **Paucimannose** (6) : 1OYH, 2J1G, 2PE4, 3DGC, 5L2J, 6CMG
- **Complexe** (6) : 2QL1, 3AVE, 4C54, 5IW3, 5XWD, 6F9T
- **O-glycane** exploitable (3) : 3WSR\*, 3WV0\*, 6Z2P (+ 3AYA, 5A58 exclus)

### Formes de sortie GlycoShield à ingérer

| Forme | n | PDB | Contenu |
|---|---|---|---|
| Complète | 16 | 1OYH, 2PE4, 2QL1, 2QWB, 3AVE, 3DGC, 3WSR, 3WV0, 4B7I, 5IW3, 5L2J, 5XWD, 6CMG, 6WXU, 6X3X, 6Z2P | XTC (~milliers de frames) + SASA par résidu + glycane isolé |
| Merged-only | 5 | 1BHG, 2J1G, 4C54, 6F9T, 6FB3 | un seul PDB à 30 modèles, **sans** SASA ni XTC |
| Échec | 2 | 3AYA, 5A58 | aucune sortie |

`merged_traj_pdb.pdb` (30 modèles) est présent dans **tous** les cas ayant une
sortie, y compris les complètes : c'est l'artefact GlycoShield canonique
(voir §4, décision sur l'ensemble GlycoShield).

---

## 3. Diversité et encodage des entrées

Les trois sources encodent le glycane de façon incompatible. La normalisation
est le cœur du problème.

| Source | Frames | Encodage glycane | Poids | SASA fournie | Site |
|---|---|---|---|---|---|
| Référence (exp.) | PDB unique (parfois altlocs) | HETATM, chaîne séparée, **noms PDB** (NAG/MAN/FUC/BMA) | 1 conformère | non | enregistrements `LINK` (ASN/THR/SER → C1) |
| GlycoShape / Re-Glyco | `all.pdb` (30 MODELs) + `first.xtc` | ATOM, chaîne renumérotée, **noms PDB** | ~30, pondérés par populations de cluster ; `ensemble_stats.csv` | b-factor (`sasa.pdb`), `hotspots.pdb` | `job_meta.json` → `selectedGlycans` |
| GlycoShield | topologie PDB + **XTC** (complète) ou `merged_traj_pdb.pdb` (30 MODELs) | ATOM, chaîne X, **noms CHARMM** (BGL/BMA/AMA/AFU) | uniforme | `maxResidueSASA_probe_*.txt` (complète uniquement) | nom de fichier / site d'entrée |

Conséquences directes pour le pipeline :

- **Nomenclature** : CHARMM (GlycoShield) vs PDB (GlycoShape, référence) →
  normalisées vers IUPAC via GlyContact.
- **Renumérotation / chaîne** : le mapping résidu↔atome doit s'ancrer sur
  l'extrémité réductrice + la topologie de l'arbre, **jamais** sur le resid.
- **Sites multiples** : plusieurs références portent plusieurs sites
  (ex. 6F9T : ASN72 + ASN109 ; 1OYH : 4 sites × 2 chaînes). On ne benchmarke
  que le site annoté dans l'ODS, sélectionné à l'ingestion.

---

## 4. Décisions méthodologiques (verrouillées)

1. **Alignement double.**
   - Superposition sur la **protéine** entière → mesure où le glycane est
     placé/orienté par rapport à la surface (pertinent pour le shielding).
   - Calage **local** du glycane (extrémité réductrice) via
     `superimpose_glycans` → mesure la forme intrinsèque de l'arbre,
     indépendamment de sa position.

2. **Ensembles.**
   - Comparaisons sur **distributions normalisées** (aire 1, insensibles à la
     taille de l'ensemble), avec **poids de cluster GlycoShape honorés**.
   - Face à l'unique conformère expérimental, on rapporte **deux lectures** :
     *meilleur accord* (RMSD minimal sur l'ensemble → l'outil peut-il
     atteindre la conformation expérimentale ?) et *typicité* (le conformère
     expérimental tombe-t-il dans la zone peuplée par l'outil → est-il
     typique ou rare ?).

3. **SASA recalculée en interne.** Un seul moteur (GlyContact / Shrake-Rupley),
   sonde **1,4 Å**, appliqué identiquement aux trois sources. Règle d'un coup le
   problème des 5 cas merged-only sans SASA. La SASA fournie par les outils ne
   sert que de vérification croisée.

4. **O-glycanes.** Même jeu de descripteurs (torsions, puckering, ΔSASA) —
   c'est l'intérêt d'inclure 3WSR / 3WV0 pour tester la O-reglycosylation de
   GlycoShield — résultats stratifiés par type (cf. §2).

5. **Ensemble GlycoShield canonique = `merged_traj_pdb.pdb` (30 modèles)**,
   pour les 21 cas ayant une sortie. Garantit un traitement uniforme et la
   parité avec les 30 conformères de GlycoShape. Le XTC complet (~milliers de
   frames), quand il existe, sert uniquement à une métrique de flexibilité
   GlycoShield supplémentaire et optionnelle.

---

## 5. Architecture en 6 couches

Flux : 3 entrées hétérogènes → représentation interne unifiée → descripteurs
GlyContact (3 axes) → comparaison → figures.

### Couche 0 — Manifeste et config
`config/dataset.yaml`, source de vérité unique. Par PDB : type de glycane,
site(s) annoté(s) (resnum auth + chaîne), GlyTouCan, chemins des trois entrées,
statut (`triple` / `shield_only` / `exclu`), forme de sortie GlycoShield.
Tout le pipeline lit ce manifeste ; rien n'est codé en dur.

### Couche 1 — Ingestion et QC
Un lecteur par format produisant un bundle standardisé : protéine apo,
ensemble de frames du glycane, vecteur de poids, annotation du site, provenance,
nomenclature normalisée. La QC vit ici : complétude atomique, existence du
résidu site, cohérence de composition, clashs. C'est le filet qui attrape les
échecs (les exclusions 3AYA / 5A58 sont des échecs de cette étape).

### Couche 2 — Correspondance et alignement
Mapping résidu/atome entre les trois représentations, ancré sur l'extrémité
réductrice + la topologie de l'arbre (graphe GlyContact). Superposition de
chaque protéine-outil sur la référence pour un repère commun.

### Couche 3 — Descripteurs (GlyContact)
Par (pdb, outil, frame) :
- **Forme** : Rg, span, matrice de distances / contact map, RMSD au conformère
  expérimental.
- **Conformation** : torsions φ/ψ/ω par liaison ; Cremer-Pople Q/θ + classe de
  pucker par cycle.
- **Shielding** : ΔSASA par résidu protéique (apo vs glycosylé), fréquence de
  masquage sur l'ensemble.

Sortie en tables longues (« tidy »), schéma unique.

### Couche 4 — Comparaison et statistiques
Distances entre distributions (circulaires pour les torsions), accord des
classes de pucker, RMSD à l'expérimental, corrélation des profils de shielding.
Agrégation par type de glycane ; tests adaptés aux données circulaires /
appariées.

### Couche 5 — Figures et tableaux
Overlays type Ramachandran (exp vs 2 outils), distributions de pucker, barres
de RMSD, heatmaps de shielding sur la protéine, tables de synthèse, sessions
PyMOL optionnelles.

---

## 6. Frontière GlyContact / code maison

**GlyContact fournit :** graphe + connectivité + nomenclature universelle ;
torsions φ/ψ/ω ; Cremer-Pople ; SASA Shrake-Rupley ; `superimpose_glycans` /
Kabsch RMSD ; contact maps ; flexibilité.

**Code maison :** ingestion des 3 formats + QC ; mapping et alignement
protéine ; gestion des poids et ensembles ; comparaison inter-outils + stats
circulaires ; figures ; manifeste.

---

## 7. Organisation du dépôt (uv)

```text
projet/
├── pyproject.toml                # uv ; dépendances (glycontact, mdanalysis, …)
├── ARCHITECTURE.md               # ce document
├── config/
│   └── dataset.yaml              # couche 0 — manifeste
├── src/glycobench/
│   ├── ingest/                   # 1 · lecteurs par format + QC
│   ├── mapping/                  # 2 · correspondance + alignement
│   ├── descriptors/              # 3 · adaptateurs GlyContact (3 axes)
│   ├── compare/                  # 4 · stats inter-outils
│   └── figures/                  # 5 · figures + tables
├── stages/                       # 1 script court d'orchestration par couche
├── results/<pdb>/<tool>/         # bundles standardisés (intermédiaires)
└── deliverables/                 # tables + figures finales
```

---

## 8. Points à traiter au moment du code

1. **Adaptateur GlyContact** (principal risque d'intégration). GlyContact est
   pensé pour des PDB de glycane isolé / la base GlycoShape. Il faut lui passer
   des frames *normalisées et extraites* (glycane seul, noms → PDB/IUPAC), pas
   la glycoprotéine brute en noms CHARMM. À valider sur le premier cas réel.
2. **Ensemble GlycoShield** : décidé — 30 modèles de `merged_traj_pdb.pdb`
   partout (cf. §4.5).
3. **Sélection du site** quand la référence en porte plusieurs : seul le site
   annoté dans l'ODS, résolu à l'ingestion via le mapping site → glycane.

---

## 9. Prochaine étape

Construire la **couche 0** (`config/dataset.yaml`), dont tout le reste dépend.
Elle encode la partition et les statuts du §2 ainsi que les chemins des
entrées.
