#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Hierarchical similarity retrieval for crystalline materials.

Step 1: build a persistent multi-way search tree from train.csv
    Space group -> Wyckoff occupancy trie -> family@Wyckoff trie -> element@Wyckoff trie

Step 2: retrieve up to k nearest materials from the train index for rows from one or more CSV files.

Similarity hierarchy
--------------------
L1:
    1.0 if another train material with exactly the same international space-group number exists.
    Otherwise 0.0 and L2/L3/L4 are NULL.

L2:
    If an exact Wyckoff-occupancy signature exists among same-SPG candidates, L2 = 1.0.
    Otherwise:
        L2 = number of atoms whose (element, Wyckoff-letter) matches
             / max(N_query, N_reference)
    and L3/L4 are NULL.

L3 (only when L2 == 1 because Wyckoff occupancy is exact):
    L3 = exact matched (element, Wyckoff-letter) atom count / N_query

L4:
    Starts from L3. For unmatched atoms on the same Wyckoff letter, the script performs
    maximum-weight bipartite matching so that ordering of substitutions cannot change the score.

    Normal same-group substitution (H excluded; lanthanoids/actinoids excluded from this branch):
        bonus_weight = max(0, 1 - abs(period_A - period_B) / 4)

    Lanthanoid <-> lanthanoid OR actinoid <-> actinoid:
        bonus_weight = max(0, 1 - abs(Z_A - Z_B) / 14)

    L4 = L3 + sum(bonus_weight for matched unmatched atoms) / N_query

Important implementation choice
-------------------------------
By default, structures are symmetry-refined to a conventional cell before extracting Wyckoff
letters (SpacegroupAnalyzer.get_refined_structure()). This makes signatures more comparable when
input CIF cell settings differ. Use --standardize none to operate directly on the input CIF cell.

The script assumes ordered structures. Disordered structures are rejected explicitly because the
requested atom-count similarity equations are defined for discrete element occupations.
"""

from material_similarity.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
