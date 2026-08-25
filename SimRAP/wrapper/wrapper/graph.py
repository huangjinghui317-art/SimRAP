#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""CIF-to-periodic-graph construction."""

import hashlib
import math
from collections import OrderedDict
from typing import Dict, List

import torch
from pymatgen.core import Structure


class CrystalGraphBuilder:
    """
    CIF -> periodic graph with an in-memory LRU cache.

    Graph:
        node_z     [N]
        edge_src   [E]
        edge_dst   [E]
        edge_dist  [E]
        graph_feat [3]

    graph_feat:
        log(1 + volume_per_atom)
        log(1 + density)
        log(1 + num_atoms)
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        cache_size: int = 20000,
    ):
        self.cutoff = float(cutoff)
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[
            str,
            Dict[str, torch.Tensor],
        ] = OrderedDict()

    def _cache_key(
        self,
        cif: str,
    ) -> str:
        return hashlib.sha1(
            cif.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

    def _cache_get(
        self,
        key: str,
    ):
        if self.cache_size <= 0:
            return None

        graph = self.cache.get(key)

        if graph is not None:
            self.cache.move_to_end(key)

        return graph

    def _cache_put(
        self,
        key: str,
        graph: Dict[str, torch.Tensor],
    ):
        if self.cache_size <= 0:
            return

        self.cache[key] = graph
        self.cache.move_to_end(key)

        while len(self.cache) > self.cache_size:
            self.cache.popitem(
                last=False
            )

    @staticmethod
    def _atomic_numbers(
        structure: Structure,
    ) -> List[int]:
        z = []

        for site in structure:
            if site.is_ordered:
                z.append(
                    int(site.specie.Z)
                )
            else:
                # Disordered site: choose the species
                # with the largest occupancy.
                specie = max(
                    site.species.items(),
                    key=lambda kv: kv[1],
                )[0]

                z.append(
                    int(specie.Z)
                )

        return z

    def from_cif(
        self,
        cif: str,
    ) -> Dict[str, torch.Tensor]:

        key = self._cache_key(cif)

        cached = self._cache_get(key)

        if cached is not None:
            return cached

        try:
            structure = Structure.from_str(
                cif,
                fmt="cif",
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse CIF: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if len(structure) == 0:
            raise ValueError(
                "Structure contains zero atoms."
            )

        node_z = torch.tensor(
            self._atomic_numbers(structure),
            dtype=torch.long,
        )

        try:
            (
                center_idx,
                neighbor_idx,
                _offsets,
                distances,
            ) = structure.get_neighbor_list(
                r=self.cutoff,
                exclude_self=True,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to build neighbor list: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Message direction:
        # periodic neighbor -> center atom
        edge_src = torch.as_tensor(
            neighbor_idx,
            dtype=torch.long,
        )

        edge_dst = torch.as_tensor(
            center_idx,
            dtype=torch.long,
        )

        edge_dist = torch.as_tensor(
            distances,
            dtype=torch.float32,
        )

        n_atoms = len(structure)

        volume_per_atom = (
            float(structure.volume)
            / max(n_atoms, 1)
        )

        try:
            density = float(
                structure.density
            )
        except Exception:
            density = 0.0

        graph_feat = torch.tensor(
            [
                math.log1p(
                    max(volume_per_atom, 0.0)
                ),
                math.log1p(
                    max(density, 0.0)
                ),
                math.log1p(
                    float(n_atoms)
                ),
            ],
            dtype=torch.float32,
        )

        graph = {
            "node_z": node_z,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_dist": edge_dist,
            "graph_feat": graph_feat,
        }

        self._cache_put(
            key,
            graph,
        )

        return graph
