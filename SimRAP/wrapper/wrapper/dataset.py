#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Dataset, graph batching, collate, and device-transfer utilities."""

from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset

from .graph import CrystalGraphBuilder
from .preprocessing import RowRecord


class BulkFusionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[RowRecord],
        graph_builder: CrystalGraphBuilder,
    ):
        self.records = list(records)
        self.graph_builder = graph_builder

    def __len__(self):
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ):
        r = self.records[index]

        try:
            query_graph = (
                self.graph_builder.from_cif(
                    r.query_cif
                )
            )

            neighbor_graphs = []

            neighbor_targets = []
            neighbor_sims = []
            neighbor_ranks = []

            for n in r.neighbors:
                neighbor_graphs.append(
                    self.graph_builder.from_cif(
                        n["cif"]
                    )
                )

                neighbor_targets.append(
                    float(n["target"])
                )

                neighbor_sims.append(
                    float(n["similarity"])
                )

                neighbor_ranks.append(
                    int(n["rank"])
                )

        except Exception as exc:
            raise RuntimeError(
                f"Dataset source row "
                f"{r.source_row_idx}: {exc}"
            ) from exc

        return {
            "source_row_idx":
                r.source_row_idx,

            "query_graph":
                query_graph,

            "neighbor_graphs":
                neighbor_graphs,

            "neighbor_targets":
                neighbor_targets,

            "neighbor_sims":
                neighbor_sims,

            "neighbor_ranks":
                neighbor_ranks,

            "backbone_output":
                float(r.backbone_output),

            "target":
                float(r.target),
        }


def batch_graphs(
    graphs: Sequence[
        Dict[str, torch.Tensor]
    ],
):
    if len(graphs) == 0:
        return None

    node_z_list = []
    edge_src_list = []
    edge_dst_list = []
    edge_dist_list = []
    graph_index_list = []
    graph_feat_list = []

    node_offset = 0

    for graph_idx, g in enumerate(graphs):
        n_nodes = int(
            g["node_z"].shape[0]
        )

        node_z_list.append(
            g["node_z"]
        )

        graph_index_list.append(
            torch.full(
                (n_nodes,),
                graph_idx,
                dtype=torch.long,
            )
        )

        if g["edge_src"].numel() > 0:
            edge_src_list.append(
                g["edge_src"]
                + node_offset
            )

            edge_dst_list.append(
                g["edge_dst"]
                + node_offset
            )

            edge_dist_list.append(
                g["edge_dist"]
            )

        graph_feat_list.append(
            g["graph_feat"]
        )

        node_offset += n_nodes

    node_z = torch.cat(
        node_z_list,
        dim=0,
    )

    graph_index = torch.cat(
        graph_index_list,
        dim=0,
    )

    if edge_src_list:
        edge_src = torch.cat(
            edge_src_list,
            dim=0,
        )

        edge_dst = torch.cat(
            edge_dst_list,
            dim=0,
        )

        edge_dist = torch.cat(
            edge_dist_list,
            dim=0,
        )

    else:
        edge_src = torch.empty(
            (0,),
            dtype=torch.long,
        )

        edge_dst = torch.empty(
            (0,),
            dtype=torch.long,
        )

        edge_dist = torch.empty(
            (0,),
            dtype=torch.float32,
        )

    graph_feat = torch.stack(
        graph_feat_list,
        dim=0,
    )

    return {
        "node_z": node_z,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_dist": edge_dist,
        "graph_index": graph_index,
        "graph_feat": graph_feat,
        "num_graphs": len(graphs),
    }


def collate_bulk_fusion(
    samples: Sequence[Dict[str, Any]],
):
    batch_size = len(samples)

    query_graph = batch_graphs(
        [
            s["query_graph"]
            for s in samples
        ]
    )

    flat_neighbor_graphs = []
    neighbor_owner = []
    neighbor_position = []
    neighbor_target = []
    neighbor_similarity = []
    neighbor_rank = []

    lengths = []

    for sample_idx, s in enumerate(samples):
        n = len(
            s["neighbor_graphs"]
        )

        lengths.append(n)

        for pos in range(n):
            flat_neighbor_graphs.append(
                s["neighbor_graphs"][pos]
            )

            neighbor_owner.append(
                sample_idx
            )

            neighbor_position.append(
                pos
            )

            neighbor_target.append(
                s["neighbor_targets"][pos]
            )

            neighbor_similarity.append(
                s["neighbor_sims"][pos]
            )

            neighbor_rank.append(
                s["neighbor_ranks"][pos]
            )

    neighbor_graph = batch_graphs(
        flat_neighbor_graphs
    )

    return {
        "source_row_idx": [
            int(s["source_row_idx"])
            for s in samples
        ],

        "query_graph":
            query_graph,

        "neighbor_graph":
            neighbor_graph,

        "neighbor_owner":
            torch.tensor(
                neighbor_owner,
                dtype=torch.long,
            ),

        "neighbor_position":
            torch.tensor(
                neighbor_position,
                dtype=torch.long,
            ),

        "neighbor_target":
            torch.tensor(
                neighbor_target,
                dtype=torch.float32,
            ),

        "neighbor_similarity":
            torch.tensor(
                neighbor_similarity,
                dtype=torch.float32,
            ),

        "neighbor_rank":
            torch.tensor(
                neighbor_rank,
                dtype=torch.float32,
            ),

        "neighbor_lengths":
            torch.tensor(
                lengths,
                dtype=torch.long,
            ),

        "backbone_output":
            torch.tensor(
                [
                    s["backbone_output"]
                    for s in samples
                ],
                dtype=torch.float32,
            ),

        "target":
            torch.tensor(
                [
                    s["target"]
                    for s in samples
                ],
                dtype=torch.float32,
            ),

        "batch_size":
            batch_size,
    }


def move_graph_batch(
    graph_batch,
    device: torch.device,
):
    if graph_batch is None:
        return None

    return {
        key: (
            value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value
        in graph_batch.items()
    }


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
):
    out = dict(batch)

    out["query_graph"] = (
        move_graph_batch(
            batch["query_graph"],
            device,
        )
    )

    out["neighbor_graph"] = (
        move_graph_batch(
            batch["neighbor_graph"],
            device,
        )
    )

    for key in [
        "neighbor_owner",
        "neighbor_position",
        "neighbor_target",
        "neighbor_similarity",
        "neighbor_rank",
        "neighbor_lengths",
        "backbone_output",
        "target",
    ]:
        out[key] = (
            batch[key].to(device)
        )

    return out
