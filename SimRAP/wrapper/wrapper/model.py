#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""CrystalGraph + BiGRU + source-aware wrapper architecture."""

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class ModelConfig:
    atom_embed_dim: int = 128
    graph_hidden_dim: int = 128
    graph_layers: int = 3

    rbf_dim: int = 32
    cutoff: float = 5.0

    source_embed_dim: int = 16

    neighbor_token_dim: int = 192

    gru_hidden_dim: int = 128
    gru_layers: int = 1
    bidirectional: bool = True

    backbone_output_token_dim: int = 64

    fusion_hidden_dim: int = 256

    dropout: float = 0.15

    rank_scale: float = 1.0


class GaussianRBF(nn.Module):
    def __init__(
        self,
        cutoff: float,
        num_rbf: int,
    ):
        super().__init__()

        centers = torch.linspace(
            0.0,
            float(cutoff),
            int(num_rbf),
        )

        if num_rbf > 1:
            delta = float(
                centers[1]
                - centers[0]
            )
        else:
            delta = float(cutoff)

        gamma = 1.0 / max(
            delta * delta,
            1e-8,
        )

        self.register_buffer(
            "centers",
            centers,
        )

        self.gamma = float(gamma)

    def forward(
        self,
        distances: torch.Tensor,
    ):
        diff = (
            distances.unsqueeze(-1)
            - self.centers
        )

        return torch.exp(
            -self.gamma
            * diff.pow(2)
        )


class CrystalMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rbf_dim: int,
        dropout: float,
    ):
        super().__init__()

        input_dim = (
            hidden_dim * 2
            + rbf_dim
        )

        self.message_mlp = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim * 2,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rbf: torch.Tensor,
    ):
        if edge_src.numel() == 0:
            return self.norm(h)

        msg_input = torch.cat(
            [
                h[edge_src],
                h[edge_dst],
                edge_rbf,
            ],
            dim=-1,
        )

        msg = self.message_mlp(
            msg_input
        )

        agg = torch.zeros_like(h)

        agg.index_add_(
            0,
            edge_dst,
            msg,
        )

        deg = torch.zeros(
            h.shape[0],
            device=h.device,
            dtype=h.dtype,
        )

        deg.index_add_(
            0,
            edge_dst,
            torch.ones(
                edge_dst.shape[0],
                device=h.device,
                dtype=h.dtype,
            ),
        )

        agg = (
            agg
            / deg.clamp_min(
                1.0
            ).unsqueeze(-1)
        )

        update = self.update_mlp(
            agg
        )

        return self.norm(
            h + update
        )


class CrystalGraphEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        # 0 reserved for padding/unknown.
        self.atom_embedding = nn.Embedding(
            119,
            config.atom_embed_dim,
            padding_idx=0,
        )

        self.atom_projection = nn.Linear(
            config.atom_embed_dim,
            config.graph_hidden_dim,
        )

        self.rbf = GaussianRBF(
            cutoff=config.cutoff,
            num_rbf=config.rbf_dim,
        )

        self.layers = nn.ModuleList(
            [
                CrystalMessageLayer(
                    hidden_dim=
                        config.graph_hidden_dim,
                    rbf_dim=
                        config.rbf_dim,
                    dropout=
                        config.dropout,
                )
                for _ in range(
                    config.graph_layers
                )
            ]
        )

        self.graph_projection = nn.Sequential(
            nn.Linear(
                config.graph_hidden_dim + 3,
                config.graph_hidden_dim,
            ),
            nn.SiLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.graph_hidden_dim,
                config.graph_hidden_dim,
            ),
            nn.LayerNorm(
                config.graph_hidden_dim
            ),
        )

    def forward(
        self,
        graph_batch: Dict[str, Any],
    ):
        z = graph_batch[
            "node_z"
        ].clamp(
            min=0,
            max=118,
        )

        h = self.atom_projection(
            self.atom_embedding(z)
        )

        edge_src = graph_batch[
            "edge_src"
        ]

        edge_dst = graph_batch[
            "edge_dst"
        ]

        edge_dist = graph_batch[
            "edge_dist"
        ]

        edge_rbf = self.rbf(
            edge_dist
        )

        for layer in self.layers:
            h = layer(
                h,
                edge_src,
                edge_dst,
                edge_rbf,
            )

        graph_index = graph_batch[
            "graph_index"
        ]

        num_graphs = int(
            graph_batch[
                "num_graphs"
            ]
        )

        pooled = torch.zeros(
            (
                num_graphs,
                h.shape[-1],
            ),
            device=h.device,
            dtype=h.dtype,
        )

        pooled.index_add_(
            0,
            graph_index,
            h,
        )

        counts = torch.zeros(
            num_graphs,
            device=h.device,
            dtype=h.dtype,
        )

        counts.index_add_(
            0,
            graph_index,
            torch.ones(
                graph_index.shape[0],
                device=h.device,
                dtype=h.dtype,
            ),
        )

        pooled = (
            pooled
            / counts.clamp_min(
                1.0
            ).unsqueeze(-1)
        )

        graph_feat = graph_batch[
            "graph_feat"
        ].to(
            dtype=h.dtype
        )

        return self.graph_projection(
            torch.cat(
                [
                    pooled,
                    graph_feat,
                ],
                dim=-1,
            )
        )


class BulkModulusFusionModel(nn.Module):
    """
    Source-aware crystal-graph + BiGRU regression model.

    Source IDs:
        0 = similarity retrieval neighbor
        1 = backbone model output
    """

    RETRIEVAL_SOURCE_ID = 0
    BACKBONE_OUTPUT_SOURCE_ID = 1

    def __init__(
        self,
        config: ModelConfig,
        target_mean: float,
        target_std: float,
    ):
        super().__init__()

        self.config = config

        self.register_buffer(
            "target_mean",
            torch.tensor(
                float(target_mean),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "target_std",
            torch.tensor(
                max(
                    float(target_std),
                    1e-6,
                ),
                dtype=torch.float32,
            ),
        )

        self.graph_encoder = (
            CrystalGraphEncoder(
                config
            )
        )

        self.source_embedding = (
            nn.Embedding(
                2,
                config.source_embed_dim,
            )
        )

        # neighbor token:
        # n_graph
        # |q_graph - n_graph|
        # neighbor_target_scaled
        # similarity
        # normalized rank
        # source embedding
        neighbor_raw_dim = (
            config.graph_hidden_dim * 2
            + 3
            + config.source_embed_dim
        )

        self.neighbor_token_encoder = (
            nn.Sequential(
                nn.Linear(
                    neighbor_raw_dim,
                    config.neighbor_token_dim,
                ),
                nn.SiLU(),
                nn.Dropout(
                    config.dropout
                ),
                nn.Linear(
                    config.neighbor_token_dim,
                    config.neighbor_token_dim,
                ),
                nn.LayerNorm(
                    config.neighbor_token_dim
                ),
            )
        )

        self.neighbor_gru = nn.GRU(
            input_size=
                config.neighbor_token_dim,
            hidden_size=
                config.gru_hidden_dim,
            num_layers=
                config.gru_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.gru_layers > 1
                else 0.0
            ),
            bidirectional=
                config.bidirectional,
        )

        self.gru_output_dim = (
            config.gru_hidden_dim
            * (
                2
                if config.bidirectional
                else 1
            )
        )

        backbone_output_raw_dim = (
            1
            + config.source_embed_dim
        )

        self.backbone_output_encoder = nn.Sequential(
            nn.Linear(
                backbone_output_raw_dim,
                config.backbone_output_token_dim,
            ),
            nn.SiLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.backbone_output_token_dim,
                config.backbone_output_token_dim,
            ),
            nn.LayerNorm(
                config.backbone_output_token_dim
            ),
        )

        fusion_dim = (
            config.graph_hidden_dim
            + self.gru_output_dim
            + config.backbone_output_token_dim
            + 2
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(
                fusion_dim,
                config.fusion_hidden_dim,
            ),
            nn.SiLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.fusion_hidden_dim,
                config.fusion_hidden_dim // 2,
            ),
            nn.SiLU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.fusion_hidden_dim // 2,
                1,
            ),
        )

    def scale_value(
        self,
        x: torch.Tensor,
    ):
        return (
            x
            - self.target_mean
        ) / self.target_std

    def unscale_value(
        self,
        x: torch.Tensor,
    ):
        return (
            x
            * self.target_std
            + self.target_mean
        )

    def _neighbor_summary(
        self,
        batch: Dict[str, Any],
        query_embedding: torch.Tensor,
    ):
        batch_size = int(
            batch["batch_size"]
        )

        lengths = batch[
            "neighbor_lengths"
        ]

        summary = torch.zeros(
            (
                batch_size,
                self.gru_output_dim,
            ),
            device=query_embedding.device,
            dtype=query_embedding.dtype,
        )

        neighbor_graph = batch[
            "neighbor_graph"
        ]

        if (
            neighbor_graph is None
            or batch[
                "neighbor_owner"
            ].numel() == 0
        ):
            return summary

        n_embedding = (
            self.graph_encoder(
                neighbor_graph
            )
        )

        owner = batch[
            "neighbor_owner"
        ]

        position = batch[
            "neighbor_position"
        ]

        q_for_neighbor = (
            query_embedding[
                owner
            ]
        )

        graph_delta = torch.abs(
            q_for_neighbor
            - n_embedding
        )

        neighbor_target_scaled = (
            self.scale_value(
                batch[
                    "neighbor_target"
                ]
            )
        )

        similarity = batch[
            "neighbor_similarity"
        ]

        rank_scale = max(
            float(
                self.config.rank_scale
            ),
            1.0,
        )

        rank_feature = (
            batch[
                "neighbor_rank"
            ]
            / rank_scale
        )

        source_ids = torch.full(
            (
                owner.shape[0],
            ),
            self.RETRIEVAL_SOURCE_ID,
            device=owner.device,
            dtype=torch.long,
        )

        source_emb = (
            self.source_embedding(
                source_ids
            )
        )

        raw_token = torch.cat(
            [
                n_embedding,
                graph_delta,
                neighbor_target_scaled
                    .unsqueeze(-1),
                similarity
                    .unsqueeze(-1),
                rank_feature
                    .unsqueeze(-1),
                source_emb,
            ],
            dim=-1,
        )

        token = (
            self.neighbor_token_encoder(
                raw_token
            )
        )

        max_len = int(
            lengths.max().item()
        )

        padded = torch.zeros(
            (
                batch_size,
                max_len,
                token.shape[-1],
            ),
            device=token.device,
            dtype=token.dtype,
        )

        padded[
            owner,
            position,
        ] = token

        positive_mask = (
            lengths > 0
        )

        if not positive_mask.any():
            return summary

        positive_indices = (
            torch.nonzero(
                positive_mask,
                as_tuple=False,
            )
            .squeeze(-1)
        )

        seq = padded[
            positive_indices
        ]

        seq_lengths = lengths[
            positive_indices
        ]

        packed = pack_padded_sequence(
            seq,
            lengths=
                seq_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, h_n = self.neighbor_gru(
            packed
        )

        num_directions = (
            2
            if self.config.bidirectional
            else 1
        )

        # [layers*directions, B+, H]
        # -> [layers, directions, B+, H]
        h_n = h_n.view(
            self.config.gru_layers,
            num_directions,
            seq.shape[0],
            self.config.gru_hidden_dim,
        )

        last_layer = h_n[-1]

        if self.config.bidirectional:
            positive_summary = torch.cat(
                [
                    last_layer[0],
                    last_layer[1],
                ],
                dim=-1,
            )
        else:
            positive_summary = (
                last_layer[0]
            )

        summary[
            positive_indices
        ] = positive_summary

        return summary

    def forward(
        self,
        batch: Dict[str, Any],
    ):
        query_embedding = (
            self.graph_encoder(
                batch["query_graph"]
            )
        )

        neighbor_summary = (
            self._neighbor_summary(
                batch,
                query_embedding,
            )
        )

        backbone_output_scaled = (
            self.scale_value(
                batch["backbone_output"]
            )
        )

        backbone_output_source_ids = (
            torch.full(
                (
                    batch["backbone_output"]
                    .shape[0],
                ),
                self.BACKBONE_OUTPUT_SOURCE_ID,
                device=
                    batch["backbone_output"]
                    .device,
                dtype=torch.long,
            )
        )

        backbone_output_source_emb = (
            self.source_embedding(
                backbone_output_source_ids
            )
        )

        backbone_output_token = (
            self.backbone_output_encoder(
                torch.cat(
                    [
                        backbone_output_scaled
                            .unsqueeze(-1),
                        backbone_output_source_emb,
                    ],
                    dim=-1,
                )
            )
        )

        lengths = (
            batch[
                "neighbor_lengths"
            ]
            .to(
                dtype=query_embedding.dtype
            )
        )

        neighbor_count_feature = (
            torch.log1p(
                lengths
            )
            / math.log1p(
                max(
                    float(
                        self.config.rank_scale
                    ),
                    1.0,
                )
                + 1.0
            )
        )

        fusion = torch.cat(
            [
                query_embedding,
                neighbor_summary,
                backbone_output_token,
                backbone_output_scaled
                    .unsqueeze(-1),
                neighbor_count_feature
                    .unsqueeze(-1),
            ],
            dim=-1,
        )

        correction = (
            self.fusion_head(
                fusion
            ).squeeze(-1)
        )

        # The backbone output is an explicit baseline.
        # Model learns source-aware correction.
        pred_scaled = (
            backbone_output_scaled
            + correction
        )

        return pred_scaled
