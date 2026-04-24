#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_pos_labels: int,
        num_ner_labels: int,
        padding_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.pos_head = nn.Linear(hidden_dim * 2, num_pos_labels)
        self.ner_head = nn.Linear(hidden_dim * 2, num_ner_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        x = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        pos_logits = self.pos_head(out)
        ner_logits = self.ner_head(out)
        return out, pos_logits, ner_logits


class RelationalGraphConv(nn.Module):
    """
    Simple relational message passing:
    message(i <- j) = Linear([h_j ; e_ij])
    aggregated over incoming neighbors j where adj[i,j] or adj[src,dst] convention as passed.
    Here we use torch.bmm-style dense processing on pair tensor.
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.msg_linear = nn.Linear(node_dim + edge_dim, out_dim)
        self.self_linear = nn.Linear(node_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_repr: torch.Tensor,      # [B, N, D]
        adj: torch.Tensor,            # [B, N, N]
        edge_repr: torch.Tensor,      # [B, N, N, E]
    ) -> torch.Tensor:
        B, N, D = node_repr.shape

        # sender representations: j
        h_j = node_repr.unsqueeze(1).expand(B, N, N, D)   # target i, source j
        msg_input = torch.cat([h_j, edge_repr], dim=-1)   # [B, N, N, D+E]
        msgs = self.msg_linear(msg_input)                 # [B, N, N, O]

        # mask by adjacency
        msgs = msgs * adj.unsqueeze(-1)

        # aggregate source j -> target i
        agg = msgs.sum(dim=2)  # [B, N, O]

        out = self.self_linear(node_repr) + agg
        out = F.relu(out)
        out = self.dropout(out)
        return out


class RelationalGCNEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            layers.append(RelationalGraphConv(in_dim, edge_dim, hidden_dim, dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        node_repr: torch.Tensor,
        adj: torch.Tensor,
        edge_repr: torch.Tensor,
    ) -> torch.Tensor:
        h = node_repr
        for layer in self.layers:
            h = layer(h, adj, edge_repr)
        return h


class EventExtractor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        affix_vocab_size: int,
        pos_label_size: int,
        ner_label_size: int,
        trigger_label_size: int,
        role_label_size: int,
        word_emb_dim: int = 128,
        graph_emb_dim: int = 64,
        affix_emb_dim: int = 32,
        pos_emb_dim: int = 32,
        ner_emb_dim: int = 32,
        case_type_size: int = 7,
        case_type_emb_dim: int = 16,
        morph_feat_dim: int = 8,
        lstm_hidden_dim: int = 128,
        gcn_hidden_dim: int = 128,
        gcn_layers: int = 2,
        edge_type_size: int = 18,
        edge_type_emb_dim: int = 16,
        padding_idx: int = 0,
        dropout: float = 0.3,
        max_pair_distance: int = 3,
        distance_emb_dim: int = 16,
    ):
        super().__init__()

        self.max_pair_distance = max_pair_distance
        self.distance_emb_dim = distance_emb_dim
        self.gcn_hidden_dim = gcn_hidden_dim

        # 1) sequence tagger
        self.tagger = BiLSTMTagger(
            vocab_size=vocab_size,
            emb_dim=word_emb_dim,
            hidden_dim=lstm_hidden_dim,
            num_pos_labels=pos_label_size,
            num_ner_labels=ner_label_size,
            padding_idx=padding_idx,
            dropout=dropout,
        )

        # optional graph-initial token embedding
        self.graph_embedding = nn.Embedding(vocab_size, graph_emb_dim, padding_idx=padding_idx)

        self.affix_embedding = nn.Embedding(
            affix_vocab_size, affix_emb_dim, padding_idx=padding_idx
        )
        self.pos_embedding = nn.Embedding(
            pos_label_size, pos_emb_dim, padding_idx=padding_idx
        )
        self.ner_embedding = nn.Embedding(
            ner_label_size, ner_emb_dim, padding_idx=padding_idx
        )
        self.case_type_embedding = nn.Embedding(
            case_type_size, case_type_emb_dim, padding_idx=0
        )

        # NEW: edge-type embeddings
        self.edge_type_embedding = nn.Embedding(
            edge_type_size, edge_type_emb_dim, padding_idx=0
        )

        # distance embedding for pair bias/features
        self.distance_embedding = nn.Embedding(
            (2 * max_pair_distance) + 3,
            distance_emb_dim,
        )

        fusion_dim = (
            (lstm_hidden_dim * 2)
            + graph_emb_dim
            + affix_emb_dim
            + pos_emb_dim
            + ner_emb_dim
            + case_type_emb_dim
            + morph_feat_dim
        )

        # NEW: relational GCN instead of plain GCN
        self.rgcn = RelationalGCNEncoder(
            node_dim=fusion_dim,
            edge_dim=edge_type_emb_dim,
            hidden_dim=gcn_hidden_dim,
            num_layers=gcn_layers,
            dropout=dropout,
        )

        # trigger classifier
        self.trigger_head = nn.Linear(gcn_hidden_dim, trigger_label_size)

        # NEW: trigger-conditioned attention projections
        self.trigger_query = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
        self.argument_key = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
        self.argument_value = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)

        # confidence features
        self.trigger_conf_proj = nn.Linear(trigger_label_size, 1)
        self.ner_conf_proj = nn.Linear(ner_label_size, 1)

        # biaffine-style compatibility
        self.bilinear = nn.Bilinear(gcn_hidden_dim, gcn_hidden_dim, 1)

        pair_extra_dim = (
            distance_emb_dim   # distance emb
            + 1                # adjacency feature
            + 1                # self feature
            + 1                # attention score
            + 1                # bilinear score
            + 1                # trigger confidence
            + 1                # ner confidence
            + 1                # case confidence
        )

        pair_dim = (gcn_hidden_dim * 4) + pair_extra_dim

        self.arg_exist_mlp = nn.Sequential(
            nn.Linear(pair_dim, gcn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden_dim, 2),
        )

        self.role_mlp = nn.Sequential(
            nn.Linear(pair_dim, gcn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden_dim, role_label_size),
        )

    def initialize_graph_embeddings(self, embedding_matrix: torch.Tensor) -> None:
        if embedding_matrix.shape != self.graph_embedding.weight.shape:
            raise ValueError(
                f"Embedding shape mismatch: got {embedding_matrix.shape}, "
                f"expected {self.graph_embedding.weight.shape}"
            )
        with torch.no_grad():
            self.graph_embedding.weight.copy_(embedding_matrix)

    def _build_distance_embedding(
        self,
        B: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        idx = torch.arange(N, device=device)
        signed_dist = idx.unsqueeze(0) - idx.unsqueeze(1)
        signed_dist = signed_dist.clamp(
            -self.max_pair_distance - 1,
            self.max_pair_distance + 1,
        )
        signed_dist = signed_dist + (self.max_pair_distance + 1)
        dist_emb = self.distance_embedding(signed_dist)  # [N, N, Dd]
        dist_emb = dist_emb.unsqueeze(0).expand(B, N, N, self.distance_emb_dim)
        return dist_emb

    def _compute_trigger_conditioned_attention(
        self,
        node_repr: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """
        Computes trigger-conditioned pair attention score [B, N, N, 1]
        score(t, a) = softmax_t_over_arguments(Q_t K_a^T / sqrt(d))
        """
        Q = self.trigger_query(node_repr)    # [B, N, D]
        K = self.argument_key(node_repr)     # [B, N, D]

        scores = torch.matmul(Q, K.transpose(1, 2)) / (node_repr.size(-1) ** 0.5)  # [B, N, N]
        attn = torch.softmax(scores, dim=-1)
        return attn.unsqueeze(-1)

    def build_pair_representation(
        self,
        node_repr: torch.Tensor,         # [B, N, D]
        adj: torch.Tensor,               # [B, N, N]
        trigger_logits: torch.Tensor,    # [B, N, T]
        ner_logits: torch.Tensor,        # [B, N, C]
        case_type_ids: torch.Tensor,     # [B, N]
    ):
        B, N, D = node_repr.size()
        device = node_repr.device

        hi = node_repr.unsqueeze(2).expand(B, N, N, D)  # trigger side
        hj = node_repr.unsqueeze(1).expand(B, N, N, D)  # argument side

        pair_abs = torch.abs(hi - hj)
        pair_mul = hi * hj

        dist_emb = self._build_distance_embedding(B, N, device)

        adj_feat = adj.unsqueeze(-1)

        self_feat = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N).unsqueeze(-1)

        # trigger-conditioned attention
        attn_score = self._compute_trigger_conditioned_attention(node_repr)  # [B, N, N, 1]

        # bilinear compatibility
        bilinear_score = self.bilinear(
            hi.reshape(B * N * N, D),
            hj.reshape(B * N * N, D),
        ).reshape(B, N, N, 1)

        # trigger confidence from trigger logits
        trigger_probs = torch.softmax(trigger_logits, dim=-1)
        trigger_conf = trigger_probs.max(dim=-1).values  # [B, N]
        trigger_conf = trigger_conf.unsqueeze(2).expand(B, N, N).unsqueeze(-1)

        # argument-side ner confidence
        ner_probs = torch.softmax(ner_logits, dim=-1)
        ner_conf = ner_probs.max(dim=-1).values  # [B, N]
        ner_conf = ner_conf.unsqueeze(1).expand(B, N, N).unsqueeze(-1)

        # argument-side case confidence
        case_conf = (case_type_ids != 0).float().unsqueeze(1).expand(B, N, N).unsqueeze(-1)

        pair_repr = torch.cat(
            [
                hi,
                hj,
                pair_abs,
                pair_mul,
                dist_emb,
                adj_feat,
                self_feat,
                attn_score,
                bilinear_score,
                trigger_conf,
                ner_conf,
                case_conf,
            ],
            dim=-1,
        )

        # base pair mask from local distance + not self
        idx = torch.arange(N, device=device)
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        distance_mask = (dist <= self.max_pair_distance).unsqueeze(0).expand(B, N, N)
        self_mask = (torch.eye(N, device=device) == 0).unsqueeze(0).expand(B, N, N)

        pair_base_mask = distance_mask & self_mask
        return pair_repr, pair_base_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        adj: torch.Tensor,
        edge_type_ids: torch.Tensor,
        affix_ids: torch.Tensor,
        case_type_ids: torch.Tensor,
        morph_feats: torch.Tensor,
        pos_ids: torch.Tensor | None = None,
        ner_ids: torch.Tensor | None = None,
        use_gold_tags: bool = False,
    ):
        # 1) sequence branch
        seq_repr, pos_logits, ner_logits = self.tagger(input_ids, lengths)

        pred_pos = pos_logits.argmax(-1)
        pred_ner = ner_logits.argmax(-1)

        pos_for_embed = pos_ids if (use_gold_tags and pos_ids is not None) else pred_pos
        ner_for_embed = ner_ids if (use_gold_tags and ner_ids is not None) else pred_ner

        # 2) graph / symbolic branch
        graph_base = self.graph_embedding(input_ids)
        affix_repr = self.affix_embedding(affix_ids)
        pos_repr = self.pos_embedding(pos_for_embed)
        ner_repr = self.ner_embedding(ner_for_embed)
        case_type_repr = self.case_type_embedding(case_type_ids)
        edge_repr = self.edge_type_embedding(edge_type_ids)  # [B, N, N, Ee]

        # 3) fuse token features
        fused = torch.cat(
            [
                seq_repr,
                graph_base,
                affix_repr,
                pos_repr,
                ner_repr,
                case_type_repr,
                morph_feats,
            ],
            dim=-1,
        )

        # 4) relational graph reasoning
        node_repr = self.rgcn(fused, adj, edge_repr)

        # 5) trigger prediction
        trigger_logits = self.trigger_head(node_repr)

        # 6) pair representation + 2-stage argument heads
        pair_repr, pair_base_mask = self.build_pair_representation(
            node_repr=node_repr,
            adj=adj,
            trigger_logits=trigger_logits,
            ner_logits=ner_logits,
            case_type_ids=case_type_ids,
        )

        arg_exist_logits = self.arg_exist_mlp(pair_repr)
        role_logits = self.role_mlp(pair_repr)

        return {
            "seq_repr": seq_repr,
            "pos_logits": pos_logits,
            "ner_logits": ner_logits,
            "trigger_logits": trigger_logits,
            "arg_exist_logits": arg_exist_logits,
            "role_logits": role_logits,
            "pair_base_mask": pair_base_mask,
            "node_repr": node_repr,
        }


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Kept for compatibility with your current pipeline.
    For relational GCN we still use normalized dense adjacency as message mask/weight.
    """
    B, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N)
    adj_hat = adj + eye
    deg = adj_hat.sum(dim=-1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    D = torch.diag_embed(deg_inv_sqrt)
    return torch.bmm(torch.bmm(D, adj_hat), D)