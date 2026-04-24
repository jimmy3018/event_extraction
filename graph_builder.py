
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch


@dataclass
class SentenceGraph:
    tokens: List[str]
    edges: List[Tuple[int, int, str]]


EDGE_TYPE_TO_ID = {
    "none": 0,
    "self": 1,
    "next": 2,
    "prev": 3,
    "trigger_arg": 4,
    "arg_trigger": 5,
    "trigger_entity": 6,
    "entity_trigger": 7,
    "trigger_casearg": 8,
    "casearg_trigger": 9,
    "case_hint": 10,
    "case_back": 11,
    "arg_support": 12,
    "trigger_support": 13,
    "skip": 14,
    "skip_back": 15,
    "trigger_arg_strong": 16,
    "arg_trigger_strong": 17,
}


class GraphBuilder:
    """
    Structured event-oriented graph builder.

    Node categories are implicit from token labels:
    - trigger candidates
    - argument candidates
    - case-marked candidates
    - context tokens

    Edge families:
    - self / next / prev
    - trigger <-> argument
    - case-marked <-> trigger
    - argument <-> argument local support
    - trigger <-> trigger local support
    - short skip edges
    """

    VERB_TAGS = {"VERB", "AUX"}
    ENTITY_TAGS = {"PER", "ORG", "LOC", "TIM", "CUR", "DES"}
    ARG_POS_TAGS = {"NOUN", "PROPN", "PRON", "NUM"}

    def build(
        self,
        tokens: List[str],
        pos_tags: List[str],
        ner_tags: List[str],
        case_flags: List[int],
    ) -> SentenceGraph:
        n = len(tokens)
        edges: List[Tuple[int, int, str]] = []

        trigger_candidates = self._get_trigger_candidates(pos_tags)
        argument_candidates = self._get_argument_candidates(pos_tags, ner_tags)
        case_candidates = self._get_case_candidates(case_flags)

        edges.extend(self._build_sequence_edges(n))
        edges.extend(
            self._build_trigger_argument_edges(
                trigger_candidates=trigger_candidates,
                argument_candidates=argument_candidates,
                ner_tags=ner_tags,
                case_flags=case_flags,
                max_dist=5,
            )
        )
        edges.extend(
            self._build_case_trigger_edges(
                case_candidates=case_candidates,
                trigger_candidates=trigger_candidates,
                max_dist=5,
            )
        )
        edges.extend(
            self._build_argument_argument_edges(
                argument_candidates=argument_candidates,
                max_dist=2,
            )
        )
        edges.extend(
            self._build_trigger_trigger_edges(
                trigger_candidates=trigger_candidates,
                max_dist=3,
            )
        )
        edges.extend(self._build_skip_edges(n, skip=2))

        edges = list(dict.fromkeys(edges))
        return SentenceGraph(tokens=tokens, edges=edges)

    def _get_trigger_candidates(self, pos_tags: List[str]) -> List[int]:
        return [i for i, p in enumerate(pos_tags) if p in self.VERB_TAGS]

    def _get_argument_candidates(self, pos_tags: List[str], ner_tags: List[str]) -> List[int]:
        out = []
        for i, (p, t) in enumerate(zip(pos_tags, ner_tags)):
            if t in self.ENTITY_TAGS or p in self.ARG_POS_TAGS:
                out.append(i)
        return out

    def _get_case_candidates(self, case_flags: List[int]) -> List[int]:
        return [i for i, v in enumerate(case_flags) if v == 1]

    def _build_sequence_edges(self, n: int) -> List[Tuple[int, int, str]]:
        edges = []
        for i in range(n):
            edges.append((i, i, "self"))
            if i + 1 < n:
                edges.append((i, i + 1, "next"))
                edges.append((i + 1, i, "prev"))
        return edges

    def _build_trigger_argument_edges(
        self,
        trigger_candidates: List[int],
        argument_candidates: List[int],
        ner_tags: List[str],
        case_flags: List[int],
        max_dist: int = 5,
    ) -> List[Tuple[int, int, str]]:
        edges = []
        for t in trigger_candidates:
            for a in argument_candidates:
                if t == a:
                    continue
                if abs(t - a) > max_dist:
                    continue

                if ner_tags[a] in self.ENTITY_TAGS and case_flags[a] == 1:
                    ta = "trigger_arg_strong"
                    at = "arg_trigger_strong"
                elif ner_tags[a] in self.ENTITY_TAGS:
                    ta = "trigger_entity"
                    at = "entity_trigger"
                elif case_flags[a] == 1:
                    ta = "trigger_casearg"
                    at = "casearg_trigger"
                else:
                    ta = "trigger_arg"
                    at = "arg_trigger"

                edges.append((t, a, ta))
                edges.append((a, t, at))
        return edges

    def _build_case_trigger_edges(
        self,
        case_candidates: List[int],
        trigger_candidates: List[int],
        max_dist: int = 5,
    ) -> List[Tuple[int, int, str]]:
        edges = []
        for c in case_candidates:
            for t in trigger_candidates:
                if c == t:
                    continue
                if abs(c - t) <= max_dist:
                    edges.append((c, t, "case_hint"))
                    edges.append((t, c, "case_back"))
        return edges

    def _build_argument_argument_edges(
        self,
        argument_candidates: List[int],
        max_dist: int = 2,
    ) -> List[Tuple[int, int, str]]:
        edges = []
        for i in argument_candidates:
            for j in argument_candidates:
                if i == j:
                    continue
                if abs(i - j) <= max_dist:
                    edges.append((i, j, "arg_support"))
        return edges

    def _build_trigger_trigger_edges(
        self,
        trigger_candidates: List[int],
        max_dist: int = 3,
    ) -> List[Tuple[int, int, str]]:
        edges = []
        for i in trigger_candidates:
            for j in trigger_candidates:
                if i == j:
                    continue
                if abs(i - j) <= max_dist:
                    edges.append((i, j, "trigger_support"))
        return edges

    def _build_skip_edges(self, n: int, skip: int = 2) -> List[Tuple[int, int, str]]:
        edges = []
        for i in range(n):
            j = i + skip
            if j < n:
                edges.append((i, j, "skip"))
                edges.append((j, i, "skip_back"))
        return edges

    @staticmethod
    def adjacency_matrix(graph: SentenceGraph):
        n = len(graph.tokens)
        adj = torch.zeros((n, n), dtype=torch.float)
        for src, dst, _ in graph.edges:
            adj[src, dst] = 1.0
        return adj

    @staticmethod
    def edge_type_matrix(graph: SentenceGraph):
        n = len(graph.tokens)
        mat = torch.zeros((n, n), dtype=torch.long)
        for src, dst, etype in graph.edges:
            mat[src, dst] = EDGE_TYPE_TO_ID.get(etype, 0)
        return mat

    @staticmethod
    def normalized_adjacency(graph: SentenceGraph):
        adj = GraphBuilder.adjacency_matrix(graph)
        n = adj.size(0)
        eye = torch.eye(n, dtype=torch.float)
        adj_hat = adj + eye
        deg = adj_hat.sum(dim=-1).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        return d_inv_sqrt @ adj_hat @ d_inv_sqrt

    @staticmethod
    def compute_degree_salience(graph: SentenceGraph):
        adj = GraphBuilder.adjacency_matrix(graph)
        scores = adj.sum(dim=-1)
        if scores.numel() == 0:
            return scores
        return scores / (scores.max() + 1e-8)

    @staticmethod
    def compute_pagerank_salience(
        graph: SentenceGraph,
        alpha: float = 0.85,
        max_iter: int = 20,
        tol: float = 1e-6,
    ):
        adj = GraphBuilder.adjacency_matrix(graph)
        n = adj.size(0)
        if n == 0:
            return torch.zeros(0, dtype=torch.float)

        row_sum = adj.sum(dim=-1, keepdim=True)
        dead_end_mask = (row_sum == 0).squeeze(-1)
        if dead_end_mask.any():
            adj = adj.clone()
            for i in range(n):
                if dead_end_mask[i]:
                    adj[i] = 1.0
            row_sum = adj.sum(dim=-1, keepdim=True)

        trans = adj / row_sum.clamp(min=1.0)
        score = torch.full((n,), 1.0 / n, dtype=torch.float)
        base = torch.full((n,), (1.0 - alpha) / n, dtype=torch.float)

        for _ in range(max_iter):
            new_score = base + alpha * torch.matmul(trans.t(), score)
            if torch.norm(new_score - score, p=1).item() < tol:
                score = new_score
                break
            score = new_score

        return score / (score.max() + 1e-8)

    @staticmethod
    def compute_gcn_salience(graph: SentenceGraph, steps: int = 2):
        a_norm = GraphBuilder.normalized_adjacency(graph)
        n = a_norm.size(0)
        if n == 0:
            return torch.zeros(0, dtype=torch.float)

        h = torch.ones(n, dtype=torch.float)
        for _ in range(steps):
            h = torch.matmul(a_norm, h)
        return h / (h.max() + 1e-8)

    @staticmethod
    def compute_node_salience(graph: SentenceGraph, method: str = "pagerank"):
        if method == "degree":
            return GraphBuilder.compute_degree_salience(graph)
        if method == "gcn":
            return GraphBuilder.compute_gcn_salience(graph)
        return GraphBuilder.compute_pagerank_salience(graph)

    @staticmethod
    def compute_event_salience(
        trigger_span: List[int],
        argument_spans: List[List[int]],
        node_scores,
        sentence_index: int = 0,
        total_sentences: int = 1,
    ) -> float:
        if node_scores.numel() == 0:
            return 0.0

        t_start, t_end = trigger_span
        trigger_score = node_scores[t_start:t_end + 1].mean().item()

        arg_score = 0.0
        valid_arg_count = 0
        for span in argument_spans:
            if span is None or len(span) != 2:
                continue
            a_start, a_end = span
            if a_start < 0 or a_end >= len(node_scores) or a_start > a_end:
                continue
            arg_score += node_scores[a_start:a_end + 1].mean().item()
            valid_arg_count += 1

        if valid_arg_count > 0:
            arg_score /= valid_arg_count

        arg_count_bonus = min(valid_arg_count / 3.0, 1.0)

        if total_sentences <= 1:
            position_score = 1.0
        else:
            position_score = 1.0 - (sentence_index / max(total_sentences - 1, 1))

        salience = (
            0.50 * trigger_score +
            0.20 * arg_score +
            0.15 * arg_count_bonus +
            0.15 * position_score
        )
        return float(max(0.0, min(1.0, salience)))

    @staticmethod
    def attach_event_salience(
        graph: SentenceGraph,
        events: List[Dict[str, Any]],
        sentence_index: int = 0,
        total_sentences: int = 1,
        method: str = "pagerank",
    ) -> List[Dict[str, Any]]:
        node_scores = GraphBuilder.compute_node_salience(graph, method=method)

        enriched = []
        for ev in events:
            ev_copy = dict(ev)
            trigger_span = ev_copy.get("trigger_span", None)
            if trigger_span is None or len(trigger_span) != 2:
                ev_copy["salience"] = 0.0
                enriched.append(ev_copy)
                continue

            arg_spans = []
            for arg in ev_copy.get("arguments", []):
                span = arg.get("span", None)
                if span is not None:
                    arg_spans.append(span)

            ev_copy["salience"] = GraphBuilder.compute_event_salience(
                trigger_span=trigger_span,
                argument_spans=arg_spans,
                node_scores=node_scores,
                sentence_index=sentence_index,
                total_sentences=total_sentences,
            )
            enriched.append(ev_copy)

        return enriched
