
from __future__ import annotations

import json
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from utils import Vocab
from morphology import HeuristicMorphSegmenter
from graph_builder import GraphBuilder

PAD_LABEL = 0

CASE_TYPE_TO_ID = {
    "NONE": 0,
    "LOC": 1,
    "AGENT": 2,
    "OBJ": 3,
    "GEN": 4,
    "TIME": 5,
    "OTHER": 6,
}


class MEREDDataset(Dataset):
    """
    Sentence-level MERED style dataset.

    Expects each document to contain:
    - doc_id
    - sentence_annotations: list of sentence dicts with tokens, pos_labels, ner_labels, events, entities

    Produces sentence-level tensors for:
    - tokens / labels
    - adjacency and edge type matrices
    - affixes, case types, morphology features
    - trigger labels and pairwise role matrix
    """

    def __init__(
        self,
        path: str,
        config,
        vocabs: Dict[str, Vocab] | None = None,
        build_vocabs: bool = False,
    ):
        self.path = path
        self.config = config
        self.segmenter = HeuristicMorphSegmenter()
        self.graph_builder = GraphBuilder()

        self.docs = self._load(path)
        self.instances = self._build_instances(self.docs)

        if build_vocabs:
            self.vocabs = self._build_vocabs()
        else:
            if vocabs is None:
                raise ValueError("vocabs must be provided when build_vocabs=False")
            self.vocabs = vocabs

        self.encoded = [self._encode_instance(x) for x in self.instances]

    def _load(self, path: str) -> List[Dict[str, Any]]:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return [data]
        return data

    def _build_instances(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        instances: List[Dict[str, Any]] = []

        for doc in docs:
            doc_id = doc["doc_id"]
            sentence_annotations = doc.get("sentence_annotations", [])

            for sent in sentence_annotations:
                sid = sent["sid"]
                text = sent.get("text", "")
                tokens = sent.get("tokens", [])
                pos_labels = sent.get("pos_labels", ["X"] * len(tokens))
                ner_labels = sent.get("ner_labels", ["O"] * len(tokens))
                sent_events = sent.get("events", [])
                sent_entities = sent.get("entities", [])

                if not (len(tokens) == len(pos_labels) == len(ner_labels)):
                    raise ValueError(
                        f"Length mismatch in doc {doc_id}, sid {sid}: "
                        f"len(tokens)={len(tokens)}, len(pos_labels)={len(pos_labels)}, len(ner_labels)={len(ner_labels)}"
                    )

                n = len(tokens)
                trigger_labels = ["O"] * n
                role_matrix = [["O" for _ in range(n)] for _ in range(n)]
                trigger_spans = []
                argument_spans = []

                morph_infos = [self.segmenter.segment(tok) for tok in tokens]
                case_flags = [1 if getattr(mi, "case_markers", []) else 0 for mi in morph_infos]
                graph = self.graph_builder.build(tokens, pos_labels, ner_labels, case_flags)

                enriched_events = self.graph_builder.attach_event_salience(
                    graph=graph,
                    events=sent_events,
                    sentence_index=max(sid - 1, 0),
                    total_sentences=max(len(sentence_annotations), 1),
                    method="pagerank",
                )

                event_salience_map = {}
                for ev in enriched_events:
                    if "event_id" in ev:
                        event_salience_map[ev["event_id"]] = ev.get("salience", 0.0)

                for ev in sent_events:
                    ev_type = ev.get("type", "O")
                    trig_span = ev.get("trigger_span", None)

                    if trig_span is None or len(trig_span) != 2:
                        continue

                    start, end = trig_span
                    if start < 0 or end >= n or start > end:
                        continue

                    for i in range(start, end + 1):
                        trigger_labels[i] = ev_type

                    trigger_head = start
                    ev_salience = float(event_salience_map.get(ev.get("event_id"), 0.0))

                    trigger_spans.append({
                        "event_id": ev.get("event_id"),
                        "type": ev_type,
                        "span": [start, end],
                        "multiword": end > start,
                        "token_text": tokens[start:end + 1],
                        "trigger_text": ev.get("trigger", " ".join(tokens[start:end + 1])),
                        "salience": ev_salience,
                    })

                    for arg in ev.get("arguments", []):
                        arg_role = arg.get("role", "O")
                        arg_span = arg.get("span", None)

                        if arg_span is None or len(arg_span) != 2:
                            continue

                        a_start, a_end = arg_span
                        if a_start < 0 or a_end >= n or a_start > a_end:
                            continue

                        for i in range(a_start, a_end + 1):
                            role_matrix[trigger_head][i] = arg_role

                        argument_spans.append({
                            "event_id": ev.get("event_id"),
                            "role": arg_role,
                            "trigger_head": trigger_head,
                            "trigger_span": [start, end],
                            "arg_span": [a_start, a_end],
                            "multiword": a_end > a_start,
                            "token_text": tokens[a_start:a_end + 1],
                            "arg_text": arg.get("text", " ".join(tokens[a_start:a_end + 1])),
                            "salience": ev_salience,
                        })

                instances.append({
                    "doc_id": doc_id,
                    "sid": sid,
                    "text": text,
                    "tokens": tokens,
                    "pos_labels": pos_labels,
                    "ner_labels": ner_labels,
                    "entities": sent_entities,
                    "events": sent_events,
                    "trigger_labels": trigger_labels,
                    "role_matrix": role_matrix,
                    "morph_infos": morph_infos,
                    "graph": graph,
                    "trigger_spans": trigger_spans,
                    "argument_spans": argument_spans,
                })

        return instances

    def _build_vocabs(self) -> Dict[str, Vocab]:
        word_vocab = Vocab([self.config.pad_token, self.config.unk_token])
        affix_vocab = Vocab([self.config.pad_token, self.config.unk_token])
        pos_vocab = Vocab([self.config.pad_token, "X"])
        ner_vocab = Vocab([self.config.pad_token, "O"])
        trigger_vocab = Vocab([self.config.pad_token, "O"])
        role_vocab = Vocab([self.config.pad_token, "O"])

        word_tokens = []
        affix_tokens = []

        for inst in self.instances:
            word_tokens.extend(inst["tokens"])

            for mi in inst["morph_infos"]:
                affix_tokens.extend(getattr(mi, "affixes", []) if getattr(mi, "affixes", []) else ["<NO_AFFIX>"])

            for x in inst["pos_labels"]:
                pos_vocab.add(x)
            for x in inst["ner_labels"]:
                ner_vocab.add(x)
            for x in inst["trigger_labels"]:
                trigger_vocab.add(x)
            for row in inst["role_matrix"]:
                for role in row:
                    role_vocab.add(role)

        word_vocab.build(word_tokens)
        affix_vocab.build(affix_tokens)

        return {
            "word": word_vocab,
            "affix": affix_vocab,
            "pos": pos_vocab,
            "ner": ner_vocab,
            "trigger": trigger_vocab,
            "role": role_vocab,
        }

    def _encode_instance(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        wv = self.vocabs["word"]
        av = self.vocabs["affix"]
        pv = self.vocabs["pos"]
        nv = self.vocabs["ner"]
        tv = self.vocabs["trigger"]
        rv = self.vocabs["role"]

        input_ids = [wv.encode(t, self.config.unk_token) for t in inst["tokens"]]
        pos_ids = [pv.encode(x, "X") for x in inst["pos_labels"]]
        ner_ids = [nv.encode(x, "O") for x in inst["ner_labels"]]
        trigger_ids = [tv.encode(x, "O") for x in inst["trigger_labels"]]

        affix_ids = []
        case_type_ids = []
        morph_feats = []

        for mi in inst["morph_infos"]:
            aff = getattr(mi, "affixes", [])[0] if getattr(mi, "affixes", []) else "<NO_AFFIX>"
            affix_ids.append(av.encode(aff, self.config.unk_token))

            case_type = getattr(mi, "case_type", "NONE")
            case_type_id = CASE_TYPE_TO_ID.get(case_type, 0)
            case_type_ids.append(case_type_id)

            full_feat = list(getattr(mi, "morph_flags", [])) + list(getattr(mi, "case_type_vec", []))
            if len(full_feat) < self.config.morph_feat_dim:
                full_feat = full_feat + [0.0] * (self.config.morph_feat_dim - len(full_feat))
            else:
                full_feat = full_feat[:self.config.morph_feat_dim]
            morph_feats.append(full_feat)

        role_ids = []
        for row in inst["role_matrix"]:
            role_ids.append([rv.encode(x, "O") for x in row])

        return {
            "doc_id": inst["doc_id"],
            "sid": inst["sid"],
            "tokens": inst["tokens"],
            "entities": inst.get("entities", []),
            "events": inst.get("events", []),
            "trigger_spans": inst.get("trigger_spans", []),
            "argument_spans": inst.get("argument_spans", []),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "ner_ids": torch.tensor(ner_ids, dtype=torch.long),
            "trigger_ids": torch.tensor(trigger_ids, dtype=torch.long),
            "role_ids": torch.tensor(role_ids, dtype=torch.long),
            "affix_ids": torch.tensor(affix_ids, dtype=torch.long),
            "case_type_ids": torch.tensor(case_type_ids, dtype=torch.long),
            "morph_feats": torch.tensor(morph_feats, dtype=torch.float),
            "adj": self.graph_builder.adjacency_matrix(inst["graph"]),
            "edge_type_ids": self.graph_builder.edge_type_matrix(inst["graph"]),
        }

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    bsz = len(batch)

    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    pos_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    ner_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    trigger_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    affix_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    case_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    lengths = torch.zeros((bsz,), dtype=torch.long)

    feat_dim = batch[0]["morph_feats"].size(-1)
    morph_feats = torch.zeros((bsz, max_len, feat_dim), dtype=torch.float)
    adj = torch.zeros((bsz, max_len, max_len), dtype=torch.float)
    edge_type_ids = torch.zeros((bsz, max_len, max_len), dtype=torch.long)
    role_ids = torch.zeros((bsz, max_len, max_len), dtype=torch.long)

    meta = []

    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        lengths[i] = n

        input_ids[i, :n] = item["input_ids"]
        pos_ids[i, :n] = item["pos_ids"]
        ner_ids[i, :n] = item["ner_ids"]
        trigger_ids[i, :n] = item["trigger_ids"]
        affix_ids[i, :n] = item["affix_ids"]
        case_type_ids[i, :n] = item["case_type_ids"]
        morph_feats[i, :n] = item["morph_feats"]
        adj[i, :n, :n] = item["adj"]
        edge_type_ids[i, :n, :n] = item["edge_type_ids"]
        role_ids[i, :n, :n] = item["role_ids"]

        meta.append({
            "doc_id": item["doc_id"],
            "sid": item["sid"],
            "tokens": item["tokens"],
            "entities": item.get("entities", []),
            "events": item.get("events", []),
            "trigger_spans": item.get("trigger_spans", []),
            "argument_spans": item.get("argument_spans", []),
        })

    return {
        "input_ids": input_ids,
        "pos_ids": pos_ids,
        "ner_ids": ner_ids,
        "trigger_ids": trigger_ids,
        "affix_ids": affix_ids,
        "case_type_ids": case_type_ids,
        "morph_feats": morph_feats,
        "adj": adj,
        "edge_type_ids": edge_type_ids,
        "role_ids": role_ids,
        "lengths": lengths,
        "meta": meta,
    }
