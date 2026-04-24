
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import MEREDDataset, collate_fn
from models import EventExtractor, normalize_adjacency
from utils import Vocab


def safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def restore_vocab(v) -> Vocab:
    if hasattr(v, "stoi") and hasattr(v, "itos"):
        return v

    if isinstance(v, dict) and "stoi" in v and "itos" in v:
        obj = Vocab([])
        obj.stoi = v["stoi"]
        obj.itos = v["itos"]
        return obj

    raise ValueError(f"Unsupported vocab format: {type(v)}")


def vocab_size(vocab) -> int:
    if hasattr(vocab, "itos"):
        return len(vocab.itos)
    return len(vocab)


def get_pos_candidate_ids(pos_vocab) -> Set[int]:
    candidate_tags = {"NOUN", "PROPN", "PRON", "NUM"}
    return {pos_vocab.stoi[tag] for tag in candidate_tags if tag in pos_vocab.stoi}


def build_argument_candidate_mask_from_predictions(
    pred_pos: torch.Tensor,
    pred_ner: torch.Tensor,
    pred_trigger: torch.Tensor,
    case_type_ids: torch.Tensor,
    pair_base_mask: torch.Tensor,
    pos_candidate_ids: Set[int],
    ner_o_idx: int,
    trigger_o_idx: int,
) -> torch.Tensor:
    trigger_mask = pred_trigger != trigger_o_idx

    pos_ok = torch.zeros_like(pred_pos, dtype=torch.bool)
    for idx in pos_candidate_ids:
        pos_ok |= (pred_pos == idx)

    ner_ok = pred_ner != ner_o_idx
    case_ok = case_type_ids != 0
    arg_mask = pos_ok | ner_ok | case_ok

    pair_mask = trigger_mask.unsqueeze(2) & arg_mask.unsqueeze(1) & pair_base_mask
    return pair_mask


def decode_two_stage_roles(
    arg_exist_logits: torch.Tensor,
    role_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    role_o_idx: int,
    exist_threshold: float = 0.50,
    top_k_per_trigger: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    exist_probs = torch.softmax(arg_exist_logits, dim=-1)[..., 1]
    pred_exist = (exist_probs >= exist_threshold).long()
    pred_role = role_logits.argmax(-1)

    valid_pairs = pair_mask & (pred_exist == 1)

    B, N, _ = exist_probs.shape
    topk_mask = torch.zeros_like(valid_pairs, dtype=torch.bool)

    for b in range(B):
        for t in range(N):
            cand_idx = torch.where(valid_pairs[b, t])[0]
            if cand_idx.numel() == 0:
                continue
            cand_scores = exist_probs[b, t, cand_idx]
            k = min(top_k_per_trigger, cand_idx.numel())
            top_pos = torch.topk(cand_scores, k=k).indices
            keep_idx = cand_idx[top_pos]
            topk_mask[b, t, keep_idx] = True

    pred_exist = torch.where(topk_mask, pred_exist, torch.zeros_like(pred_exist))
    pred_role = torch.where(
        topk_mask,
        pred_role,
        torch.full_like(pred_role, fill_value=role_o_idx),
    )
    return pred_exist, pred_role


def compute_token_classification_metrics(
    gold_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    vocab,
    ignore_index: int = 0,
    negative_label: str = "O",
) -> Dict[str, Any]:
    if gold_tensor.numel() == 0 or pred_tensor.numel() == 0:
        return {"overall": prf(0, 0, 0), "per_label": {}}

    gold = gold_tensor.reshape(-1).tolist()
    pred = pred_tensor.reshape(-1).tolist()

    tp = fp = fn = 0
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for g, p in zip(gold, pred):
        if g == ignore_index:
            continue

        g_lab = vocab.decode(g)
        p_lab = vocab.decode(p)

        if g_lab == negative_label and p_lab == negative_label:
            continue

        if g_lab == p_lab and g_lab != negative_label:
            tp += 1
            label_stats[g_lab]["tp"] += 1
        else:
            if p_lab != negative_label:
                fp += 1
                label_stats[p_lab]["fp"] += 1
            if g_lab != negative_label:
                fn += 1
                label_stats[g_lab]["fn"] += 1

    overall = prf(tp, fp, fn)
    per_label = {}
    for label, stats in sorted(label_stats.items()):
        per_label[label] = prf(stats["tp"], stats["fp"], stats["fn"])

    return {"overall": overall, "per_label": per_label}


def compute_binary_pair_metrics(
    gold_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    mask_tensor: torch.Tensor | None = None,
    positive_value: int = 1,
) -> Dict[str, Any]:
    if gold_tensor.numel() == 0 or pred_tensor.numel() == 0:
        return prf(0, 0, 0)

    gold = gold_tensor.reshape(-1)
    pred = pred_tensor.reshape(-1)

    if mask_tensor is not None:
        mask = mask_tensor.reshape(-1).bool()
        gold = gold[mask]
        pred = pred[mask]

    if gold.numel() == 0 or pred.numel() == 0:
        return prf(0, 0, 0)

    gold_pos = gold == positive_value
    pred_pos = pred == positive_value

    tp = int((gold_pos & pred_pos).sum().item())
    fp = int((~gold_pos & pred_pos).sum().item())
    fn = int((gold_pos & ~pred_pos).sum().item())

    return prf(tp, fp, fn)


def collect_trigger_tuples_from_list(
    trigger_list: List[torch.Tensor],
    meta: List[Dict[str, Any]],
    vocab,
) -> List[Tuple]:
    tuples = []
    for b, trigger_ids in enumerate(trigger_list):
        tokens = meta[b]["tokens"]
        doc_id = meta[b]["doc_id"]
        sid = meta[b]["sid"]

        for i in range(len(tokens)):
            label = vocab.decode(int(trigger_ids[i].item()))
            if label != "O":
                tuples.append((doc_id, sid, i, tokens[i], label))
    return tuples


def collect_argument_tuples_from_list(
    role_list: List[torch.Tensor],
    trigger_list: List[torch.Tensor],
    meta: List[Dict[str, Any]],
    role_vocab,
    trigger_vocab,
) -> List[Tuple]:
    tuples = []

    for b, role_ids in enumerate(role_list):
        tokens = meta[b]["tokens"]
        doc_id = meta[b]["doc_id"]
        sid = meta[b]["sid"]
        trigger_ids = trigger_list[b]
        n = len(tokens)

        for trig_idx in range(n):
            trig_label = trigger_vocab.decode(int(trigger_ids[trig_idx].item()))
            if trig_label == "O":
                continue

            for arg_idx in range(n):
                role = role_vocab.decode(int(role_ids[trig_idx, arg_idx].item()))
                if role == "O":
                    continue

                tuples.append((
                    doc_id,
                    sid,
                    trig_idx,
                    tokens[trig_idx],
                    trig_label,
                    arg_idx,
                    tokens[arg_idx],
                    role,
                ))
    return tuples


def collect_argument_existence_tuples_from_list(
    exist_list: List[torch.Tensor],
    trigger_list: List[torch.Tensor],
    meta: List[Dict[str, Any]],
    trigger_vocab,
) -> List[Tuple]:
    tuples = []

    for b, exist_ids in enumerate(exist_list):
        tokens = meta[b]["tokens"]
        doc_id = meta[b]["doc_id"]
        sid = meta[b]["sid"]
        trigger_ids = trigger_list[b]
        n = len(tokens)

        for trig_idx in range(n):
            trig_label = trigger_vocab.decode(int(trigger_ids[trig_idx].item()))
            if trig_label == "O":
                continue

            for arg_idx in range(n):
                if int(exist_ids[trig_idx, arg_idx].item()) == 1:
                    tuples.append((
                        doc_id,
                        sid,
                        trig_idx,
                        tokens[trig_idx],
                        trig_label,
                        arg_idx,
                        tokens[arg_idx],
                    ))
    return tuples


def set_prf(gold_items: List[Tuple], pred_items: List[Tuple]) -> Dict[str, Any]:
    gold_set = Counter(gold_items)
    pred_set = Counter(pred_items)

    tp = 0
    for item, gc in gold_set.items():
        tp += min(gc, pred_set.get(item, 0))

    fp = sum(pred_set.values()) - tp
    fn = sum(gold_set.values()) - tp

    return prf(tp, fp, fn)


def event_level_metrics_from_lists(
    gold_roles_list: List[torch.Tensor],
    pred_roles_list: List[torch.Tensor],
    gold_triggers_list: List[torch.Tensor],
    pred_triggers_list: List[torch.Tensor],
    gold_exist_list: List[torch.Tensor],
    pred_exist_list: List[torch.Tensor],
    meta: List[Dict[str, Any]],
    role_vocab,
    trigger_vocab,
) -> Dict[str, Any]:
    gold_trigger_items = collect_trigger_tuples_from_list(gold_triggers_list, meta, trigger_vocab)
    pred_trigger_items = collect_trigger_tuples_from_list(pred_triggers_list, meta, trigger_vocab)

    gold_exist_items = collect_argument_existence_tuples_from_list(
        gold_exist_list, gold_triggers_list, meta, trigger_vocab
    )
    pred_exist_items = collect_argument_existence_tuples_from_list(
        pred_exist_list, pred_triggers_list, meta, trigger_vocab
    )

    gold_arg_items = collect_argument_tuples_from_list(
        gold_roles_list, gold_triggers_list, meta, role_vocab, trigger_vocab
    )
    pred_arg_items = collect_argument_tuples_from_list(
        pred_roles_list, pred_triggers_list, meta, role_vocab, trigger_vocab
    )

    return {
        "trigger_identification": set_prf(
            [(d, s, i, tok) for d, s, i, tok, lab in gold_trigger_items],
            [(d, s, i, tok) for d, s, i, tok, lab in pred_trigger_items],
        ),
        "trigger_classification": set_prf(gold_trigger_items, pred_trigger_items),
        "argument_existence_identification": set_prf(gold_exist_items, pred_exist_items),
        "argument_identification": set_prf(
            [(d, s, ti, tt, tl, ai, at) for d, s, ti, tt, tl, ai, at, r in gold_arg_items],
            [(d, s, ti, tt, tl, ai, at) for d, s, ti, tt, tl, ai, at, r in pred_arg_items],
        ),
        "argument_role_classification": set_prf(gold_arg_items, pred_arg_items),
    }


def decode_token_labels(ids: torch.Tensor, vocab, length: int) -> List[str]:
    return [vocab.decode(int(ids[i].item())) for i in range(length)]


def extract_events_from_prediction(
    tokens: List[str],
    trigger_ids: torch.Tensor,
    role_ids: torch.Tensor,
    trigger_vocab,
    role_vocab,
) -> List[Dict[str, Any]]:
    events = []
    n = len(tokens)

    for trig_idx in range(n):
        trig_label = trigger_vocab.decode(int(trigger_ids[trig_idx].item()))
        if trig_label == "O":
            continue

        args = []
        for arg_idx in range(n):
            role = role_vocab.decode(int(role_ids[trig_idx, arg_idx].item()))
            if role == "O":
                continue
            args.append({
                "arg_index": arg_idx,
                "arg_token": tokens[arg_idx],
                "role": role,
            })

        events.append({
            "trigger_index": trig_idx,
            "trigger_token": tokens[trig_idx],
            "trigger_type": trig_label,
            "arguments": args,
        })
    return events


def build_prediction_dump(
    all_gold_pos: List[torch.Tensor],
    all_pred_pos: List[torch.Tensor],
    all_gold_ner: List[torch.Tensor],
    all_pred_ner: List[torch.Tensor],
    all_gold_trig: List[torch.Tensor],
    all_pred_trig: List[torch.Tensor],
    all_gold_role: List[torch.Tensor],
    all_pred_role: List[torch.Tensor],
    all_gold_exist: List[torch.Tensor],
    all_pred_exist: List[torch.Tensor],
    all_meta: List[Dict[str, Any]],
    vocabs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dump = []

    for i in range(len(all_meta)):
        meta = all_meta[i]
        tokens = meta["tokens"]
        n = len(tokens)

        gold_pos_labels = decode_token_labels(all_gold_pos[i], vocabs["pos"], n)
        pred_pos_labels = decode_token_labels(all_pred_pos[i], vocabs["pos"], n)

        gold_ner_labels = decode_token_labels(all_gold_ner[i], vocabs["ner"], n)
        pred_ner_labels = decode_token_labels(all_pred_ner[i], vocabs["ner"], n)

        gold_trigger_labels = decode_token_labels(all_gold_trig[i], vocabs["trigger"], n)
        pred_trigger_labels = decode_token_labels(all_pred_trig[i], vocabs["trigger"], n)

        token_rows = []
        for j in range(n):
            token_rows.append({
                "index": j,
                "token": tokens[j],
                "gold_pos": gold_pos_labels[j],
                "pred_pos": pred_pos_labels[j],
                "gold_ner": gold_ner_labels[j],
                "pred_ner": pred_ner_labels[j],
                "gold_trigger": gold_trigger_labels[j],
                "pred_trigger": pred_trigger_labels[j],
            })

        gold_events = extract_events_from_prediction(
            tokens=tokens,
            trigger_ids=all_gold_trig[i],
            role_ids=all_gold_role[i],
            trigger_vocab=vocabs["trigger"],
            role_vocab=vocabs["role"],
        )

        pred_events = extract_events_from_prediction(
            tokens=tokens,
            trigger_ids=all_pred_trig[i],
            role_ids=all_pred_role[i],
            trigger_vocab=vocabs["trigger"],
            role_vocab=vocabs["role"],
        )

        dump.append({
            "doc_id": meta["doc_id"],
            "sid": meta["sid"],
            "tokens": tokens,
            "entities": meta.get("entities", []),
            "gold_events_metadata": meta.get("events", []),
            "gold_trigger_spans": meta.get("trigger_spans", []),
            "gold_argument_spans": meta.get("argument_spans", []),
            "gold_argument_existence": all_gold_exist[i].tolist(),
            "pred_argument_existence": all_pred_exist[i].tolist(),
            "token_predictions": token_rows,
            "gold_events_decoded": gold_events,
            "pred_events_decoded": pred_events,
        })

    return dump


def evaluate_model(model, dataloader, vocabs, cfg, device="cpu") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()

    role_o_idx = vocabs["role"].stoi["O"]
    ner_o_idx = vocabs["ner"].stoi["O"]
    trigger_o_idx = vocabs["trigger"].stoi["O"]
    pos_candidate_ids = get_pos_candidate_ids(vocabs["pos"])

    all_gold_pos = []
    all_pred_pos = []
    all_gold_ner = []
    all_pred_ner = []
    all_gold_trig = []
    all_pred_trig = []

    all_gold_exist = []
    all_pred_exist = []
    all_gold_exist_flat = []
    all_pred_exist_flat = []

    all_gold_role = []
    all_pred_role = []
    all_gold_role_flat = []
    all_pred_role_flat = []

    all_meta = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            ner_ids = batch["ner_ids"].to(device)
            trigger_ids = batch["trigger_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            affix_ids = batch["affix_ids"].to(device)
            case_type_ids = batch["case_type_ids"].to(device)
            edge_type_ids = batch["edge_type_ids"].to(device)
            morph_feats = batch["morph_feats"].to(device)
            adj = normalize_adjacency(batch["adj"].to(device))
            lengths = batch["lengths"].to(device)

            out = model(
                input_ids=input_ids,
                lengths=lengths,
                adj=adj,
                edge_type_ids=edge_type_ids,
                affix_ids=affix_ids,
                case_type_ids=case_type_ids,
                morph_feats=morph_feats,
            )

            pred_pos = out["pos_logits"].argmax(-1)
            pred_ner = out["ner_logits"].argmax(-1)

            trigger_probs = torch.softmax(out["trigger_logits"], dim=-1)
            pred_trig = trigger_probs.argmax(-1)
            trigger_conf, _ = trigger_probs.max(dim=-1)

            pred_trig = torch.where(
                (pred_trig != trigger_o_idx) & (trigger_conf >= getattr(cfg, "trigger_threshold", 0.50)),
                pred_trig,
                torch.full_like(pred_trig, trigger_o_idx),
            )

            pair_mask = build_argument_candidate_mask_from_predictions(
                pred_pos=pred_pos,
                pred_ner=pred_ner,
                pred_trigger=pred_trig,
                case_type_ids=case_type_ids,
                pair_base_mask=out["pair_base_mask"],
                pos_candidate_ids=pos_candidate_ids,
                ner_o_idx=ner_o_idx,
                trigger_o_idx=trigger_o_idx,
            )

            pred_exist, pred_role = decode_two_stage_roles(
                arg_exist_logits=out["arg_exist_logits"],
                role_logits=out["role_logits"],
                pair_mask=pair_mask,
                role_o_idx=role_o_idx,
                exist_threshold=getattr(cfg, "arg_exist_threshold", 0.50),
                top_k_per_trigger=getattr(cfg, "top_k_args_per_trigger", 25),
            )

            gold_exist = (role_ids != role_o_idx).long()

            pred_exist = torch.where(pair_mask, pred_exist, torch.zeros_like(pred_exist))
            gold_exist = torch.where(pair_mask, gold_exist, torch.zeros_like(gold_exist))

            pred_pos = pred_pos.cpu()
            pred_ner = pred_ner.cpu()
            pred_trig = pred_trig.cpu()
            pred_exist = pred_exist.cpu()
            pred_role = pred_role.cpu()
            pair_mask = pair_mask.cpu()

            pos_ids = pos_ids.cpu()
            ner_ids = ner_ids.cpu()
            trigger_ids = trigger_ids.cpu()
            role_ids = role_ids.cpu()
            gold_exist = gold_exist.cpu()
            lengths_cpu = lengths.cpu().tolist()

            for b, n in enumerate(lengths_cpu):
                all_gold_pos.append(pos_ids[b, :n])
                all_pred_pos.append(pred_pos[b, :n])
                all_gold_ner.append(ner_ids[b, :n])
                all_pred_ner.append(pred_ner[b, :n])
                all_gold_trig.append(trigger_ids[b, :n])
                all_pred_trig.append(pred_trig[b, :n])

                gold_exist_mat = gold_exist[b, :n, :n]
                pred_exist_mat = pred_exist[b, :n, :n]
                gold_role_mat = role_ids[b, :n, :n]
                pred_role_mat = pred_role[b, :n, :n]
                valid_mask_mat = pair_mask[b, :n, :n]

                all_gold_exist.append(gold_exist_mat)
                all_pred_exist.append(pred_exist_mat)
                all_gold_exist_flat.append(gold_exist_mat[valid_mask_mat].reshape(-1))
                all_pred_exist_flat.append(pred_exist_mat[valid_mask_mat].reshape(-1))
                all_gold_role.append(gold_role_mat)
                all_pred_role.append(pred_role_mat)
                all_gold_role_flat.append(gold_role_mat[valid_mask_mat].reshape(-1))
                all_pred_role_flat.append(pred_role_mat[valid_mask_mat].reshape(-1))
                all_meta.append(batch["meta"][b])

    gold_pos = torch.cat(all_gold_pos, dim=0) if all_gold_pos else torch.tensor([], dtype=torch.long)
    pred_pos = torch.cat(all_pred_pos, dim=0) if all_pred_pos else torch.tensor([], dtype=torch.long)
    gold_ner = torch.cat(all_gold_ner, dim=0) if all_gold_ner else torch.tensor([], dtype=torch.long)
    pred_ner = torch.cat(all_pred_ner, dim=0) if all_pred_ner else torch.tensor([], dtype=torch.long)
    gold_trig = torch.cat(all_gold_trig, dim=0) if all_gold_trig else torch.tensor([], dtype=torch.long)
    pred_trig = torch.cat(all_pred_trig, dim=0) if all_pred_trig else torch.tensor([], dtype=torch.long)
    gold_exist_flat = torch.cat(all_gold_exist_flat, dim=0) if all_gold_exist_flat else torch.tensor([], dtype=torch.long)
    pred_exist_flat = torch.cat(all_pred_exist_flat, dim=0) if all_pred_exist_flat else torch.tensor([], dtype=torch.long)
    gold_role_flat = torch.cat(all_gold_role_flat, dim=0) if all_gold_role_flat else torch.tensor([], dtype=torch.long)
    pred_role_flat = torch.cat(all_pred_role_flat, dim=0) if all_pred_role_flat else torch.tensor([], dtype=torch.long)

    results = {
        "pos": compute_token_classification_metrics(
            gold_pos, pred_pos, vocabs["pos"], ignore_index=0, negative_label="X"
        ),
        "ner": compute_token_classification_metrics(
            gold_ner, pred_ner, vocabs["ner"], ignore_index=0, negative_label="O"
        ),
        "trigger_token": compute_token_classification_metrics(
            gold_trig, pred_trig, vocabs["trigger"], ignore_index=0, negative_label="O"
        ),
        "argument_existence_token": {
            "overall": compute_binary_pair_metrics(gold_exist_flat, pred_exist_flat, positive_value=1),
            "per_label": {
                "EXIST": compute_binary_pair_metrics(gold_exist_flat, pred_exist_flat, positive_value=1)
            },
        },
        "role_token": compute_token_classification_metrics(
            gold_role_flat, pred_role_flat, vocabs["role"], ignore_index=-100, negative_label="O"
        ),
        "event_level": event_level_metrics_from_lists(
            all_gold_role,
            all_pred_role,
            all_gold_trig,
            all_pred_trig,
            all_gold_exist,
            all_pred_exist,
            all_meta,
            vocabs["role"],
            vocabs["trigger"],
        ),
    }

    prediction_dump = build_prediction_dump(
        all_gold_pos=all_gold_pos,
        all_pred_pos=all_pred_pos,
        all_gold_ner=all_gold_ner,
        all_pred_ner=all_pred_ner,
        all_gold_trig=all_gold_trig,
        all_pred_trig=all_pred_trig,
        all_gold_role=all_gold_role,
        all_pred_role=all_pred_role,
        all_gold_exist=all_gold_exist,
        all_pred_exist=all_pred_exist,
        all_meta=all_meta,
        vocabs=vocabs,
    )

    return results, prediction_dump


def pretty_print(results: Dict[str, Any]) -> None:
    print("\n=== TOKEN-LEVEL METRICS ===")
    for task in ["pos", "ner", "trigger_token", "argument_existence_token", "role_token"]:
        overall = results[task]["overall"]
        print(f"\n[{task}]")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1:        {overall['f1']:.4f}")
        print(f"  TP/FP/FN:  {overall['tp']}/{overall['fp']}/{overall['fn']}")

    print("\n=== EVENT-LEVEL METRICS ===")
    for name, vals in results["event_level"].items():
        print(f"\n[{name}]")
        print(f"  Precision: {vals['precision']:.4f}")
        print(f"  Recall:    {vals['recall']:.4f}")
        print(f"  F1:        {vals['f1']:.4f}")
        print(f"  TP/FP/FN:  {vals['tp']}/{vals['fp']}/{vals['fn']}")


def load_checkpoint(ckpt_path: str, cfg: Config):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw_vocabs = ckpt["vocabs"]
    vocabs = {k: restore_vocab(v) for k, v in raw_vocabs.items()}

    print("Loaded vocab sizes:")
    for k, v in vocabs.items():
        print(f"  {k}: {vocab_size(v)}")

    model = EventExtractor(
        vocab_size=vocab_size(vocabs["word"]),
        affix_vocab_size=vocab_size(vocabs["affix"]),
        pos_label_size=vocab_size(vocabs["pos"]),
        ner_label_size=vocab_size(vocabs["ner"]),
        trigger_label_size=vocab_size(vocabs["trigger"]),
        role_label_size=vocab_size(vocabs["role"]),
        word_emb_dim=cfg.word_emb_dim,
        graph_emb_dim=cfg.graph_emb_dim,
        affix_emb_dim=cfg.affix_emb_dim,
        pos_emb_dim=cfg.pos_emb_dim,
        ner_emb_dim=cfg.ner_emb_dim,
        case_type_size=getattr(cfg, "case_type_size", 7),
        case_type_emb_dim=getattr(cfg, "case_type_emb_dim", 16),
        morph_feat_dim=cfg.morph_feat_dim,
        lstm_hidden_dim=cfg.lstm_hidden_dim,
        gcn_hidden_dim=cfg.gcn_hidden_dim,
        gcn_layers=cfg.gcn_layers,
        edge_type_size=getattr(cfg, "edge_type_size", 18),
        edge_type_emb_dim=getattr(cfg, "edge_type_emb_dim", 16),
        dropout=cfg.dropout,
        max_pair_distance=getattr(cfg, "max_pair_distance", 3),
        distance_emb_dim=getattr(cfg, "distance_emb_dim", 16),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, vocabs


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Evaluating on: {cfg.test_file}")
    model, vocabs = load_checkpoint("mered_event_model.pt", cfg)
    model = model.to(device)

    eval_ds = MEREDDataset(cfg.test_file, cfg, vocabs=vocabs, build_vocabs=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Loaded test sentence instances: {len(eval_ds)}")
    print(f"trigger_threshold={getattr(cfg, 'trigger_threshold', 0.60)}")
    print(f"arg_exist_threshold={getattr(cfg, 'arg_exist_threshold', 0.75)}")
    print(f"top_k_args_per_trigger={getattr(cfg, 'top_k_args_per_trigger', 2)}")

    results, prediction_dump = evaluate_model(model, eval_loader, vocabs, cfg, device=device)
    pretty_print(results)

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open("prediction_dump.json", "w", encoding="utf-8") as f:
        json.dump(prediction_dump, f, ensure_ascii=False, indent=2)

    print("\nSaved evaluation results to evaluation_results.json")
    print("Saved detailed predictions to prediction_dump.json")


if __name__ == "__main__":
    main()
