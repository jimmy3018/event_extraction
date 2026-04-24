
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gensim.models import FastText
from torch.utils.data import DataLoader

from config import Config
from dataset import MEREDDataset, collate_fn
from models import EventExtractor, normalize_adjacency
from utils import set_seed


def vocab_size(vocab) -> int:
    if hasattr(vocab, "itos"):
        return len(vocab.itos)
    return len(vocab)


def print_dataset_stats(train_ds: MEREDDataset, max_examples: int = 3) -> None:
    pos_counter = Counter()
    ner_counter = Counter()
    trigger_counter = Counter()
    role_counter = Counter()

    for inst in train_ds.instances:
        pos_counter.update(inst["pos_labels"])
        ner_counter.update(inst["ner_labels"])
        trigger_counter.update(inst["trigger_labels"])
        for row in inst["role_matrix"]:
            role_counter.update(row)

    print("\n=== TRAIN DATASET STATS ===")
    print(f"Documents loaded: {len(train_ds.docs)}")
    print(f"Sentence instances: {len(train_ds.instances)}")
    print("POS distribution:", pos_counter)
    print("NER distribution:", ner_counter)
    print("Trigger distribution:", trigger_counter)
    print("Role distribution:", role_counter)

    print("\n=== SAMPLE INSTANCES ===")
    for i in range(min(max_examples, len(train_ds.instances))):
        inst = train_ds.instances[i]
        positive_roles = [
            (r, c, inst["role_matrix"][r][c])
            for r in range(len(inst["role_matrix"]))
            for c in range(len(inst["role_matrix"][r]))
            if inst["role_matrix"][r][c] != "O"
        ]
        print(f"\n--- Instance {i + 1} ---")
        print("doc_id:", inst["doc_id"])
        print("sid:", inst["sid"])
        print("tokens:", inst["tokens"])
        print("pos_labels:", inst["pos_labels"])
        print("ner_labels:", inst["ner_labels"])
        print("trigger_labels:", inst["trigger_labels"])
        print("positive_roles:", positive_roles[:20])


def build_fasttext_sentence_corpus(train_ds: MEREDDataset) -> List[List[str]]:
    corpus: List[List[str]] = []
    for inst in train_ds.instances:
        tokens = inst["tokens"]
        if tokens:
            corpus.append(tokens)
    return corpus


def train_fasttext_embeddings(train_ds: MEREDDataset, vocabs, cfg: Config) -> torch.Tensor:
    corpus = build_fasttext_sentence_corpus(train_ds)
    print(f"\nSentence corpus size for FastText: {len(corpus)}")

    ft = FastText(
        sentences=corpus,
        vector_size=cfg.graph_emb_dim,
        window=getattr(cfg, "fasttext_window", 5),
        min_count=getattr(cfg, "fasttext_min_count", 1),
        workers=getattr(cfg, "fasttext_workers", 4),
        sg=1,
        epochs=getattr(cfg, "fasttext_epochs", 10),
    )

    word_vocab = vocabs["word"]
    matrix = torch.randn(vocab_size(word_vocab), cfg.graph_emb_dim) * 0.02

    pad_idx = word_vocab.stoi.get(cfg.pad_token, 0)
    matrix[pad_idx] = 0.0

    for token, idx in word_vocab.stoi.items():
        if token == cfg.pad_token:
            continue
        if token in ft.wv:
            matrix[idx] = torch.tensor(ft.wv[token], dtype=torch.float)

    print("Initialized embedding matrix from sentence-level FastText.")
    return matrix


def get_task_weights(cfg: Config):
    pos_w = getattr(cfg, "pos_loss_weight", 0.5)
    ner_w = getattr(cfg, "ner_loss_weight", 0.5)
    trig_w = getattr(cfg, "trigger_loss_weight", 2.0)
    exist_w = getattr(cfg, "arg_exist_loss_weight", 1.0)
    role_w = getattr(cfg, "role_loss_weight", 2.0)
    return pos_w, ner_w, trig_w, exist_w, role_w


def get_pos_candidate_ids(pos_vocab) -> Set[int]:
    candidate_tags = {"NOUN", "PROPN", "PRON", "NUM"}
    return {pos_vocab.stoi[tag] for tag in candidate_tags if tag in pos_vocab.stoi}


def build_argument_candidate_mask(
    pos_ids: torch.Tensor,
    ner_ids: torch.Tensor,
    trigger_ids: torch.Tensor,
    case_type_ids: torch.Tensor,
    pair_base_mask: torch.Tensor,
    pos_candidate_ids: Set[int],
    ner_o_idx: int,
    trigger_o_idx: int,
    pos_verb_ids: Set[int],
) -> torch.Tensor:
    trigger_mask = trigger_ids != trigger_o_idx

    pos_ok = torch.zeros_like(pos_ids, dtype=torch.bool)
    for idx in pos_candidate_ids:
        pos_ok |= (pos_ids == idx)

    ner_ok = ner_ids != ner_o_idx
    case_ok = case_type_ids != 0

    not_verb = torch.ones_like(pos_ids, dtype=torch.bool)
    for idx in pos_verb_ids:
        not_verb &= (pos_ids != idx)

    arg_mask = (pos_ok | ner_ok | case_ok) & not_verb & (~trigger_mask)
    pair_mask = trigger_mask.unsqueeze(2) & arg_mask.unsqueeze(1) & pair_base_mask
    return pair_mask


def masked_mean(
    loss_tensor: torch.Tensor,
    mask: torch.Tensor,
    weight_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_f = mask.float()
    if weight_tensor is not None:
        mask_f = mask_f * weight_tensor
    denom = mask_f.sum().clamp(min=1.0)
    return (loss_tensor * mask_f).sum() / denom


def masked_accuracy(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> float:
    correct = ((pred == gold) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def token_accuracy(pred_logits: torch.Tensor, gold: torch.Tensor, ignore_index: int = 0) -> float:
    pred = pred_logits.argmax(-1)
    mask = gold != ignore_index
    correct = ((pred == gold) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def save_history(history: Dict[str, List[float]], out_json: str = "training_history.json") -> None:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_curves(history: Dict[str, List[float]]) -> None:
    epochs = list(range(1, len(history["total_loss"]) + 1))

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, history["total_loss"], label="Total Loss")
    plt.plot(epochs, history["pos_loss"], label="POS Loss")
    plt.plot(epochs, history["ner_loss"], label="NER Loss")
    plt.plot(epochs, history["trigger_loss"], label="Trigger Loss")
    plt.plot(epochs, history["arg_exist_loss"], label="Arg Exist Loss")
    plt.plot(epochs, history["role_loss"], label="Role Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, history["pos_acc"], label="POS Acc")
    plt.plot(epochs, history["ner_acc"], label="NER Acc")
    plt.plot(epochs, history["trigger_acc"], label="Trigger Acc")
    plt.plot(epochs, history["arg_exist_acc"], label="Arg Exist Acc")
    plt.plot(epochs, history["role_acc"], label="Role Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_curves.png", dpi=200)
    plt.close()


def main():
    cfg = Config()
    set_seed(cfg.seed)

    print(f"Training file: {cfg.train_file}")
    print(f"Test file: {cfg.test_file}")

    train_ds = MEREDDataset(cfg.train_file, cfg, build_vocabs=True)
    vocabs = train_ds.vocabs

    print_dataset_stats(train_ds, max_examples=3)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print("\n=== VOCAB SIZES ===")
    for name, vocab in vocabs.items():
        print(f"{name}: {vocab_size(vocab)}")

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

    if getattr(cfg, "use_fasttext", False):
        embedding_matrix = train_fasttext_embeddings(train_ds, vocabs, cfg)
        if hasattr(model, "initialize_graph_embeddings"):
            model.initialize_graph_embeddings(embedding_matrix)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    role_o_idx = vocabs["role"].stoi["O"]
    ner_o_idx = vocabs["ner"].stoi["O"]
    trigger_o_idx = vocabs["trigger"].stoi["O"]

    pos_candidate_ids = get_pos_candidate_ids(vocabs["pos"])
    pos_verb_ids = {
        vocabs["pos"].stoi[tag]
        for tag in {"VERB", "AUX"}
        if tag in vocabs["pos"].stoi
    }

    base_criterion = nn.CrossEntropyLoss(ignore_index=0)

    role_weights = torch.ones(vocab_size(vocabs["role"]), dtype=torch.float)
    role_weights[role_o_idx] = getattr(cfg, "role_o_weight", 0.1)
    role_weights = role_weights.to(device)
    role_criterion = nn.CrossEntropyLoss(weight=role_weights, reduction="none")

    exist_weights = torch.ones(2, dtype=torch.float)
    exist_weights[0] = getattr(cfg, "arg_nonexist_weight", 0.35)
    exist_weights[1] = getattr(cfg, "arg_exist_weight", 1.0)
    exist_weights = exist_weights.to(device)
    exist_criterion = nn.CrossEntropyLoss(weight=exist_weights, reduction="none")

    pos_w, ner_w, trig_w, exist_w, role_w = get_task_weights(cfg)

    history = {
        "total_loss": [],
        "pos_loss": [],
        "ner_loss": [],
        "trigger_loss": [],
        "arg_exist_loss": [],
        "role_loss": [],
        "pos_acc": [],
        "ner_acc": [],
        "trigger_acc": [],
        "arg_exist_acc": [],
        "role_acc": [],
    }

    best_loss = float("inf")
    epochs_without_improvement = 0
    neg_sample_rate = getattr(cfg, "arg_negative_sample_rate", 0.05)

    for epoch in range(cfg.max_epochs):
        model.train()

        total_loss = 0.0
        total_pos_loss = 0.0
        total_ner_loss = 0.0
        total_trig_loss = 0.0
        total_exist_loss = 0.0
        total_role_loss = 0.0

        total_pos_acc = 0.0
        total_ner_acc = 0.0
        total_trig_acc = 0.0
        total_exist_acc = 0.0
        total_role_acc = 0.0

        for batch in train_loader:
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
                pos_ids=pos_ids,
                ner_ids=ner_ids,
                use_gold_tags=True,
            )

            pos_loss = base_criterion(
                out["pos_logits"].reshape(-1, out["pos_logits"].size(-1)),
                pos_ids.reshape(-1),
            )
            ner_loss = base_criterion(
                out["ner_logits"].reshape(-1, out["ner_logits"].size(-1)),
                ner_ids.reshape(-1),
            )
            trig_loss = base_criterion(
                out["trigger_logits"].reshape(-1, out["trigger_logits"].size(-1)),
                trigger_ids.reshape(-1),
            )

            gold_exist = (role_ids != role_o_idx).long()

            candidate_mask = build_argument_candidate_mask(
                pos_ids=pos_ids,
                ner_ids=ner_ids,
                trigger_ids=trigger_ids,
                case_type_ids=case_type_ids,
                pair_base_mask=out["pair_base_mask"],
                pos_candidate_ids=pos_candidate_ids,
                ner_o_idx=ner_o_idx,
                trigger_o_idx=trigger_o_idx,
                pos_verb_ids=pos_verb_ids,
            )

            pos_mask = (gold_exist == 1) & candidate_mask
            neg_mask = (gold_exist == 0) & candidate_mask
            sampled_neg_mask = neg_mask & (torch.rand_like(gold_exist.float()) < neg_sample_rate)
            exist_train_mask = pos_mask | sampled_neg_mask

            exist_loss_tensor = exist_criterion(
                out["arg_exist_logits"].reshape(-1, out["arg_exist_logits"].size(-1)),
                gold_exist.reshape(-1),
            ).reshape_as(gold_exist)
            exist_loss = masked_mean(exist_loss_tensor, exist_train_mask)

            positive_role_mask = role_ids != role_o_idx
            role_train_mask = candidate_mask & positive_role_mask

            salience_weights = torch.ones_like(role_ids, dtype=torch.float, device=device)
            for b, meta in enumerate(batch["meta"]):
                for arg_info in meta.get("argument_spans", []):
                    trig_head = arg_info.get("trigger_head", None)
                    arg_span = arg_info.get("arg_span", None)
                    salience = float(arg_info.get("salience", 0.0))
                    if trig_head is None or arg_span is None or len(arg_span) != 2:
                        continue
                    a_start = arg_span[0]
                    if trig_head < role_ids.size(1) and a_start < role_ids.size(2):
                        salience_weights[b, trig_head, a_start] = 1.0 + salience

            role_loss_tensor = role_criterion(
                out["role_logits"].reshape(-1, out["role_logits"].size(-1)),
                role_ids.reshape(-1),
            ).reshape_as(role_ids)
            role_loss = masked_mean(role_loss_tensor, role_train_mask, salience_weights)

            weighted_pos_loss = pos_w * pos_loss
            weighted_ner_loss = ner_w * ner_loss
            weighted_trig_loss = trig_w * trig_loss
            weighted_exist_loss = exist_w * exist_loss
            weighted_role_loss = role_w * role_loss
            loss = (
                weighted_pos_loss
                + weighted_ner_loss
                + weighted_trig_loss
                + weighted_exist_loss
                + weighted_role_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_ner_loss += ner_loss.item()
            total_trig_loss += trig_loss.item()
            total_exist_loss += exist_loss.item()
            total_role_loss += role_loss.item()

            total_pos_acc += token_accuracy(out["pos_logits"], pos_ids, ignore_index=0)
            total_ner_acc += token_accuracy(out["ner_logits"], ner_ids, ignore_index=0)
            total_trig_acc += token_accuracy(out["trigger_logits"], trigger_ids, ignore_index=0)

            pred_exist = out["arg_exist_logits"].argmax(-1)
            total_exist_acc += masked_accuracy(pred_exist, gold_exist, exist_train_mask)

            pred_role = out["role_logits"].argmax(-1)
            total_role_acc += masked_accuracy(pred_role, role_ids, role_train_mask)

        num_batches = max(len(train_loader), 1)
        avg_loss = total_loss / num_batches
        avg_pos_loss = total_pos_loss / num_batches
        avg_ner_loss = total_ner_loss / num_batches
        avg_trig_loss = total_trig_loss / num_batches
        avg_exist_loss = total_exist_loss / num_batches
        avg_role_loss = total_role_loss / num_batches

        avg_pos_acc = total_pos_acc / num_batches
        avg_ner_acc = total_ner_acc / num_batches
        avg_trig_acc = total_trig_acc / num_batches
        avg_exist_acc = total_exist_acc / num_batches
        avg_role_acc = total_role_acc / num_batches

        history["total_loss"].append(avg_loss)
        history["pos_loss"].append(avg_pos_loss)
        history["ner_loss"].append(avg_ner_loss)
        history["trigger_loss"].append(avg_trig_loss)
        history["arg_exist_loss"].append(avg_exist_loss)
        history["role_loss"].append(avg_role_loss)
        history["pos_acc"].append(avg_pos_acc)
        history["ner_acc"].append(avg_ner_acc)
        history["trigger_acc"].append(avg_trig_acc)
        history["arg_exist_acc"].append(avg_exist_acc)
        history["role_acc"].append(avg_role_acc)

        print(
            f"\nEpoch {epoch + 1}/{cfg.max_epochs}"
            f"\n  total_loss:      {avg_loss:.6f}"
            f"\n  pos_loss:        {avg_pos_loss:.6f} | pos_acc:        {avg_pos_acc:.4f}"
            f"\n  ner_loss:        {avg_ner_loss:.6f} | ner_acc:        {avg_ner_acc:.4f}"
            f"\n  trigger_loss:    {avg_trig_loss:.6f} | trigger_acc:    {avg_trig_acc:.4f}"
            f"\n  arg_exist_loss:  {avg_exist_loss:.6f} | arg_exist_acc:  {avg_exist_acc:.4f}"
            f"\n  role_loss:       {avg_role_loss:.6f} | role_acc:       {avg_role_acc:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "train_file": cfg.train_file,
            "test_file": cfg.test_file,
            "vocabs": {
                "word": {"stoi": vocabs["word"].stoi, "itos": vocabs["word"].itos},
                "affix": {"stoi": vocabs["affix"].stoi, "itos": vocabs["affix"].itos},
                "pos": {"stoi": vocabs["pos"].stoi, "itos": vocabs["pos"].itos},
                "ner": {"stoi": vocabs["ner"].stoi, "itos": vocabs["ner"].itos},
                "trigger": {"stoi": vocabs["trigger"].stoi, "itos": vocabs["trigger"].itos},
                "role": {"stoi": vocabs["role"].stoi, "itos": vocabs["role"].itos},
            },
            "config": cfg.__dict__,
            "epoch": epoch + 1,
            "avg_train_loss": avg_loss,
            "history": history,
        }

        torch.save(checkpoint, "mered_event_model_last.pt")

        save_history(history)
        plot_curves(history)

        improvement = best_loss - avg_loss
        if improvement > cfg.min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, "mered_event_model.pt")
            print(f"  Saved new best model to mered_event_model.pt (best_loss={best_loss:.6f})")
        else:
            epochs_without_improvement += 1
            print(
                f"  No significant improvement. "
                f"patience={epochs_without_improvement}/{cfg.early_stopping_patience}"
            )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print("\nEarly stopping triggered.")
            print(f"Best loss: {best_loss:.6f}")
            break

    print("\nTraining complete.")
    print("Best model saved to mered_event_model.pt")
    print("Last epoch model saved to mered_event_model_last.pt")
    print("Training history saved to training_history.json")
    print("Loss curves saved to loss_curves.png")
    print("Accuracy curves saved to accuracy_curves.png")


if __name__ == "__main__":
    main()
