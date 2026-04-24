from dataclasses import dataclass


@dataclass
class Config:
    # =========================
    # DATA
    # =========================
    train_file: str = "train.json"
    test_file: str = "test.json"

    max_seq_len: int = 128
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    # Disable fasttext for speed
    use_fasttext: bool = False

    # =========================
    # MODEL (balanced)
    # =========================
    word_emb_dim: int = 128
    graph_emb_dim: int = 64
    affix_emb_dim: int = 32

    pos_emb_dim: int = 32
    ner_emb_dim: int = 32

    morph_feat_dim: int = 8

    lstm_hidden_dim: int = 128
    gcn_hidden_dim: int = 128
    gcn_layers: int = 2

    dropout: float = 0.3

    # IMPORTANT (must match training checkpoint!)
    max_pair_distance: int = 3
    distance_emb_dim: int = 16

    case_type_size: int = 7
    case_type_emb_dim: int = 16

    # =========================
    # TRAINING (fast + stable)
    # =========================
    batch_size: int = 8   # ↑ faster, still stable
    max_epochs: int = 30  # ↓ no need 100
    lr: float = 8e-4      # slightly lower = stable
    weight_decay: float = 1e-5

    seed: int = 42

    early_stopping_patience: int = 5
    min_delta: float = 1e-4

    # =========================
    # GRAPH / AUGMENTATION
    # =========================
    random_walks_per_start: int = 2   # ↓ faster
    random_walk_length: int = 4

    fasttext_window: int = 3
    fasttext_epochs: int = 10
    fasttext_min_count: int = 1
    fasttext_workers: int = 2

    # =========================
    # LOSS BALANCING (CRITICAL)
    # =========================
    pos_loss_weight: float = 0.4
    ner_loss_weight: float = 0.4

    # 🔥 BOOST trigger learning
    trigger_loss_weight: float = 3.0

    # 🔥 improve argument detection
    arg_exist_loss_weight: float = 1.5
    role_loss_weight: float = 3.0

    # reduce "O" dominance
    role_o_weight: float = 0.05

    # existence balancing
    arg_nonexist_weight: float = 0.3
    arg_exist_weight: float = 1.5

    # negative sampling (IMPORTANT)
    arg_negative_sample_rate: float = 0.02

    # =========================
    # INFERENCE TUNING (CRITICAL)
    # =========================
    arg_exist_threshold: float = 0.60