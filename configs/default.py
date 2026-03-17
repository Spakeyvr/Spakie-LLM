from dataclasses import dataclass
from dataclasses import field


@dataclass
class SpakieConfig:
    # Model
    vocab_size: int = 8192
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False

    # Pretraining
    pretrain_batch_size: int = 16
    pretrain_grad_accum_steps: int = 4
    pretrain_lr: float = 3e-4
    pretrain_max_steps: int = 10_000
    pretrain_warmup_steps: int = 200
    pretrain_weight_decay: float = 0.1
    pretrain_grad_clip: float = 1.0
    pretrain_eval_interval: int = 250
    pretrain_eval_batches: int = 20
    pretrain_patience: int = 20

    # SFT
    sft_batch_size: int = 8
    sft_grad_accum_steps: int = 2
    sft_lr: float = 1e-5
    sft_epochs: int = 3
    sft_weight_decay: float = 0.1
    sft_grad_clip: float = 1.0
    sft_patience: int = 5
    sft_download_max_examples: int = 26_000

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    chat_data_dir: str = "data/chat"
    tokenizer_prefix: str = "tokenizer/spakie"
    checkpoint_dir: str = "checkpoints"

    # Corpus planning
    target_processed_tokens: int = 1_000_000_000
    large_corpus_dir: str = "data/raw/large_corpus"
    corpus_report_path: str = "data/processed/corpus_report.json"
    token_shard_dir: str = "data/processed/shards"
    token_shard_size: int = 5_000_000
    min_doc_chars: int = 400
    max_repeated_line_ratio: float = 0.25
    max_noise_ratio: float = 0.35
    train_split_fraction: float = 0.95
    estimated_chars_per_token: float = 4.0
    corpus_source_mix: dict[str, float] = field(default_factory=lambda: {
        "web": 0.75,
        "books": 0.10,
        "reference": 0.08,
        "technical": 0.07,
    })
    corpus_raw_char_budgets: dict[str, int] = field(default_factory=lambda: {
        "fineweb-edu": 3_400_000_000,
        "gutenberg": 500_000_000,
        "wikipedia": 400_000_000,
        "stackexchange": 180_000_000,
        "arxiv": 70_000_000,
    })
    corpus_keep_expectations: dict[str, float] = field(default_factory=lambda: {
        "fineweb-edu": 0.60,
        "gutenberg": 0.90,
        "wikipedia": 0.95,
        "stackexchange": 0.85,
        "arxiv": 0.95,
    })
    corpus_source_token_caps: dict[str, int] = field(default_factory=lambda: {
        "fineweb-edu": 820_000_000,
        "gutenberg": 160_000_000,
        "wikipedia": 120_000_000,
        "stackexchange": 70_000_000,
        "arxiv": 40_000_000,
    })
