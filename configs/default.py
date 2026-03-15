from dataclasses import dataclass


@dataclass
class SpakieConfig:
    # Model
    vocab_size: int = 8192
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False

    # Pretraining
    pretrain_batch_size: int = 32
    pretrain_grad_accum_steps: int = 4
    pretrain_lr: float = 3e-4
    pretrain_max_steps: int = 10_000
    pretrain_warmup_steps: int = 200
    pretrain_weight_decay: float = 0.1
    pretrain_grad_clip: float = 1.0
    pretrain_eval_interval: int = 250
    pretrain_eval_batches: int = 20
    pretrain_patience: int = 5

    # SFT
    sft_batch_size: int = 8
    sft_grad_accum_steps: int = 2
    sft_lr: float = 1e-5
    sft_epochs: int = 3
    sft_weight_decay: float = 0.1
    sft_grad_clip: float = 1.0
    sft_patience: int = 5

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
