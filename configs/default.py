import math
import os
from dataclasses import dataclass
from dataclasses import field


DEFAULT_PRESET = "300m"
SUPPORTED_PRESETS = ("92m", "180m", "300m")
DEFAULT_TARGET_TRAIN_TOKENS = 2_000_000_000
CORPUS_SOURCE_ALIASES = {
    "c4": "c4_en",
    "cosmopedia": "cosmopedia_v2",
    "cosmopedia-v2": "cosmopedia_v2",
    "fineweb": "fineweb_sample",
    "fineweb-sample": "fineweb_sample",
    "open-web-math": "openwebmath",
    "open_web_math": "openwebmath",
    "wikipedia": "wikipedia_snapshot",
}


def default_corpus_source_plan() -> dict[str, dict[str, int | str | bool]]:
    return {
        "fineweb-edu": {
            "kind": "web",
            "target_tokens": 750_000_000,
            "target_raw_chars": 3_000_000_000,
            "enabled": True,
        },
        "dolma": {
            "kind": "web",
            "target_tokens": 450_000_000,
            "target_raw_chars": 1_800_000_000,
            "enabled": False,
        },
        "refinedweb": {
            "kind": "web",
            "target_tokens": 125_000_000,
            "target_raw_chars": 500_000_000,
            "enabled": True,
        },
        "fineweb_sample": {
            "kind": "web",
            "target_tokens": 300_000_000,
            "target_raw_chars": 1_200_000_000,
            "enabled": True,
        },
        "c4_en": {
            "kind": "web",
            "target_tokens": 125_000_000,
            "target_raw_chars": 500_000_000,
            "enabled": True,
        },
        "gutenberg": {
            "kind": "books",
            "target_tokens": 450_000_000,
            "target_raw_chars": 1_800_000_000,
            "enabled": True,
        },
        "wikipedia_snapshot": {
            "kind": "reference",
            "target_tokens": 150_000_000,
            "target_raw_chars": 600_000_000,
            "enabled": True,
        },
        "stackexchange": {
            "kind": "technical",
            "target_tokens": 200_000_000,
            "target_raw_chars": 800_000_000,
            "enabled": True,
        },
        "openwebmath": {
            "kind": "technical",
            "target_tokens": 175_000_000,
            "target_raw_chars": 700_000_000,
            "enabled": True,
        },
        "arxiv": {
            "kind": "technical",
            "target_tokens": 55_263_158,
            "target_raw_chars": 221_052_632,
            "enabled": True,
        },
        "cosmopedia_v2": {
            "kind": "synthetic_education",
            "target_tokens": 450_000_000,
            "target_raw_chars": 1_800_000_000,
            "enabled": True,
        },
    }


def normalize_corpus_source(source_name: str) -> str:
    source = (source_name or "").strip().lower()
    return CORPUS_SOURCE_ALIASES.get(source, source)


def derive_processed_token_target(target_train_tokens: int, train_split_fraction: float) -> int:
    if target_train_tokens <= 0:
        return 0
    if not 0 < train_split_fraction < 1:
        raise ValueError("train_split_fraction must be between 0 and 1")
    return math.ceil(target_train_tokens / train_split_fraction)


def derive_pretrain_max_steps(target_tokens: int, tokens_per_step: int) -> int:
    if target_tokens <= 0 or tokens_per_step <= 0:
        return 0
    return math.ceil(target_tokens / tokens_per_step)


@dataclass
class SpakieConfig:
    preset_name: str = DEFAULT_PRESET

    # Model
    vocab_size: int = 8192
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False
    activation_checkpointing: bool = False

    # Pretraining
    pretrain_batch_size: int = 128
    pretrain_grad_accum_steps: int = 2
    pretrain_lr: float = 6e-4  # scaled for 8x larger effective batch vs original 16×4
    pretrain_target_tokens: int = 0
    pretrain_max_steps: int = 0
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
    sft_download_max_examples: int = 72_000

    # Generation
    temperature: float = 0.1
    top_k: int = 1
    top_p: float = 1.0

    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    chat_data_dir: str = "data/chat"
    eval_data_dir: str = "data/eval"
    tokenizer_prefix: str = "tokenizer/spakie"
    checkpoint_root_dir: str = "checkpoints"
    checkpoint_dir: str = ""

    # Corpus planning
    target_train_tokens: int = DEFAULT_TARGET_TRAIN_TOKENS
    target_processed_tokens: int = 0
    large_corpus_dir: str = "data/raw/large_corpus"
    corpus_report_path: str = "data/processed/corpus_report.json"
    token_shard_dir: str = "data/processed/shards"
    token_shard_size: int = 5_000_000
    min_doc_chars: int = 400
    source_min_doc_chars: dict[str, int] = field(default_factory=lambda: {
        "wikipedia_snapshot": 200,
        "wikipedia": 200,
        "arxiv": 220,
        "stackexchange": 180,
        "openwebmath": 180,
        "open_corpus": 180,
        "dictionary": 80,
    })
    max_repeated_line_ratio: float = 0.25
    max_noise_ratio: float = 0.35
    train_split_fraction: float = 0.95
    estimated_chars_per_token: float = 4.0
    corpus_source_plan: dict[str, dict[str, int | str | bool]] = field(default_factory=default_corpus_source_plan)
    sft_source_limits: dict[str, int] = field(default_factory=lambda: {
        "alpaca": 16_000,
        "dolly": 10_000,
        "squad": 12_000,
        "sciq": 8_000,
        "boolq": 8_000,
        "arc_easy": 8_000,
        "arc_challenge": 5_000,
        "openbookqa": 5_000,
    })

    def __post_init__(self):
        self.preset_name = normalize_preset_name(self.preset_name)
        self.corpus_source_plan = {
            normalize_corpus_source(source_name): dict(plan)
            for source_name, plan in self.corpus_source_plan.items()
        }
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.checkpoint_root_dir, self.preset_name)
        self.refresh_derived_fields()

    def refresh_derived_fields(self) -> None:
        self.target_processed_tokens = derive_processed_token_target(self.target_train_tokens, self.train_split_fraction)
        if self.pretrain_target_tokens <= 0:
            self.pretrain_target_tokens = self.target_train_tokens
        self.pretrain_max_steps = derive_pretrain_max_steps(self.pretrain_target_tokens, self.pretrain_tokens_per_step())

    def pretrain_tokens_per_step(self) -> int:
        return self.pretrain_batch_size * self.pretrain_grad_accum_steps * self.max_seq_len

    def scaled_corpus_source_plan(
        self,
        *,
        target_processed_tokens: int | None = None,
        requested_sources: list[str] | None = None,
    ) -> dict[str, dict[str, int | str | bool]]:
        selected_sources = [
            normalize_corpus_source(source_name)
            for source_name in (requested_sources or [])
        ]
        plan = {
            source_name: dict(entry)
            for source_name, entry in self.corpus_source_plan.items()
            if entry.get("enabled", True)
        }
        if selected_sources:
            plan = {source_name: plan[source_name] for source_name in selected_sources if source_name in plan}
        if not plan:
            return {}

        target_total = target_processed_tokens or self.target_processed_tokens
        base_total = sum(int(entry["target_tokens"]) for entry in plan.values())
        if base_total <= 0:
            return plan

        scale = target_total / base_total
        scaled_plan: dict[str, dict[str, int | str | bool]] = {}
        source_names = list(plan.keys())
        remaining_tokens = target_total
        base_char_total = sum(int(entry["target_raw_chars"]) for entry in plan.values())
        remaining_chars = max(1, round(base_char_total * scale))
        for index, source_name in enumerate(source_names):
            entry = dict(plan[source_name])
            if index == len(source_names) - 1:
                scaled_tokens = remaining_tokens
            else:
                scaled_tokens = max(1, round(int(entry["target_tokens"]) * scale))
                scaled_tokens = min(scaled_tokens, remaining_tokens)
            base_chars = int(entry["target_raw_chars"])
            if index == len(source_names) - 1:
                scaled_chars = max(base_chars, round(base_chars * scale), remaining_chars)
            else:
                scaled_chars = max(1, round(base_chars * scale))
                scaled_chars = min(scaled_chars, remaining_chars)
            remaining_tokens -= scaled_tokens
            remaining_chars -= scaled_chars
            entry["target_tokens"] = int(scaled_tokens)
            entry["target_raw_chars"] = int(scaled_chars)
            scaled_plan[source_name] = entry
        return scaled_plan

    def should_use_pretrain_early_stopping(self) -> bool:
        return self.pretrain_patience > 0 and self.pretrain_target_tokens <= 0


def normalize_preset_name(preset_name: str) -> str:
    preset = (preset_name or DEFAULT_PRESET).lower()
    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported preset '{preset_name}'. Available presets: {', '.join(SUPPORTED_PRESETS)}")
    return preset


def get_preset_config(preset_name: str = DEFAULT_PRESET) -> SpakieConfig:
    preset = normalize_preset_name(preset_name)
    config = SpakieConfig(preset_name=preset)

    if preset == "92m":
        config.pretrain_batch_size = 64
        config.pretrain_grad_accum_steps = 1
        config.refresh_derived_fields()

    elif preset == "180m":
        config.n_layers = 16
        config.d_model = 896
        config.n_heads = 14
        config.d_ff = 3584
        config.activation_checkpointing = True
        config.pretrain_batch_size = 2
        config.pretrain_grad_accum_steps = 16
        config.sft_batch_size = 2
        config.sft_grad_accum_steps = 4
        config.refresh_derived_fields()

    elif preset == "300m":
        config.n_layers = 24
        config.d_model = 1024
        config.n_heads = 16
        config.d_ff = 4096
        # Activation checkpointing trades a small amount of recompute for ~30%
        # activation memory, which lets us double pretrain_batch_size and halve
        # grad-accum while holding tokens/step constant (64 * 2 * 512 = 65_536,
        # identical to the previous 32 * 4 * 512). Halving the accum loop halves
        # per-step Python overhead, which compounds with mx.compile wins.
        config.activation_checkpointing = True
        config.pretrain_batch_size = 64
        config.pretrain_grad_accum_steps = 2
        config.pretrain_lr = 6e-4
        config.sft_batch_size = 16
        config.sft_grad_accum_steps = 2
        config.refresh_derived_fields()

    return config


def checkpoint_search_dirs(config: SpakieConfig) -> list[str]:
    dirs = [config.checkpoint_dir]
    # Smoke-run outputs live under subdirs of the main checkpoint dir. Include
    # them as fallbacks so chat.py can still find *something* when the only
    # checkpoints around are from a --smoke run (placed after real dirs so real
    # checkpoints with the same filename always win).
    for subdir in ("smoke_pretrain", "smoke_sft"):
        smoke_dir = os.path.join(config.checkpoint_dir, subdir)
        if smoke_dir not in dirs:
            dirs.append(smoke_dir)
    if config.preset_name == DEFAULT_PRESET:
        legacy_dir = config.checkpoint_root_dir
        if legacy_dir not in dirs:
            dirs.append(legacy_dir)
    return dirs


def inherit_model_shape(config: SpakieConfig, checkpoint_config) -> SpakieConfig:
    for field_name in (
        "vocab_size",
        "n_layers",
        "n_heads",
        "d_model",
        "d_ff",
        "max_seq_len",
        "dropout",
        "bias",
        "activation_checkpointing",
    ):
        if hasattr(checkpoint_config, field_name):
            setattr(config, field_name, getattr(checkpoint_config, field_name))
    return config
