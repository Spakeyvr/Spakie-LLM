import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


_CFG = yaml.safe_load((Path(__file__).parent / "default.yaml").read_text())
_D = _CFG["defaults"]

DEFAULT_PRESET: str = _CFG["default_preset"]
SUPPORTED_PRESETS: tuple[str, ...] = tuple(_CFG["supported_presets"])
CHAT_SYSTEM_PROMPT: str = _CFG["chat_system_prompt"]
DEFAULT_TARGET_TRAIN_TOKENS: int = _D["target_train_tokens"]
CORPUS_SOURCE_ALIASES: dict[str, str] = _CFG["corpus_source_aliases"]


def default_corpus_source_plan() -> dict[str, dict[str, int | str | bool]]:
    return {k: dict(v) for k, v in _CFG["corpus_source_plan"].items()}


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
    vocab_size: int = _D["vocab_size"]
    n_layers: int = _D["n_layers"]
    n_heads: int = _D["n_heads"]
    d_model: int = _D["d_model"]
    d_ff: int = _D["d_ff"]
    max_seq_len: int = _D["max_seq_len"]
    dropout: float = _D["dropout"]
    bias: bool = _D["bias"]
    activation_checkpointing: bool = _D["activation_checkpointing"]

    # Pretraining
    pretrain_batch_size: int = _D["pretrain_batch_size"]
    pretrain_grad_accum_steps: int = _D["pretrain_grad_accum_steps"]
    pretrain_lr: float = _D["pretrain_lr"]
    pretrain_target_tokens: int = _D["pretrain_target_tokens"]
    pretrain_max_steps: int = _D["pretrain_max_steps"]
    pretrain_warmup_steps: int = _D["pretrain_warmup_steps"]
    pretrain_weight_decay: float = _D["pretrain_weight_decay"]
    pretrain_grad_clip: float = _D["pretrain_grad_clip"]
    pretrain_eval_interval: int = _D["pretrain_eval_interval"]
    pretrain_eval_batches: int = _D["pretrain_eval_batches"]
    pretrain_patience: int = _D["pretrain_patience"]
    pretrain_optimizer: str = _D["pretrain_optimizer"]

    # SFT
    sft_batch_size: int = _D["sft_batch_size"]
    sft_grad_accum_steps: int = _D["sft_grad_accum_steps"]
    sft_lr: float = _D["sft_lr"]
    sft_epochs: int = _D["sft_epochs"]
    sft_weight_decay: float = _D["sft_weight_decay"]
    sft_grad_clip: float = _D["sft_grad_clip"]
    sft_patience: int = _D["sft_patience"]
    sft_download_max_examples: int = _D["sft_download_max_examples"]
    sft_optimizer: str = _D["sft_optimizer"]

    # Optimizer
    allow_adamw_fallback: bool = _D["allow_adamw_fallback"]
    muon_momentum: float = _D["muon_momentum"]
    muon_nesterov: bool = _D["muon_nesterov"]
    muon_ns_steps: int = _D["muon_ns_steps"]
    muon_ns_coefficients: tuple[float, float, float] = tuple(_D["muon_ns_coefficients"])
    muon_eps: float = _D["muon_eps"]
    muon_adjust_lr_fn: str = _D["muon_adjust_lr_fn"]
    muon_qkv_split: bool = _D["muon_qkv_split"]
    muon_verified: bool = _D["muon_verified"]

    # Generation
    temperature: float = _D["temperature"]
    top_k: int = _D["top_k"]
    top_p: float = _D["top_p"]

    # Paths
    raw_data_dir: str = _D["raw_data_dir"]
    processed_data_dir: str = _D["processed_data_dir"]
    chat_data_dir: str = _D["chat_data_dir"]
    chat_raw_dir: str = _D["chat_raw_dir"]
    eval_data_dir: str = _D["eval_data_dir"]
    tokenizer_prefix: str = _D["tokenizer_prefix"]
    checkpoint_root_dir: str = _D["checkpoint_root_dir"]
    checkpoint_dir: str = _D["checkpoint_dir"]

    # Corpus planning
    target_train_tokens: int = _D["target_train_tokens"]
    target_processed_tokens: int = _D["target_processed_tokens"]
    large_corpus_dir: str = _D["large_corpus_dir"]
    corpus_report_path: str = _D["corpus_report_path"]
    token_shard_dir: str = _D["token_shard_dir"]
    token_shard_size: int = _D["token_shard_size"]
    min_doc_chars: int = _D["min_doc_chars"]
    source_min_doc_chars: dict[str, int] = field(
        default_factory=lambda: dict(_CFG["source_min_doc_chars"])
    )
    max_repeated_line_ratio: float = _D["max_repeated_line_ratio"]
    max_noise_ratio: float = _D["max_noise_ratio"]
    mean_word_length_min: float = _D["mean_word_length_min"]
    mean_word_length_max: float = _D["mean_word_length_max"]
    min_stopword_count: int = _D["min_stopword_count"]
    max_symbol_word_ratio: float = _D["max_symbol_word_ratio"]
    max_top_2gram_char_share: float = _D["max_top_2gram_char_share"]
    max_top_3gram_char_share: float = _D["max_top_3gram_char_share"]
    max_dup_5gram_char_share: float = _D["max_dup_5gram_char_share"]
    max_top_char_share: float = _D["max_top_char_share"]
    max_url_email_line_ratio: float = _D["max_url_email_line_ratio"]
    train_split_fraction: float = _D["train_split_fraction"]
    estimated_chars_per_token: float = _D["estimated_chars_per_token"]

    # Near-duplicate detection (MinHash + LSH). Catches paraphrased copies and
    # near-identical mirrors that exact-hash dedup misses.
    near_dup_jaccard_threshold: float = _D["near_dup_jaccard_threshold"]
    near_dup_num_perm: int = _D["near_dup_num_perm"]
    near_dup_shingle_size: int = _D["near_dup_shingle_size"]

    # fastText language identification (replaces ASCII-ratio heuristic).
    # The lid.176 model is downloaded on first use and cached at this path.
    langid_min_confidence: float = _D["langid_min_confidence"]
    langid_model_path: str = _D["langid_model_path"]
    langid_model_url: str = _D["langid_model_url"]
    corpus_source_plan: dict[str, dict[str, int | str | bool]] = field(default_factory=default_corpus_source_plan)
    sft_source_limits: dict[str, int] = field(
        default_factory=lambda: dict(_CFG["sft_source_limits"])
    )

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

    for key, value in _CFG["presets"].get(preset, {}).items():
        if key == "muon_ns_coefficients":
            value = tuple(value)
        setattr(config, key, value)

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
