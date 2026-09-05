import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


_CFG = yaml.safe_load((Path(__file__).parent / "default.yaml").read_text())
_D = _CFG["defaults"]

DEFAULT_PRESET: str = _CFG["default_preset"]
SUPPORTED_PRESETS: tuple[str, ...] = tuple(_CFG["supported_presets"])
CHAT_SYSTEM_PROMPT: str = _CFG["chat_system_prompt"]
CORPUS_SOURCE_ALIASES: dict[str, str] = _CFG["corpus_source_aliases"]


def default_corpus_source_plan() -> dict[str, dict[str, int | str | bool]]:
    return {k: dict(v) for k, v in _CFG["corpus_source_plan"].items()}


def default_sft_source_limits() -> dict[str, int | dict[str, int | bool]]:
    limits: dict[str, int | dict[str, int | bool]] = {}
    for source_name, entry in _CFG["sft_source_limits"].items():
        limits[source_name] = dict(entry) if isinstance(entry, dict) else entry
    return limits


def default_filter_profiles() -> dict[str, dict[str, float | bool]]:
    return {
        profile_name: dict(profile)
        for profile_name, profile in _CFG["filter_profiles"].items()
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
    vocab_size: int = _D["vocab_size"]
    n_layers: int = _D["n_layers"]
    n_heads: int = _D["n_heads"]
    n_kv_heads: int = _D.get("n_kv_heads", 0)
    d_model: int = _D["d_model"]
    d_ff: int = _D["d_ff"]
    mlp_type: str = _D["mlp_type"]
    norm_type: str = _D.get("norm_type", "layernorm")
    # RMSNorm applied to per-head Q and K before attention (Qwen3/Gemma2 style).
    # Stabilizes attention logits — especially valuable under Muon, which is more
    # prone to attention-logit growth than AdamW.
    qk_norm: bool = _D.get("qk_norm", False)
    # RoPE rotates Q/K inside attention and allocates no position table.
    position_encoding: str = _D["position_encoding"]
    rope_theta: float = _D.get("rope_theta", 100000.0)
    loss_layout: str = _D.get("loss_layout", "flat")
    residual_type: str = _D.get("residual_type", "serial")
    attention_backend: str = _D.get("attention_backend", "sdpa")
    swiglu_hidden: int = _D.get("swiglu_hidden", 0)
    max_seq_len: int = _D["max_seq_len"]
    dropout: float = _D["dropout"]
    bias: bool = _D["bias"]
    activation_checkpointing: bool = _D["activation_checkpointing"]
    mlp_checkpointing: bool = _D.get("mlp_checkpointing", False)
    compact_valid_mlp: bool = _D.get("compact_valid_mlp", False)
    compact_valid_projections: bool = _D.get("compact_valid_projections", False)
    addmm_residual_projections: bool = _D.get("addmm_residual_projections", False)
    mlp_addmm_linears: bool = _D.get("mlp_addmm_linears", False)
    fused_residual_rmsnorm: bool = _D.get("fused_residual_rmsnorm", False)
    fused_cross_entropy: bool = _D.get("fused_cross_entropy", False)
    grouped_muon: bool = _D.get("grouped_muon", False)
    compile_muon_ns: bool = _D.get("compile_muon_ns", False)
    muon_route: str = _D.get("muon_route", "all")
    contiguous_linear_inputs: bool = _D.get("contiguous_linear_inputs", False)
    # When > 0 and < B*T, compute logits + cross-entropy in chunks of this many
    # tokens during training, avoiding the (B*T, vocab_size) materialization.
    # 0 disables chunking. Only used in the MLX backend's training path.
    loss_chunk_size: int = _D.get("loss_chunk_size", 0)

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
    pretrain_checkpoint_interval: int = _D.get("pretrain_checkpoint_interval", 0)
    pretrain_patience: int = _D["pretrain_patience"]
    pretrain_optimizer: str = _D["pretrain_optimizer"]
    pretrain_lr_schedule: str = _D.get("pretrain_lr_schedule", "cosine")
    pretrain_trapezoid_decay_frac: float = _D.get("pretrain_trapezoid_decay_frac", 0.2)
    pretrain_vmap_accum_step: bool = _D.get("pretrain_vmap_accum_step", False)
    pretrain_vmap_sync_warmup_steps: int = _D.get("pretrain_vmap_sync_warmup_steps", 0)
    # Number of microbatches vmapped together per group. vmap keeps every lane's
    # forward activations resident for the backward, so peak memory scales with
    # the group size; a full-G vmap of the 300m preset (B64/G3 ~= 107 GB) exceeds
    # a 128 GB machine and panics the macOS kernel. 0 = auto: at runtime the
    # loop probes per-lane memory and picks the largest group that fits a safe
    # fraction of physical RAM. A positive value forces that group size.
    pretrain_vmap_group_size: int = _D.get("pretrain_vmap_group_size", 0)
    # Fraction of physical RAM the auto group-size probe is allowed to budget for
    # the vmap forward/backward peak. Conservative because macOS panics (not
    # OOM-errors) when wired GPU memory exhausts unified memory.
    pretrain_vmap_mem_budget_frac: float = _D.get("pretrain_vmap_mem_budget_frac", 0.70)

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
    sft_allow_unconfigured_sources: bool = _D.get(
        "sft_allow_unconfigured_sources", False
    )
    sft_refusal_sources: tuple[str, ...] = tuple(
        _D.get("sft_refusal_sources", ())
    )

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
    minimum_source_completion_ratio: float = _D.get(
        "minimum_source_completion_ratio", 0.95
    )
    maximum_source_mix_deviation: float = _D.get(
        "maximum_source_mix_deviation", 0.02
    )
    minimum_corpus_completion_ratio: float = _D.get(
        "minimum_corpus_completion_ratio", 0.85
    )
    minimum_kind_completion_ratio: float = _D.get(
        "minimum_kind_completion_ratio", 0.50
    )
    maximum_kind_mix_deviation: float = _D.get(
        "maximum_kind_mix_deviation", 0.06
    )
    maximum_unplanned_source_share: float = _D.get(
        "maximum_unplanned_source_share", 0.05
    )
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
    filter_profiles: dict[str, dict[str, float | bool]] = field(
        default_factory=default_filter_profiles
    )
    sft_source_limits: dict[str, int | dict[str, int | bool]] = field(
        default_factory=default_sft_source_limits
    )

    def __post_init__(self):
        self.preset_name = normalize_preset_name(self.preset_name)
        self.corpus_source_plan = {
            normalize_corpus_source(source_name): dict(plan)
            for source_name, plan in self.corpus_source_plan.items()
        }
        self.filter_profiles = {
            str(profile_name).lower(): dict(profile)
            for profile_name, profile in self.filter_profiles.items()
        }
        self.sft_refusal_sources = tuple(
            str(source_name).strip()
            for source_name in self.sft_refusal_sources
            if str(source_name).strip()
        )
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.checkpoint_root_dir, self.preset_name)
        self.refresh_derived_fields()

    def refresh_derived_fields(self) -> None:
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.n_kv_heads < 0:
            raise ValueError("n_kv_heads must be >= 0")
        effective_kv_heads = self.n_kv_heads or self.n_heads
        if self.n_heads % effective_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.mlp_type = (self.mlp_type or "").lower()
        if self.mlp_type != "swiglu":
            raise ValueError("Spakie models require mlp_type='swiglu'")
        self.norm_type = (self.norm_type or "layernorm").lower()
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")
        self.position_encoding = (self.position_encoding or "").lower()
        if self.position_encoding != "rope":
            raise ValueError("Spakie models require position_encoding='rope'")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        self.loss_layout = (self.loss_layout or "flat").lower()
        if self.loss_layout not in {"flat", "3d", "custom"}:
            raise ValueError("loss_layout must be 'flat', '3d', or 'custom'")
        self.residual_type = (self.residual_type or "serial").lower()
        if self.residual_type not in {"serial", "parallel"}:
            raise ValueError("residual_type must be 'serial' or 'parallel'")
        self.attention_backend = (self.attention_backend or "sdpa").lower()
        if self.attention_backend not in {"sdpa", "mfa", "mfa-varlen"}:
            raise ValueError("attention_backend must be 'sdpa', 'mfa', or 'mfa-varlen'")
        self.muon_route = (self.muon_route or "all").lower()
        if self.muon_route not in {"all", "mlp", "attn", "none"}:
            raise ValueError("muon_route must be 'all', 'mlp', 'attn', or 'none'")
        if self.swiglu_hidden < 0:
            raise ValueError("swiglu_hidden must be >= 0")
        if not 0 <= self.minimum_source_completion_ratio <= 1:
            raise ValueError("minimum_source_completion_ratio must be between 0 and 1")
        if not 0 <= self.maximum_source_mix_deviation <= 1:
            raise ValueError("maximum_source_mix_deviation must be between 0 and 1")
        for name in (
            "minimum_corpus_completion_ratio",
            "minimum_kind_completion_ratio",
            "maximum_kind_mix_deviation",
            "maximum_unplanned_source_share",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        self.target_processed_tokens = derive_processed_token_target(self.target_train_tokens, self.train_split_fraction)
        if self.pretrain_target_tokens <= 0:
            self.pretrain_target_tokens = self.target_train_tokens
        self.pretrain_max_steps = derive_pretrain_max_steps(self.pretrain_target_tokens, self.pretrain_tokens_per_step())

    def pretrain_tokens_per_step(self) -> int:
        return self.pretrain_batch_size * self.pretrain_grad_accum_steps * self.max_seq_len

    def sft_source_enabled(self, source_name: str) -> bool:
        entry = self.sft_source_limits.get(source_name)
        if entry is None:
            return self.sft_allow_unconfigured_sources
        if isinstance(entry, dict):
            return bool(entry.get("enabled", True))
        return int(entry) > 0

    def corpus_source_kind(self, source_name: str) -> str:
        entry = self.corpus_source_plan.get(normalize_corpus_source(source_name), {})
        return str(entry.get("kind", "prose")).lower()

    def filter_profile_for_source(self, source_name: str) -> dict[str, float | bool]:
        kind = self.corpus_source_kind(source_name)
        profile_name = kind if kind in self.filter_profiles else "prose"
        return dict(self.filter_profiles.get(profile_name, {}))

    def sft_source_limit(self, source_name: str) -> int:
        entry = self.sft_source_limits.get(source_name, 0)
        if isinstance(entry, dict):
            return int(entry.get("limit", 0))
        return int(entry)

    def sft_source_download_limit(self, source_name: str) -> int:
        """Return the raw download budget, defaulting to the merge limit."""
        entry = self.sft_source_limits.get(source_name, 0)
        if isinstance(entry, dict):
            return int(entry.get("download_limit", entry.get("limit", 0)))
        return int(entry)

    def enabled_sft_sources(self, available_sources: set[str] | None = None) -> list[str]:
        sources = []
        for source_name in self.sft_source_limits:
            if available_sources is not None and source_name not in available_sources:
                continue
            if self.sft_source_enabled(source_name):
                sources.append(source_name)
        return sources

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


CHECKPOINT_CONFIG_SCHEMA_VERSION = 3


def config_to_dict(config: SpakieConfig) -> dict:
    """Return a primitive-only, versionable checkpoint representation."""
    return asdict(config)


def config_from_dict(payload: dict) -> SpakieConfig:
    """Rebuild the exact saved config, rejecting unknown or missing structure.

    ``SpakieConfig.__post_init__`` refreshes derived fields, so the two derived
    values are restored from the checkpoint after validation. This preserves an
    explicit ``--max-steps`` override instead of silently deriving a new run.
    """
    if not isinstance(payload, dict):
        raise TypeError("checkpoint config must be a dictionary")
    known_fields = {item.name for item in fields(SpakieConfig)}
    unknown = sorted(set(payload) - known_fields)
    if unknown:
        raise ValueError(f"checkpoint config contains unknown fields: {', '.join(unknown)}")
    values = dict(payload)
    missing = sorted(known_fields - set(values))
    if missing:
        raise ValueError(f"checkpoint config is missing fields: {', '.join(missing)}")

    if "muon_ns_coefficients" in values:
        values["muon_ns_coefficients"] = tuple(values["muon_ns_coefficients"])
    saved_derived = {
        key: values[key]
        for key in ("target_processed_tokens", "pretrain_max_steps")
        if key in values
    }
    config = SpakieConfig(**values)
    for key, value in saved_derived.items():
        setattr(config, key, value)
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
    return dirs
