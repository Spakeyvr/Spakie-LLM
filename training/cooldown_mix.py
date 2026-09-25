"""Backend-agnostic cooldown data mix (the "anneal v7" recipe).

During the final ``cooldown_mix_steps`` optimizer updates, some rows of every
microbatch are replaced by windows drawn from auxiliary token arrays (quality
sources, curricula). Which component fills a row is a pure function of the row's
global index, so the mix is exactly reproducible across resumes and identical
for the MLX and Torch loops.

Manifest (JSON), paths relative to the manifest file:

    {"slots": ["historical", ..., {"cycle": "hq"}, "math", {"cycle": "skill"}],
     "cycles": {"hq": ["wikipedia_definitions", ...], "skill": ["magnitude", "context"]},
     "components": {"math": {"path": "math.npy", "starts": "math_starts.npy",
                             "natural": false, "sha256": "...", "starts_sha256": "..."}, ...},
     "tokenizer_sha256": "..."}

``slots`` is a repeating layout over the global row stream ("historical" keeps
the sampler's row). ``natural`` marks rows that receive the unlikelihood term.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HISTORICAL = "historical"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_rng(seed: int, row_global: int) -> np.random.Generator:
    return np.random.default_rng([seed, row_global])


class CooldownMix:
    def __init__(self, manifest: dict, root: Path, seq_len: int, *, seed: int = 0, verify: bool = False,
                 tokenizer_path: str | None = None, vocab_size: int | None = None):
        self.seq_len = seq_len
        self.manifest = manifest
        self.seed = seed
        self.slots = manifest["slots"]
        self.cycles = {k: list(v) for k, v in manifest.get("cycles", {}).items()}
        if not self.slots:
            raise ValueError("cooldown mix manifest needs at least one slot")
        if tokenizer_path is not None:
            expected = manifest.get("tokenizer_sha256")
            if not expected or sha256_file(Path(tokenizer_path)) != expected:
                raise ValueError(
                    f"cooldown mix was built for tokenizer {expected!r}, but training uses {tokenizer_path}; "
                    "rebuild the mix with scripts/build_cooldown_mix.py for this tokenizer"
                )
        self.components: dict[str, dict] = {}
        for name, spec in manifest["components"].items():
            path = root / spec["path"]
            if verify:
                for key, file in (("sha256", path), ("starts_sha256", root / spec["starts"] if spec.get("starts") else None)):
                    if file is None:
                        continue
                    if not spec.get(key):
                        raise ValueError(f"cooldown mix component {name} has no {key} to verify")
                    if sha256_file(file) != spec[key]:
                        raise ValueError(f"cooldown mix component {name} changed: {file}")
            tokens = np.load(path, mmap_mode="r")
            if tokens.ndim != 1 or len(tokens) <= seq_len + 1:
                raise ValueError(f"cooldown mix component {name} must be a 1-D array longer than a window")
            if vocab_size is not None and int(tokens.max()) >= vocab_size:
                raise ValueError(f"cooldown mix component {name} has token ids >= vocab size {vocab_size}")
            starts = np.load(root / spec["starts"]) if spec.get("starts") else None
            if starts is not None and (len(starts) == 0 or starts[0] != 0):
                raise ValueError(f"cooldown mix component {name} document starts must begin at 0")
            self.components[name] = {"tokens": tokens, "starts": starts, "natural": bool(spec.get("natural", False))}
        referenced = set()
        for slot in self.slots:
            if isinstance(slot, dict):
                cycle = self.cycles.get(slot.get("cycle"))
                if not cycle:
                    raise ValueError(f"unknown cooldown mix cycle: {slot}")
                referenced.update(cycle)
            elif slot != HISTORICAL:
                referenced.add(slot)
        missing = referenced - set(self.components)
        if missing:
            raise ValueError(f"cooldown mix slots reference missing components: {sorted(missing)}")
        # Position of each slot within its cycle's per-layout occurrences.
        self._cycle_offsets: list[int] = []
        self._cycle_counts: dict[str, int] = {}
        for slot in self.slots:
            if isinstance(slot, dict):
                name = slot["cycle"]
                self._cycle_offsets.append(self._cycle_counts.get(name, 0))
                self._cycle_counts[name] = self._cycle_counts.get(name, 0) + 1
            else:
                self._cycle_offsets.append(0)

    @classmethod
    def from_path(cls, manifest_path: str, seq_len: int, *, seed: int = 0, verify: bool = False,
                  tokenizer_path: str | None = None, vocab_size: int | None = None) -> "CooldownMix":
        path = Path(manifest_path)
        return cls(json.loads(path.read_text()), path.parent, seq_len, seed=seed, verify=verify,
                   tokenizer_path=tokenizer_path, vocab_size=vocab_size)

    def fingerprint(self) -> str:
        """Identity of everything that determines sampled windows (manifest incl. all hashes, seed, length)."""
        payload = json.dumps({"manifest": self.manifest, "seed": self.seed, "seq_len": self.seq_len}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def component_for_row(self, row_global: int) -> str:
        layout = len(self.slots)
        slot = self.slots[row_global % layout]
        if not isinstance(slot, dict):
            return slot
        name = slot["cycle"]
        cycle = self.cycles[name]
        position = (row_global // layout) * self._cycle_counts[name] + self._cycle_offsets[row_global % layout]
        return cycle[position % len(cycle)]

    def is_natural(self, component: str) -> bool:
        return component == HISTORICAL or self.components[component]["natural"]

    def window(self, component: str, row_global: int) -> np.ndarray:
        spec = self.components[component]
        tokens, starts = spec["tokens"], spec["starts"]
        span = len(tokens) - (self.seq_len + 1)
        offset = int(_row_rng(self.seed, row_global).integers(0, span + 1))
        if starts is not None:
            offset = int(starts[np.searchsorted(starts, offset, side="right") - 1])
            offset = min(offset, span)
        return np.asarray(tokens[offset: offset + self.seq_len + 1], dtype=np.int32)

    def apply(self, x: np.ndarray, y: np.ndarray, microbatch_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replace non-historical rows in place-safe copies; return (x, y, natural_row_mask)."""
        rows = x.shape[0]
        x, y = np.array(x, copy=True), np.array(y, copy=True)
        natural = np.ones(rows, dtype=bool)
        for r in range(rows):
            row_global = microbatch_index * rows + r
            component = self.component_for_row(row_global)
            natural[r] = self.is_natural(component)
            if component == HISTORICAL:
                continue
            window = self.window(component, row_global)
            x[r], y[r] = window[:-1], window[1:]
        return x, y, natural


def load_cooldown_mix_for_training(config) -> "CooldownMix | None":
    """Load and fully verify the configured mix; pin its fingerprint in the config.

    The config is saved in every checkpoint, so a resume whose manifest, component or
    document-boundary arrays, tokenizer, or seed differ from the original run fails
    instead of silently sampling different windows.
    """
    if not config.cooldown_mix_manifest:
        return None
    mix = CooldownMix.from_path(config.cooldown_mix_manifest, config.max_seq_len, seed=config.cooldown_mix_seed,
                                verify=True, tokenizer_path=config.tokenizer_prefix + ".model",
                                vocab_size=config.vocab_size)
    fingerprint = mix.fingerprint()
    if config.cooldown_mix_fingerprint and config.cooldown_mix_fingerprint != fingerprint:
        raise ValueError(
            "cooldown mix inputs differ from the ones this run started with "
            f"(checkpoint {config.cooldown_mix_fingerprint[:12]}, now {fingerprint[:12]}); "
            "restore the original mix or start the cooldown from a checkpoint saved before it began"
        )
    config.cooldown_mix_fingerprint = fingerprint
    return mix


def cooldown_start_step(config) -> int | None:
    """First optimizer step of the cooldown mix, or None when disabled."""
    if not getattr(config, "cooldown_mix_manifest", ""):
        return None
    if config.cooldown_mix_start_step >= 0:
        return int(config.cooldown_mix_start_step)
    if config.cooldown_mix_steps <= 0:
        raise ValueError("cooldown mix needs cooldown_mix_steps > 0 or an explicit cooldown_mix_start_step")
    return max(0, int(config.pretrain_max_steps) - int(config.cooldown_mix_steps))


def unlikelihood_eligible_ids(tokenizer, vocab_size: int, min_token_id: int) -> np.ndarray:
    """Content word pieces eligible for the unlikelihood penalty (>= 3 letters, not the most frequent ids)."""
    eligible = np.zeros(vocab_size, dtype=bool)
    for token_id in range(min_token_id, vocab_size):
        piece = tokenizer.sp.id_to_piece(token_id).lstrip("▁")
        eligible[token_id] = len(piece) >= 3 and piece.isalpha()
    return eligible


def previous_token_candidates(x: np.ndarray, window: int) -> np.ndarray:
    """Candidates[b, t, j] = x[b, t - j] (the j-th most recent input token), -1 when out of range."""
    batch, length = x.shape
    out = np.full((batch, length, window), -1, dtype=np.int64)
    for j in range(window):
        out[:, j:, j] = x[:, : length - j]
    return out


def unlikelihood_inputs(x: np.ndarray, y: np.ndarray, natural: np.ndarray, eligible: np.ndarray,
                        window: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidate ids and weights for token-level unlikelihood (Welleck et al., 2019).

    The penalty is sum(weights * -log(1 - p(candidate))). Candidates are recent eligible
    content tokens other than the true next token, on natural-text rows only. Weights
    average over positions and natural rows so the term matches the validated recipe.
    """
    candidates = previous_token_candidates(x, window)
    valid = (candidates >= 0) & (candidates != y[..., None])
    safe = np.maximum(candidates, 0)
    valid &= eligible[safe]
    valid &= natural[:, None, None]
    denominator = x.shape[1] * max(int(natural.sum()), 1)
    return safe.astype(np.int32), valid.astype(np.float32) / denominator
