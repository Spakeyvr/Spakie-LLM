# Token Throughput Research

This note summarizes the throughput work, including the 300M pretrain candidate
that looked good in benchmarks but was later rejected for system stability, plus
the SFT improvement that remains accepted.

## Current Stable 300M Pretrain Default

The real `scripts/train.py` 300M pretrain default is the non-vmap `B128/G2`
path:

- Preset: `300m`
- Batch/accumulation: `B128/G2`
- Tokens per optimizer step: `131,072`
- `pretrain_vmap_accum_step: false`
- Throughput: `13,032 tok/s` on a short real-trainer probe with eval and
  rolling checkpoints enabled.
- Speedup: `+27.6%` over the fixed `B64/G2` baseline.
- Evidence: `bench_results/stability_300m_b128g2_real_train_*.log`

The faster vmap candidate below is kept as an experimental opt-in path only.
It produced strong benchmark numbers, but repeated full-trainer attempts caused
macOS kernel watchdog panics during early steps on the M5/macOS 26.5.1 test
machine. A kernel-panicing path is not considered committable as a default.

## Rejected 300M Pretrain Candidate

MLX vmap-based gradient accumulation was the strongest 300M pretrain speed
candidate.

Baseline:

- Preset: `300m`
- Batch/accumulation: `B64/G2`
- Tokens per optimizer step: `65,536`
- Throughput: `10,210 tok/s`
- Evidence: `bench_results/fixed_baseline_300m_pretrain_20260603_122214.log`

Candidate:

- Preset: `300m`
- Batch/accumulation: `B64/G3`
- Tokens per optimizer step: `98,304`
- `pretrain_vmap_accum_step: true`
- Throughput: `17,635 tok/s`
- Step mean: `17,636 tok/s`
- Speedup: `+72.7%` over the fixed baseline
- Evidence: `bench_results/candidate_300m_pretrain_b64g3_vmap_full_20260604_035606.log`

The 500-step parity run also held:

- Baseline parity throughput: `11,220 tok/s`
- Candidate parity throughput: `17,715 tok/s`
- Final train-loss delta: candidate `4.4375` vs baseline `4.453125`
- Max relative loss delta across 500 matched steps: `0.69%`
- Evidence:
  - `bench_results/parity_300m_pretrain_baseline_b64g2_seed123_500.csv`
  - `bench_results/parity_300m_pretrain_b64g3_vmap_seed123_500.csv`

Validation-loss check:

- Initial val loss matched at `9.9375`.
- Baseline step 40 val loss: `7.707031`.
- Candidate step 40 val loss: `7.683594`.
- Relative delta: about `-0.30%`, well inside the requested 5% bound.
- Evidence:
  - `bench_results/valcheck_300m_baseline_b64g2_20260605_232529.log`
  - `bench_results/valcheck_300m_vmap_b64g3_20260605_233445.log`

## Why It Helped In Benchmarks

The original MLX pretrain path paid too much overhead around gradient accumulation. The accepted path stacks the accumulation microbatches and uses `mx.vmap` inside a compiled training step, so MLX sees one larger, more regular unit of work instead of a looser Python-side accumulation loop.

This does not make every GEMM individually faster. The win comes from better amortization and scheduling around the model forward/backward work while keeping the optimizer math and loss accounting equivalent.

## Implementation Shape

The key plumbing is:

- `configs/default.yaml`: 300M defaults to `pretrain_batch_size: 128`,
  `pretrain_grad_accum_steps: 2`, and `pretrain_vmap_accum_step: false`.
- `configs/default.py`: exposes `pretrain_vmap_accum_step`.
- `scripts/train.py`: resolves the config default and passes it into MLX pretraining.
- `training/pretrain_mlx.py`: keeps the vmap accumulation training step available, but the ordinary path is the default.
- `scripts/benchmark_mlx_training.py`: can benchmark the vmap accumulation path with fixed shapes.

## Other Presets

The vmap accumulation idea was checked against smaller presets and was not adopted there.

- `92m` baseline `B92/G1`: `32,965 tok/s`.
- `92m` vmap probe `B46/G2`: `23,109 tok/s`, worse than baseline.

## Stable SFT Improvement

The SFT path now keeps the safe part of the SFT work: sortish length-bucketed
batches plus right-padding bucket trim. This avoids dense transformer work on
right-padding while keeping ordinary dense SDPA, ordinary cross-entropy, and the
same per-example token order inside each emitted microbatch.

300M SFT benchmark, real `data/chat/train.jsonl`, `B16/G2`, 40 warmup steps,
220 timed optimizer steps:

- Plain padded baseline: `8,715 tok/s` nominal, `3,604,480` physical tokens.
- Sortish bucket + trim candidate: `14,103 tok/s` nominal, `2,031,616` physical tokens.
- Speedup: `+61.8%` nominal dataset tokens per second.
- Forward/backward timed bucket: `352.5s` -> `192.8s`.
- Evidence:
  - `bench_results/stable_sft_300m_plain_baseline_b16g2_220_20260606.log`
  - `bench_results/stable_sft_300m_sortish128_trim_b16g2_220_20260606.log`

I also checked homogeneous same-shape SFT accumulation groups:

- Homogeneous-step sorted + trim: `13,982 tok/s`.
- It was slightly slower than the simpler sortish default and did not reduce
  step-throughput variance, so it remains an opt-in benchmark/sampler mode
  rather than the recommended default.
- Evidence: `bench_results/stable_sft_300m_homogeneous128_trim_b16g2_220_20260606.log`

## Ideas Tested And Rejected

These were tried because they were plausible, but did not survive sustained benchmarks or parity checks:

- Larger single microbatches such as `B96`, `B128`: promising cold starts, but sustained throughput collapsed.
- Compile-only and full-step compile variants: not enough sustained improvement versus the accepted vmap accumulation path.
- Loss-only tricks, flat loss layouts, and final-loss-only evaluation: helped measurement overhead in places but did not solve the real training bottleneck.
- SFT packing, varlen attention, aggressive sorted/static-shape variants, and
  dense block-causal packing: many probes improved cold windows but regressed or
  collapsed over longer runs. The conservative sortish-bucket plus trim path is
  the kept exception.
- Varlen attention and static `cu_seqlens`: the `.tolist()`/host-sync path was a real blocker, and the dense attention path stayed faster for this shape.
- Gathered or compacted SFT loss: too late in the graph; cross-entropy was not the dominant cost.
- Valid-token MLP/projection compaction and block-masked matmul: gather/scatter or masked-kernel overhead beat the saved pad work.
- `mx.addmm` residual projection epilogues and forced contiguity: low-risk ideas, but sustained measurements were worse or neutral.
- MLP checkpointing: recompute hurt because GEMM time was already the wall.
- Grouped Muon and optimizer-route variants: useful to understand optimizer cost, but did not close the target gap safely.
- Custom Metal/residual RMSNorm/GEMM probes: no validated sustained win over the accepted dense MLX path.
- FP16/dtype/GEMM shape audits: useful diagnostics, but no clean hidden dtype or shape fast-path miss was found that beat the accepted default.
- 300M MLX vmap pretrain accumulation: excellent benchmark throughput and loss
  parity, but rejected as a default after repeated real-trainer macOS kernel
  watchdog panics during early steps.

## 92M Probes (June 2026, seed-matched 60-step real-data runs)

- GQA `n_kv_heads: 4` matched full-MHA loss step-for-step (final loss identical,
  max delta 0.81%) with strictly less attention compute. Adopted as the 92m
  default. Evidence: `bench_results/claude_opt/92m_real_{base,gqa}.csv`.
- Param-matched SwiGLU (`swiglu_hidden: 2048`) ran 4.2% worse on the same probe
  with no throughput gain. Not adopted at 92m.
- Disabling grad clip was loss-safe (<2% delta) but gave no measurable
  throughput win at 92m. Not adopted.
- Sequential single-run throughput comparisons drifted up to ~15% from thermal
  soak; use the loss CSVs and interleaved cycles, not back-to-back tok/s.

## QK-Norm (June 2026, opt-in)

Per-head RMSNorm on Q and K before attention (Qwen3/Gemma2 style), gated by the
new `qk_norm` config flag (default `false`). Motivation: the Kimi K2 team report
that attention-logit growth under Muon is more common than under AdamW, and this
project trains with Muon at 10 NS steps.

- Implemented in both backends (`model/transformer.py`,
  `model/transformer_mlx.py`) with a Torch↔MLX forward-parity test
  (`tests/test_mlx_parity.py::test_forward_logits_match_with_qk_norm`, randomized
  gains, max abs logit diff < 1e-3).
- Not supported with the `mfa-varlen` attention backend (raises); that path is a
  benchmark-only opt-in and never combined with qk_norm.
- 92m seed-matched 60-step real-data probe: final loss identical, max per-step
  delta 0.90%, marginally lower loss in the first ~20 steps. Throughput cost is
  small (single-digit %, partly thermal). Evidence:
  `bench_results/claude_opt/92m_real_qknorm_{off,on}.csv`.

Kept opt-in rather than enabled on a preset: a 60-step probe confirms it is
loss-safe but cannot demonstrate the actual payoff (stability over long runs /
headroom to raise LR). Recommended next step before adopting as a default: a
longer run, ideally with an increased `pretrain_lr`, to show qk_norm prevents a
spike or allows a higher stable LR. Enable per-run with `--qk-norm` on
`scripts/train.py` or set `qk_norm: true` in a preset.

## 180M Probes (June 2026, 50-step seed-matched real-data runs)

All easy levers were tried and none beat the existing `B96/G2` pretrain and
`B32/G4` SFT defaults; nothing was adopted. Evidence: `bench_results/claude_180m/*.log`.
Throughput drifts down several percent across sequential runs from thermal soak;
a soaked baseline re-run (21.7k tok/s pretrain, 76s SFT epoch) is the fair
comparison for later runs.

- Pretrain `B128/G2`: 19.4k tok/s vs baseline 21.7–22.5k — clearly slower at
  this size (matches the larger-microbatch collapse pattern), val_loss fine.
- Pretrain `B96/G1`: 17.8k tok/s — Muon step overhead unamortized; worst.
- Pretrain `B96/G3`: 21.0k tok/s — no win over G2.
- Pretrain `muon_ns_steps 5`: ~+1–2% tok/s, but val_loss 7.097 vs 7.058
  baseline (+0.039 at 50 steps) — over tolerance, scrapped.
- SFT `B64/G2`, `B32/G8`, `--sft-length-bucket-size 512`: all within thermal
  noise of the 76s baseline epoch; no win.

Kept side improvement: `scripts/finetune.py` now accepts `--sft-batch-size`
and `--sft-grad-accum` overrides (config defaults unchanged).

## Current Recommendation

Use the stable non-vmap `B128/G2` 300M pretrain default for real runs. Keep the
conservative MLX SFT sortish-bucket/right-trim path. Keep vmap pretrain
accumulation, packing, varlen attention, compaction, custom kernels, grouped
Muon, and other benchmark-only branches opt-in until they have separate
sustained full-trainer evidence.
