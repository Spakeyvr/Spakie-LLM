# Token Throughput Research

This note summarizes the throughput work that produced the accepted 300M pretrain speedup, plus the main ideas that were tested and rejected.

## Accepted 300M Pretrain Change

The useful win is MLX vmap-based gradient accumulation for the 300M pretrain preset.

Baseline:

- Preset: `300m`
- Batch/accumulation: `B64/G2`
- Tokens per optimizer step: `65,536`
- Throughput: `10,210 tok/s`
- Evidence: `bench_results/fixed_baseline_300m_pretrain_20260603_122214.log`

Accepted candidate:

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

## Why It Helped

The original MLX pretrain path paid too much overhead around gradient accumulation. The accepted path stacks the accumulation microbatches and uses `mx.vmap` inside a compiled training step, so MLX sees one larger, more regular unit of work instead of a looser Python-side accumulation loop.

This does not make every GEMM individually faster. The win comes from better amortization and scheduling around the model forward/backward work while keeping the optimizer math and loss accounting equivalent.

## Implementation Shape

The key plumbing is:

- `configs/default.yaml`: 300M now uses `pretrain_grad_accum_steps: 3` and `pretrain_vmap_accum_step: true`.
- `configs/default.py`: exposes `pretrain_vmap_accum_step`.
- `scripts/train.py`: resolves the config default and passes it into MLX pretraining.
- `training/pretrain_mlx.py`: adds the vmap accumulation training step and keeps the ordinary path available.
- `scripts/benchmark_mlx_training.py`: can benchmark the vmap accumulation path with fixed shapes.

## Other Presets

The vmap accumulation idea was checked against smaller presets and was not adopted there.

- `92m` baseline `B92/G1`: `32,965 tok/s`.
- `92m` vmap probe `B46/G2`: `23,109 tok/s`, worse than baseline.
- `180m` baseline `B96/G2`: `14,818 tok/s`.

So the 300M preset is the only preset changed by the accepted default. The smaller presets are effectively unaffected.

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

## Current Recommendation

Merge the scoped 300M pretrain improvement plus the conservative MLX SFT
sortish-bucket/right-trim path. Keep packing, varlen attention, compaction,
custom kernels, grouped Muon, and other benchmark-only branches opt-in until
they have separate sustained evidence.
