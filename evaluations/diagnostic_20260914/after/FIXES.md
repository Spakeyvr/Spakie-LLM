# Implemented fixes and validation

- Torch and MLX now promote hidden states and weights before the output projection, retaining float32 logits and cross-entropy. MLX custom and fused backward paths compute in float32 and cast the final gradients back to their input dtypes. Masked, indexed, and chunked loss paths also use the float32 projection.
- The shared repetition penalty subtracts log(penalty), so a factor of 1.2 divides a repeated token's unnormalized probability by 1.2 independent of the common logit offset. This deliberately changes decoding semantics and may require tuning the factor for a particular checkpoint.
- Data filtering counts overlapping repeated five-word spans only once. Math-source repetition metrics exclude LaTeX commands while retaining the original text for repeated-line, other quality, language, and document-deduplication checks. Preparation schema is bumped to 5 so changed filtering cannot silently resume old partial shards.
- SFT accepts `--output-dir` for isolated checkpoints/status and has a top-level clean Ctrl+C handler. Actual SIGINT during recovery SFT exited cleanly and saved a resumable checkpoint.

## Measured results

- Identical trained base checkpoint and prompt: distinct output scores increased from 42 to 24,259; first-place ties decreased from 5 to 1. All four numerical probes now have a single leading token. Transformer weights/activations remain BF16 in these comparisons.
- Matched 16-window validation sample, 8,192 tokens: base loss 2.66986 before versus 2.64710 after; older SFT 2.90058 versus 2.89585. Small diagnostic sample, not a broad benchmark.
- Repeated all 128 original controlled generations after the fixes. The older SFT checkpoint now returns valid requested JSON for the Nia/age probe with penalty 1.2. Basic facts and arithmetic still fail. No broad capability recovery is claimed.
- Same 1,000-document FineMath spot sample: kept 508 before versus 811 after. Repeated-five-word rejections fell from 428 to 124. These are retention measurements, not proof that every additional document is high quality. The inspected worked trigonometry example now survives, and repeated-prose spam remains rejected in the regression test.
- Final unittest suite: 239 tests passed/run with one optional `mlx-mfa` test skipped because the package is absent. Added checks for precision under autocast, custom/fused loss gradients, offset-invariant repetition penalties, overlap accounting, and worked math retention. `py_compile` and `git diff --check` passed.

## Training and remaining limitations

A fresh SFT run from the latest base used all 218,849 canonical chat rows (207,907 training, 10,942 validation). The user requested stopping it due to heat and reduced Mac responsiveness. It stopped at update 396/3,249 and saved `checkpoints/92m/precision_recovery/sft_interrupt.safetensors`. No epoch or recovery capability evaluation completed. No background training or tests remain active.

Existing base/SFT checkpoints and processed corpus arrays were preserved. Revised filtering affects future data preparation; no full corpus rebuild was performed. The new precision path uses more head compute/memory than BF16 projection; no throughput improvement is claimed. Weak learned knowledge cannot be repaired by decoding code alone. Further training and its evaluation remain deferred at the user's request.

Raw before/after outputs, numerical probes, filtering audit, and recovery-run provenance are saved alongside this report. The parent REPORT.md records the original investigation and should be read as the pre-fix baseline.
