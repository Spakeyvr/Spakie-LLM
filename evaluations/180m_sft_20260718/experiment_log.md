# Spakie-180M SFT improvement experiment log

Started: 2026-07-18 (Europe/Vienna)

## Goal and promotion rule

- Start every full-run comparison from `checkpoints/180m/pretrain_best.safetensors` unless an experiment explicitly tests continuation from an accepted SFT checkpoint.
- Required initial run: three SFT epochs over the exact currently merged dataset.
- Primary promotion gate: repeatable broad held-out capability, not train or validation loss.
- Target: at least 30% relative overall improvement over the previous SFT model, while continuing toward 50% and rejecting serious category regressions.
- Generation is deterministic and identical across checkpoints. No inference-time tools are used.

## Immutable starting artifacts

| Artifact | SHA-256 |
|---|---|
| `checkpoints/180m/pretrain_best.safetensors` | `17599ef0d95507de637968439b4086294c5448a505bc23045c308ae76a93d92f` |
| `checkpoints/180m/pretrain_best.safetensors.meta.json` | `350bde296ff81cadd1580f41b8ff77b875af1c9a93c9779a5abcc2f1c88a874e` |
| `data/chat/train.jsonl` at experiment start | `0eb9caabf64005dd510f28203454b03ce94ad905039e2499a196f3dfc2b783fe` |
| frozen baseline copy in `data/chat_raw/experiment_180m_sft_20260718/baseline_current_included_mix.jsonl` | `0eb9caabf64005dd510f28203454b03ce94ad905039e2499a196f3dfc2b783fe` |
| stable 66-prompt legacy core | `7a405c223b8ac3001fd2b6d5fd10ea29a110d7c48800eaadb60af8061bd2f5af` |

The baseline training copy contains 87,789 conversations and no system turns. It was cloned under `data/chat_raw/` so later mixture rebuilds cannot silently change the required first run.

### Included baseline source files

- `DeepSeek-distill-V2.jsonl`
- `anti_echo.jsonl`
- `assistant_behavior.jsonl`
- `boolq.jsonl`
- `custom.jsonl`
- `factual_repairs.jsonl`
- `nemotron_instruction_following_chat_v3.jsonl`
- `no_robots.jsonl`
- `sciq.jsonl`
- `smoltalk.jsonl`
- `spakie_180m_identity.jsonl`
- `squad.jsonl`
- `triviaqa.jsonl`

No source has been moved to `data/chat_exclude/` yet.

## Previous SFT evidence

The previous `sft_best.safetensors` checkpoint is no longer present anywhere indexed on the local machine. Its saved outputs on the unchanged 66-prompt core are preserved at `evaluations/general_capability/baseline_qa_expanded.jsonl`. Those outputs allow repeatable rescoring on the shared core but not generation on newly added prompts.

The older experiment log records 18/66 mechanically for that model, but also documents many semantic false positives. That number is therefore not accepted as the final comparison baseline. The shared core will be rescored with stricter deterministic checks and an explicit manual rubric where semantics cannot be checked reliably.

## Experiment table

| ID | Source checkpoint | Training data | Config | Core score | Fresh score | Decision |
|---|---|---|---|---:|---:|---|
| E0 previous SFT | unavailable artifact; saved outputs only | previous 87,789-row mix | previous run metadata | pending strict rescore | unavailable | comparison anchor |
| E1 required 3-epoch baseline | current pretrain best | frozen 87,789-row current mix | MLX, Muon, 3 epochs, LR 3e-5, batch 64, accumulation 4 | pending | pending | pending |

## E1 hypothesis

The current pretrain checkpoint completed substantially more pretraining than the checkpoint used by the archived SFT loop. A clean three-epoch full-mixture SFT run may therefore improve broad behavior without any new curriculum. This must be measured before adding or excluding data.

