# Final 180M SFT result

## Selected checkpoint

- `checkpoints/180m/sft_fresh_balanced_e3_epoch2.safetensors`
- SHA-256: `c712494260f785cf481e5696fa9f228af80f67ef73c99ca7ff193cf1965ed790`
- Metadata SHA-256: `1b61c2e569253291857c094d18c50b9bbd27f8519ddc5079bbcaf6d018866a62`
- Prompt policy: no system turn.

## Reproducible training configuration

Source checkpoint: `checkpoints/180m/pretrain_best.safetensors`.

Dataset: `data/chat_raw/experiment_180m_sft_20260718/fresh_balanced_without_deepseek_e3.jsonl` (33,720 rows, SHA-256 `7d137a9f59fa94265439bc6e194552c50ccd21be4e9ba9fe93644bbee9e52d7b`).

Command:

```bash
env SPAKIE_MONITOR=0 .venv/bin/python3 scripts/finetune.py \
  --backend mlx --preset 180m \
  --source-checkpoint checkpoints/180m/pretrain_best.safetensors \
  --train-jsonl data/chat_raw/experiment_180m_sft_20260718/fresh_balanced_without_deepseek_e3.jsonl \
  --epochs 3 --sft-batch-size 64 --sft-grad-accum 4 --lr 3e-5 \
  --precision auto --output-name sft_fresh_balanced_e3_best.safetensors \
  --loss-log evaluations/180m_sft_20260718/e3_training_loss.csv \
  --no-model-prompt --pretokenize-sft
```

Select the saved epoch-2 checkpoint by capability evaluation, not the epoch-3 best-validation file.

## Exact dataset lists

Included foundation files:

- `custom.jsonl`
- `spakie_180m_identity.jsonl`
- `assistant_behavior.jsonl`
- `anti_echo.jsonl`
- `factual_repairs.jsonl`
- `nemotron_instruction_following_chat_v3.jsonl`
- `no_robots.jsonl`
- `smoltalk.jsonl`
- `squad.jsonl`
- `triviaqa.jsonl`
- `sciq.jsonl`
- `boolq.jsonl`

Included fourfold weighted skill files:

- `arithmetic_reasoning_scaled.jsonl`
- `logic_dates_spatial_calibration_scaled.jsonl`
- `python_semantics_scaled.jsonl`
- `language_transformation_scaled.jsonl`
- `structured_instruction_following_cleaned.jsonl`

Excluded files:

- `data/chat_exclude/DeepSeek-distill-V2_excluded_for_180m_sft.jsonl`: excluded because its complex, verbose answers dominated the 180M mix and the without-DeepSeek experiment produced the only broad improvement.
- `data/chat_exclude/structured_instruction_following_scaled_mislabeled.jsonl`: excluded because two six-wrapper prompt families had incorrect labels; replaced by the corrected recognizable file above.

## Final comparison

| Model | Clean core deterministic | Clean core semantic | Stable supplement semantic | Fresh semantic |
|---|---:|---:|---:|---:|
| Previous SFT | 7/64 | 4/64 | unavailable | unavailable |
| Initial required baseline | 7/64 | 2/64 | 3/36 | 2/27 |
| Final selected model | 13/64 | 10/64 | 2/36 | 2/27 |

The final model improves over the previous SFT by 85.7% deterministically and 150% semantically on the common repeatable core. It doubles total semantic passes versus the initial baseline across the core, stable supplement, and sealed fresh holdout (14 versus 7). Both the 30% target and 50% stretch target are reached on the shared comparison.

The fresh holdout ties rather than improves, and multiple-choice, arithmetic, coding, and structured-output behavior remain weak. Those limitations are retained explicitly rather than hidden by the aggregate gain.

## Verification

`.venv/bin/python3 -m unittest discover -s tests -v` passed all 189 tests. Evaluation result JSONL, semantic rating JSONL, rescored JSONL, and summaries are under `evaluations/sft_180m_20260718/results/`.
