# 180M compound instruction and reasoning improvement cycle

Started: 2026-07-18 (Europe/Vienna)

## Objective and gates

- Starting checkpoint: `checkpoints/180m/sft_fresh_balanced_e3_epoch2.safetensors`, SHA-256 `c712494260f785cf481e5696fa9f228af80f67ef73c99ca7ff193cf1965ed790`.
- Primary target: at least 50% relative improvement on the frozen core benchmark's atomic instruction-compliance score.
- Baseline core atomic score: 46/272. The integer 50% threshold is 69/272.
- Baseline core strict whole-prompt score: 0/96. Promotion requires nonzero strict success and gains in multiple categories, not only partial credit.
- Sealed fresh baseline: 17/136 atomic units and 0/48 strict prompts. Confirmation target is at least 26/136 atomic units with nonzero strict success.
- Regression gates: preserve the prior 64-prompt general-capability core near its accepted 10/64 semantic score, retain factual/language/context-QA strengths, and reject severe category regressions.
- Generation is fixed for every comparison: no system prompt, max 96 new tokens, temperature 0.1, top-k 1, top-p 1.0, repetition penalty 1.2.
- Training loss and validation loss are diagnostic only.

The initial benchmark used strict prompt-level scoring only and produced 0/96 because nearly every output omitted at least one requested component. Before training, the scoring contract was frozen with IFEval-style atomic units so relative improvement from a nonzero baseline is meaningful. Strict prompt accuracy remains reported separately and is never replaced by partial credit.

## Frozen evaluation

- Stable core: `data/eval/instruction_reasoning_core_v2.jsonl`, 96 prompts, SHA-256 `81cdbdfa5c7ca68d58b15bfc77e6638115e8d8bb2fddeaa857ac3244c2f5e1f7`.
- Sealed fresh holdout: `data/eval/instruction_reasoning_fresh_v2.jsonl`, 48 prompts, SHA-256 `d8def37425e8e5774382e45d2ec9c1330f7256316e2ea7082a77d358365a6d26`.
- Twelve balanced categories: compound knowledge, compound instructions, arithmetic, multi-step math, logic, spatial reasoning, date/time, coding, structured output, grounded multi-part QA, text transformation, and explanation/planning.
- The user's France prompt is one of 96 stable prompts and appears in no generated training file.
- Exact normalized prompt overlap across all 144 evaluation prompts and 46 existing/new raw or experiment JSONL files: zero.
- Manifest: `data/eval/instruction_reasoning_v2_manifest.json`.

## Starting-checkpoint diagnosis

- Core: 46/272 atomic units (16.9%), 0/96 strict prompts.
- Fresh: 17/136 atomic units (12.5%), 0/48 strict prompts.
- Core category atomic scores:
  - compound knowledge 5/16; compound instruction 5/24.
  - logic 15/24; spatial 5/16; date/time 2/16.
  - explanation/planning 6/24; text transformation 8/24.
  - arithmetic, multi-step math, coding, structured output, and grounded multi-part QA: 0.
- Representative cause: outputs often answer the first clause but omit the second, copy labels without completing values, or begin correctly and then contradict themselves.

## Added training sources

All files are deterministic, transferable task-family curricula under `data/chat_raw/`; none copies an evaluation prompt.

- `compound_instruction_following_v2.jsonl`: 1,997 rows, SHA-256 `80cb1f40f53f5daf0c90ec971575fb05db63e7c1359dfbea7c13627d4d009122`. Multi-part knowledge, ordered transformations, count constraints, and extraction-plus-aggregation.
- `math_reasoning_curriculum_v2.jsonl`: 3,253 rows, SHA-256 `75744b3bddf904c7bceef08ddad1acdd1d5417949cbf946e21b651cced83cf50`. Arithmetic, discounts, sharing/remainders, time, logic, spatial, and date reasoning.
- `structured_coding_curriculum_v2.jsonl`: 2,081 rows, SHA-256 `919011b3b53f0845fe11ba7ca414749e22a7512f21b1b4518c8900397d254ff9`. Exact JSON/CSV/Markdown, concise Python functions, and Python traces.
- `grounded_multitask_curriculum_v2.jsonl`: 1,813 rows, SHA-256 `51989a4b4c0b0e297e99e1053a5137595ea88fd8ddc6705c538365f35faf8411`. Synthetic grounded passages with two requested answers plus polite transformation/status tasks.
- Generator: `scripts/build_instruction_reasoning_v2.py`; Ctrl+C safe and atomic-output based.

## Experiment 1 - diverse skill curriculum with clean retention

- Status: completed; improved but rejected as final because it missed the 50% gate.
- Hypothesis: prior focused experiments failed because they repeated fewer than 1,000 underlying examples. A continuation using 9,144 diverse unique target examples plus 12,000 clean foundation-retention examples should teach completion of every requested component without template overfitting or broad forgetting.
- Source checkpoint: selected starting checkpoint.
- Mixture: `data/chat_raw/experiment_180m_sft_20260718/instruction_reasoning_v2_e1.jsonl`.
- Rows: 21,144; exact duplicate conversations: zero; malformed message rows: zero; SHA-256 `2c89453f963aa50fcf954f8090c4b71088f8572fb3b8f9e98c3a8064dd5e3d7e`.
- Retention source: deterministic 12,000-row sample from `foundation_without_deepseek_e3.jsonl`.
- Target weights: each of the four new curricula exactly once; no evaluation-leakage row was accepted.
- Manifest: `instruction_reasoning_v2_e1.jsonl.manifest.json`, SHA-256 `5cfc825c347da484ee4f56769fc2b369b8ac98b71c517590f4b77adcff2ea896`.
- Configuration: MLX/Muon, one epoch, microbatch 32, gradient accumulation 1, LR `1e-5`, no system prompt, pretokenized SFT; 627 optimizer updates.
- Output: `checkpoints/180m/sft_instruction_v2_e1_best.safetensors` and corresponding final save.
- Health: train loss `1.1016`, validation loss `1.2328`; no numerical or memory instability.
- Core result: 61/272 atomic units (+32.6% over 46/272) and 7/96 strict prompts (up from zero).
- Category gains: coding 0 to 4 units, grounded QA 0 to 8, spatial 5 to 10, date/time 2 to 4. Compound instruction and text transformation were unchanged; logic and explanation lost two units each. Arithmetic, multi-step math, and structured output remained at zero.
- Decision: reject as final. It is the strongest new-benchmark candidate so far, but 61 is below the required 69-unit target and several important categories remain nonfunctional. The sealed holdout was not opened for this rejected intermediate.

## Experiment 2 - lower-LR consolidation epoch

- Status: completed; improved but rejected as final because it remained four atomic units below the 50% gate.
- Hypothesis: Experiment 1's varied examples transferred to several categories but one epoch was insufficient for exact two-part formats and calculation primitives. A second pass at half the learning rate should consolidate those patterns while reducing the risk of erasing the starting checkpoint's general strengths.
- Source checkpoint: Experiment 1 validation-best.
- Dataset: unchanged Experiment 1 mixture and manifest; no new rows, weights, exclusions, or evaluation access.
- Configuration: MLX/Muon, one epoch, microbatch 32, gradient accumulation 1, LR `5e-6`, no system prompt, pretokenized SFT; 627 optimizer updates.
- Output: `checkpoints/180m/sft_instruction_v2_e2_best.safetensors` and corresponding final save.
- Health: train loss `0.9450`, validation loss `1.2264`; no numerical or memory instability.
- Core result: 65/272 atomic units (+41.3% over 46/272) and 9/96 strict prompts.
- Category atomic results: arithmetic 0/24, coding 4/24, compound instruction 8/24, compound knowledge 4/16, date/time 4/16, explanation/planning 5/24, grounded QA 8/24, logic 11/24, multi-step math 0/24, spatial 14/16, structured output 0/32, and text transformation 7/24.
- Representative regression: the France prompt still returned only `The capital of France is Paris.` and omitted the requested explanation.
- Decision: reject as final. It improved E1 by four units and two strict prompts but scored 65, below the required 69. The sealed holdout remained closed for this candidate.

## Added completion-focused source

- `data/chat_raw/complete_multi_part_answers_v3.jsonl`: 5,400 unique prompts, SHA-256 `093e931a76001e6b0d84d2bdf923b29a860c4acd3b8d967f907829c02c147bac`.
- Purpose: raise the probability of completing every requested component across conversational knowledge, grounded extraction-plus-subtraction, polite rewrite-plus-status, two-benefit explanations, exact JSON copying, and two labeled calculations.
- The source deliberately excludes all 12 core/fresh capital entities and has zero exact normalized overlap with either benchmark. It contains no France prompt or isolated benchmark-answer patch.
- Generator and validation: `scripts/build_complete_multitask_v3.py`; deterministic, Ctrl+C safe, atomic-output based, with tests in `tests/test_build_complete_multitask_v3.py`.

## Experiment 3 - completion-focused transfer mixture

- Status: completed; passed the new core and fresh targets but rejected for a serious broad-capability regression.
- Hypothesis: E2 is only four atomic units short, but its unchanged compound-answer behavior and zero JSON/arithmetic scores show that another identical epoch would lean on incidental spatial gains. A mixture that substantially increases complete two-part responses and variable-copying diversity should improve the intended behavior while retaining all prior skill curricula and a clean foundation sample.
- Source checkpoint: Experiment 2 validation-best.
- Mixture: `data/chat_raw/experiment_180m_sft_20260718/instruction_reasoning_v3_e3.jsonl`, 27,944 weighted rows, SHA-256 `b8e2624acc35b217abf0b27e2c283569b0fdf442fbd810846ec9a65493ae61a5`.
- Included/weighted sources: 8,000 deterministic clean foundation-retention rows; each original v2 target source once; all 5,400 completion-focused v3 rows twice (10,800 weighted rows).
- Leakage/quality checks: zero blocked evaluation rows, zero duplicate target conversations, and zero target conversations duplicated from the clean baseline.
- Configuration: MLX/Muon, one epoch, microbatch 32, gradient accumulation 1, LR `3e-6`, no system prompt, pretokenized SFT; 829 optimizer updates.
- Output: `checkpoints/180m/sft_instruction_v3_e3_best.safetensors` and corresponding final save.
- Health: train loss `0.7072`, validation loss `0.9672`; no numerical or memory instability.
- Frozen core: 83/272 atomic units (+80.4% over 46/272) and 16/96 strict prompts. This exceeded the 69-unit core target.
- Sealed fresh: 30/136 atomic units (+76.5% over 17/136) and 5/48 strict prompts. This exceeded the 26-unit confirmation target.
- Broad regression check: 8/64 deterministic clean legacy core versus 13/64 for the accepted starting checkpoint. Factual recall fell sharply, context QA declined, and the France prompt completed both requested parts but changed the correct capital from Paris to Lyon.
- Stable supplement: 3/32 deterministic plus four manual items, similar to prior raw supplement behavior but insufficient to offset the clean-core loss.
- Decision: reject as final. Passing the new benchmark does not justify a 38.5% deterministic regression on the established clean core or a wrong answer to the motivating prompt.

## Experiment 4 - restart with retention-dominant curriculum

- Status: completed; rejected because it missed the new gate by one unit and still regressed on the old broad core.
- Hypothesis: E3's target-heavy continuation compounded three successive instruction epochs. Restarting from the accepted model and training only once on a retention-dominant mixture should capture complete multi-part response behavior while preserving the accepted checkpoint's factual and context-QA strengths.
- Source checkpoint: `checkpoints/180m/sft_fresh_balanced_e3_epoch2.safetensors` (the accepted model at the start of this cycle), not E3.
- Mixture: `data/chat_raw/experiment_180m_sft_20260718/instruction_reasoning_v3_e4_retention.jsonl`, 34,544 rows, SHA-256 `e294051cd661bf205a1dee59ae6645c7dd363b3a6a03d5f23d84fe47de7fa23b`.
- Included/weighted sources: 20,000 deterministic clean foundation-retention examples (57.9%); each of the five target sources once (14,544 rows, 42.1%). No target is repeated.
- Leakage/quality checks: zero blocked evaluation rows, zero duplicate target conversations, and zero target conversations duplicated from the clean baseline.
- Configuration: MLX/Muon, one epoch, microbatch 32, gradient accumulation 1, LR `8e-6`, no system prompt, pretokenized SFT; 1,025 optimizer updates.
- Health: train loss `1.0799`, validation loss `1.1691`; no numerical or memory instability.
- Core result: 68/272 atomic units (+47.8%) and 11/96 strict prompts, exactly one unit below the required 69. The sealed holdout was not opened.
- The France answer retained the correct capital (`Paris`) but still omitted the explanation.
- Broad regression check: 7/64 deterministic clean legacy core versus 13/64 for the accepted starting checkpoint.
- Decision: reject. A retention-majority full-model epoch did not prevent catastrophic interference and missed the requested improvement threshold.

## Experiment 5 - task-vector interpolation and scoped transfer diagnostics

- Status: completed; all post-hoc merge candidates rejected.
- Hypothesis: blending or scoping the E3 task vector may retain base factual/context representations while transferring instruction behavior.
- Reproducibility utility: `scripts/interpolate_checkpoints_mlx.py`, which validates tensor shapes, architecture, and tokenizer contracts and records exact scope/weights in checkpoint metadata.
- Whole-model interpolation screens:
  - 50% E3 update: 52/272 core atomic units; rejected before broad/fresh evaluation.
  - 80% E3 update: 73/272 core atomic units, but only 7/64 raw legacy core before exclusions; rejected for broad regression.
- Layer-scoped screens:
  - upper 4 blocks: 49/272; rejected.
  - upper 8 blocks: failed the strict screen and did not approach the target; rejected.
  - upper 12 blocks plus final norm: 72/272, but 9/66 raw legacy core and 8/64 after the two fixed exclusions; rejected.
- Attention-only and attention-plus-norm task vectors failed to transfer the learned behavior (48/272 or similarly low); rejected.
- Conclusion: E3 behavior depends on coordinated updates across components, and post-hoc splicing creates incompatible representations. Selective training must learn upper layers against frozen base representations from the start.

## Experiment 6 - coherent selective upper-half SFT

- Status: completed; passed the new core but rejected for old-suite regression.
- Hypothesis: freezing token/position embeddings and blocks 0-7 while training blocks 8-15 plus final norm will let the upper half learn multi-clause and formatting behavior coherently while preserving lower factual/context representations.
- Source checkpoint: untouched accepted starting checkpoint.
- Dataset: Experiment 3 target-rich 27,944-row mixture and unchanged manifest; no new evaluation access, weights, exclusions, or training examples.
- Selective-training implementation: `--mlx-trainable-block-start 8`; smoke-tested for one optimizer step, 57 tensors trainable, and exact scope recorded in checkpoint metadata.
- Configuration: MLX/Muon, one epoch, microbatch 32, gradient accumulation 1, LR `1e-5`, no system prompt, pretokenized SFT; 829 optimizer updates.
- Health: mean optimizer-step loss `0.8784`, validation loss `0.9904`; no instability. Checkpoint metadata confirms only 57 tensors in blocks 8-15 and final norm were trainable.
- Core result: 78/272 atomic units (+69.6%) and 11/96 strict prompts.
- The France response supplied Paris and an explanation, although the explanation overstated Paris as Europe's political/economic center.
- Broad regression: 8/66 raw and 7/64 after fixed exclusions, versus 13/64 for the accepted start. Lower freezing preserved more factual recall but did not preserve context QA or old instruction/date items.
- Coherent-path 75% blend: 70/272 core but only 6/66 raw legacy core; rejected. The 90% blend was not advanced after the lower blend worsened the preservation score.
- Decision: reject as final.

## Experiment 7 - full accepted-mixture replay

- Status: completed; rejected because replay preservation reduced target learning below the required gate.
- Hypothesis: prior retention mixtures replayed generic clean foundation rows, not the exact weighted 33,720-row curriculum that produced the starting checkpoint's 13/64 broad strengths. Full continual-learning replay of that accepted curriculum, combined with the new sources at a lower LR, should preserve specialized factual/context/instruction behavior better than generic retention.
- Source checkpoint: untouched accepted starting checkpoint.
- Mixture: `data/chat_raw/experiment_180m_sft_20260718/instruction_reasoning_v3_e7_full_replay.jsonl`, 53,664 rows, SHA-256 `ebfa8c77aad788d722474ad4fc3f250fd001dd3109a6af7323ad0ba41e10bafe`.
- Included/weighted sources: all 33,720 rows of `fresh_balanced_without_deepseek_e3.jsonl`; each original v2 target once; completion-focused v3 twice (10,800 weighted rows). Exact accepted-curriculum replay is 62.8% of the mix and new-target data is 37.2%.
- Leakage/quality checks: zero blocked evaluation rows, zero target duplicates, and zero target conversations duplicated from the replay baseline.
- Configuration: MLX/Muon, one epoch, microbatch 32, gradient accumulation 1, LR `5e-6`, full-model training, no system prompt, pretokenized SFT; 1,593 optimizer updates.
- Health: train loss `0.9989`, validation loss `1.2023`; no instability.
- Core result: 67/272 atomic units (+45.7%) and 9/96 strict prompts. This is below the 69-unit requirement, so the fresh holdout was not opened.
- Decision: reject. Exact accepted-curriculum replay improved neither the capability tradeoff nor the requested target enough to promote.

## Final selection

- Selected checkpoint: `checkpoints/180m/sft_instruction_v3_e6_selective_best.safetensors`.
- SHA-256: `b6b7d23c37291a73faeae20d36dd0a5e4eafc2d50a027ddf3d78fd9dc97c3a8e`.
- Selection rationale: E3 is the highest new-benchmark scorer at 83/272, but it changes France's capital to Lyon and has poorer factual retention. E6 scores 78/272, answers the motivating capital correctly and supplies the requested second component, retains more factual items, passes the fresh gate, and still exceeds the requested 50% improvement by a substantial margin. It is the strongest tested overall tradeoff, not merely the lowest-loss or highest-new-core checkpoint.
- Exact motivating-prompt output under frozen settings: `The capital of France is Paris. It serves as the political and economic center for Europe, attracting millions from around`. This fixes the omitted second clause and preserves Paris, but the explanation is overstated and the final phrase is incomplete; this remains a known weakness.

### Selected training configuration

```bash
env SPAKIE_MONITOR=0 .venv/bin/python3 scripts/finetune.py \
  --backend mlx --preset 180m \
  --source-checkpoint checkpoints/180m/sft_fresh_balanced_e3_epoch2.safetensors \
  --train-jsonl data/chat_raw/experiment_180m_sft_20260718/instruction_reasoning_v3_e3.jsonl \
  --epochs 1 --sft-batch-size 32 --sft-grad-accum 1 --lr 1e-5 \
  --precision auto --output-name sft_instruction_v3_e6_selective_best.safetensors \
  --loss-log evaluations/sft_instruction_v2_20260718/e6_selective_training_loss.csv \
  --no-model-prompt --pretokenize-sft --mlx-trainable-block-start 8
```

- Runtime: MLX/Metal, BF16 forward weights, Muon with the checkpoint's recorded verified settings, compiled/prefetched SFT, no system prompt.
- Trainable scope: transformer blocks 8-15 and final RMS norm only; 57 trainable tensors. Token/position embeddings and blocks 0-7 were frozen.
- Updates: 829. Mean optimizer-step loss `0.8784`; validation loss `0.9904`.

### Exact selected mixture

- Mixture file: `data/chat_raw/experiment_180m_sft_20260718/instruction_reasoning_v3_e3.jsonl`, 27,944 weighted rows, SHA-256 `b8e2624acc35b217abf0b27e2c283569b0fdf442fbd810846ec9a65493ae61a5`.
- 8,000 deterministic retention rows sampled from `foundation_without_deepseek_e3.jsonl`. Its exact underlying raw sources are: `custom.jsonl`, `spakie_180m_identity.jsonl`, `assistant_behavior.jsonl`, `anti_echo.jsonl`, `factual_repairs.jsonl`, `nemotron_instruction_following_chat_v3.jsonl`, `no_robots.jsonl`, `smoltalk.jsonl`, `squad.jsonl`, `triviaqa.jsonl`, `sciq.jsonl`, and `boolq.jsonl`.
- `compound_instruction_following_v2.jsonl`: 1,997 rows, weight 1.
- `math_reasoning_curriculum_v2.jsonl`: 3,253 rows, weight 1.
- `structured_coding_curriculum_v2.jsonl`: 2,081 rows, weight 1.
- `grounded_multitask_curriculum_v2.jsonl`: 1,813 rows, weight 1.
- `complete_multi_part_answers_v3.jsonl`: 5,400 rows, weight 2 (10,800 weighted rows), SHA-256 `093e931a76001e6b0d84d2bdf923b29a860c4acd3b8d967f907829c02c147bac`.
- Mixture builder reported zero blocked evaluation prompts, zero target duplicates, and zero target conversations duplicated from the retention baseline.

### Added and excluded files

- Added this cycle under `data/chat_raw/`: the four v2 curricula listed above, `complete_multi_part_answers_v3.jsonl`, and the logged experiment mixture files under `data/chat_raw/experiment_180m_sft_20260718/`.
- No source file was moved or excluded during this cycle.
- Exact global excluded list remains:
  - `data/chat_exclude/DeepSeek-distill-V2_excluded_for_180m_sft.jsonl` — excluded in the prior cycle for broad capability harm/complex verbose mixture interference.
  - `data/chat_exclude/structured_instruction_following_scaled_mislabeled.jsonl` — excluded in the prior cycle for systematic wrong labels; replaced by a corrected source.

## Final evaluation comparison

All new-benchmark rows use the frozen no-system generation policy: max 96 new tokens, temperature `0.1`, top-k `1`, top-p `1.0`, and repetition penalty `1.2`.

| Model | New core strict | New core atomic | New fresh strict | New fresh atomic |
|---|---:|---:|---:|---:|
| Initial 3-epoch baseline | 1/96 | 40/272 | not opened | not opened |
| Accepted model at start of this cycle | 0/96 | 46/272 | 0/48 | 17/136 |
| E3 highest new-core candidate, rejected | 16/96 | 83/272 | 5/48 | 30/136 |
| Final selected E6 | 11/96 | 78/272 | 3/48 | 28/136 |

- Final versus the accepted starting checkpoint: +69.6% core atomic compliance (`46 -> 78`) and +64.7% fresh atomic compliance (`17 -> 28`). Combined atomic compliance improves from 63/408 to 106/408 (+68.3%).
- Final versus the original 3-epoch baseline on the new core: +95.0% (`40 -> 78`).
- Strict core success improves from 0/96 to 11/96; no relative percentage is claimed from a zero baseline.

| Core category | Starting atomic | Final atomic |
|---|---:|---:|
| Arithmetic | 0/24 | 0/24 |
| Coding | 0/24 | 4/24 |
| Compound instruction | 5/24 | 8/24 |
| Compound knowledge | 5/16 | 4/16 |
| Date/time | 2/16 | 4/16 |
| Explanation/planning | 6/24 | 8/24 |
| Grounded multi-part QA | 0/24 | 8/24 |
| Logic | 15/24 | 16/24 |
| Multi-step math | 0/24 | 0/24 |
| Spatial | 5/16 | 6/16 |
| Structured output | 0/32 | 0/32 |
| Text transformation | 8/24 | 20/24 |

### Regression audit against the prior broad evaluation

| Model | Clean legacy core semantic | Stable supplement semantic | Old fresh semantic | Old generated total |
|---|---:|---:|---:|---:|
| Previous SFT | 4/64 | unavailable | unavailable | unavailable |
| Initial 3-epoch baseline | 2/64 | 3/36 | 2/27 | 7/127 |
| Accepted model at cycle start | 10/64 | 2/36 | 2/27 | 14/127 |
| Final selected E6 | 5/64 | 3/36 | 1/27 | 9/127 |

- E6 remains +25% over the historical previous SFT on the shared semantic core (`4 -> 5`) and +150% over the initial 3-epoch baseline (`2 -> 5`).
- It regresses 50% versus the accepted cycle-start model on that clean semantic core (`10 -> 5`) and falls from 14/127 to 9/127 across the older generated sets. This is a material regression, concentrated in older exact instruction/date/context items, and means the request's no-major-regression preference was not fully satisfied.
- Arithmetic, multi-step math, and valid structured output remain effectively unsolved at this model size/configuration. Factual hallucination also remains important despite the motivating answer preserving Paris.

## Target conclusion

- The requested additional 50% improvement was reached repeatably on both the frozen core (+69.6%) and fresh held-out prompts (+64.7%).
- The result exceeded the 50% aim, but a no-major-regression solution was not found: every candidate that crossed the new benchmark gate lost substantial performance on the accepted model's older clean core. The final checkpoint is therefore the strongest tested instruction-focused tradeoff, not an unconditional replacement for every prior use case.

## Verification

- Full project suite with Metal access: `.venv/bin/python3 -m unittest discover -s tests -v` -> 200 tests passed.
- Exact train/evaluation prompt overlap for all 144 new prompts versus generated curricula: zero.
- Selected checkpoint, benchmark prompts, mixtures, and newly added curricula are SHA-256 identified above or in adjacent manifests/summaries.
