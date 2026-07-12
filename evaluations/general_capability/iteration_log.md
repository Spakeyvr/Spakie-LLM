# 180M general-capability improvement log

## Baseline

- Checkpoint: `checkpoints/180m/sft_best.safetensors`
- Prompt suite: 46 held-out prompts across 18 categories
- Machine-scored result: 5/46
- Relative strength: basic factual recall (3/4)
- Weak classes:
  - quantitative and multi-step reasoning, including fractions, proportions, elapsed time, and operation order
  - strict instruction following, especially exact length, exact format, and answer-only constraints
  - structured output such as valid JSON, Markdown tables, and labeled classifications
  - language transformation, grammar correction, passive voice, and length-bounded summarization
  - causal science and physical commonsense explanations
  - ambiguity detection and calibrated clarification instead of invented specifics
  - date, spatial, set/negation, ordering, and edge-case reasoning
  - elementary code syntax and execution semantics

## Iteration 1 curriculum

The following generalized batches were added under `data/raw_chat/`. None copies
an evaluation prompt or answer pair.

- `quantitative_reasoning_foundations.jsonl`: arithmetic precedence, proportions, percentages, fractions, elapsed time, and multi-step word problems
- `constraint_following_structured_output.jsonl`: exact-count responses, answer-only constraints, JSON, tables, and labeled formats
- `language_transformation_summarization.jsonl`: grammar repair, voice conversion, paraphrase, and bounded summaries
- `science_commonsense_causality.jsonl`: density, heat, weather, electricity, biology, and everyday physical consequences
- `calibration_logic_spatial_dates.jsonl`: clarification, uncertainty, calendar arithmetic, directions, negation, classification, and boundary cases
- `coding_semantics_basics.jsonl`: basic Python syntax, tracing, conditionals, and small functions

### Iteration 1 result

- Checkpoint: `checkpoints/180m/general_capability_iter1_best.safetensors`
- Score: 7/46, up from 5/46
- Gains: grammar correction, everyday commonsense, and comparison formatting
- Regression: the seasons explanation lost its required causal wording
- Remaining issue: the 79-example curriculum produced only five optimizer steps per epoch with the first batch configuration, so most new patterns did not receive enough update signal

## Iteration 2 curriculum and training adjustment

- Added fresh held-out variants for quantitative reasoning, exact/structured output, logic/calendar reasoning, science causality, concise language transformations, and Python semantics.
- Kept the required two epochs, but changed to batch size 1 and accumulation 1 so each curriculum example contributes an optimizer step.
- Iteration 2 trains on all iteration-1 and iteration-2 capability files together.

### Iteration 2 result

- Checkpoint: `checkpoints/180m/general_capability_iter2_best.safetensors`
- Score: 6/46; rejected because it regressed from iteration 1's 7/46
- Training/validation gap: 0.5938 versus 1.6870, indicating strong curriculum overfit
- Regressions included factual recall, commonsense safety, and comparison behavior
- Decision: preserve iteration 1 as the best candidate and do not continue from iteration 2

## Iteration 3 curriculum and training adjustment

- Added `general_capability_retention_iteration3.jsonl` to reinforce healthy factual, planning, classification, concise-answer, and everyday-assistant behavior.
- Resume from iteration 1, not the rejected iteration 2 checkpoint.
- Use batch size 2, accumulation 1, and learning rate 1e-5 for two epochs: more update signal than iteration 1, but lower-noise and lower-rate than the overfit iteration 2 run.

### Iteration 3 result

- Checkpoint: `checkpoints/180m/general_capability_iter3_best.safetensors`
- Score: 6/46; rejected
- Positive retention: factual recall reached 4/4
- No generalization in the main reasoning, formatting, date/spatial, coding, or ambiguity classes
- Decision: targeted-only curricula are insufficient for this checkpoint; iteration 4 blends targeted examples with broad original SFT retention data

## Iteration 4 curriculum

- Blend all validated generalized capability examples with a deterministic 1,000-example sample from `data/chat/train.jsonl`.
- Deduplicate by full conversation content and shuffle deterministically.
- Train from the original baseline checkpoint for two epochs at learning rate 1e-5, batch size 8, accumulation 1.

### Iteration 4 result

- Checkpoint: `checkpoints/180m/general_capability_iter4_best.safetensors`
- Score: 7/46, tied with iteration 1 but not broadly improved
- Passing classes: factual recall 3/4, commonsense 2/3, science explanation 1/3, grammar correction 1/3
- Still 0: arithmetic, multi-step reasoning, instruction constraints, structured output, coding, ambiguity calibration, date reasoning, spatial reasoning, negation, planning, and summarization
- The broad-retention blend prevented the severe iteration-2 overfit but did not unlock the missing reasoning/format skills

## Current conclusion

No iteration satisfies the promotion criterion of broad category improvement.
`checkpoints/180m/sft_best.safetensors` remains untouched. Iteration 1 and
iteration 4 are the highest-scoring candidates at 7/46, but neither should
replace the baseline because the gains are narrow and accompanied by regressions.

The repeated failure across under-update, over-update, lower-rate, retention,
and blended curricula indicates that further small corrective SFT batches are
unlikely to solve the checkpoint's general-capability deficit. The next viable
experiment is a fresh SFT run from the strongest completed pretraining checkpoint
with a substantially larger, quality-filtered general-assistant curriculum, then
use this same held-out suite as the promotion gate.

## Full-data fresh SFT iteration

- Source: `checkpoints/180m/pretrain_best.safetensors` (step 51,500, validation loss 2.5883)
- Curriculum: 87,945 unique conversations: the full 87,789-row broad SFT set plus 158 generalized repairs
- Training: exactly two epochs, batch 64, accumulation 4, learning rate 3e-5
- Epoch 1: train loss 2.4374, validation loss 2.0995
- Epoch 2: train loss 2.1329, validation loss 2.0618
- Held-out score: 8/46, the best so far, adding summarization but still leaving the central reasoning and formatting classes at zero
- Diagnosis: targeted repairs were only 0.18% of the full curriculum, too sparse to affect the weak classes

## Scaled weak-class iteration

- Added deterministic, varied scaled batches for arithmetic, structured instruction following, logic/date/spatial/calibration, language transformation, and Python semantics.
- After filtering and deduplication: 614 targeted examples.
- Blended with 1,000 broad-retention examples for a 1,614-example curriculum (38% targeted, 62% retention).
- Source checkpoint: `checkpoints/180m/general_capability_full_best.safetensors`.

### Scaled weak-class result

- Checkpoint: `checkpoints/180m/general_capability_scaled_best.safetensors`
- Score: 10/46, up from baseline 5/46 and full-data 8/46
- Category gains versus baseline: multi-step reasoning 0→1, language 0→2, science explanation 1→2, edge cases 0→1
- No category-level score decreased versus baseline
- Manual caveat: several accepted answers still contain incorrect or confused supporting text, so this is the leading candidate but not yet sufficient evidence for final promotion

## Consolidation iteration

- Expanded generalized prompt phrasings to six variants per generated task.
- Added `commonsense_summarization_retention_consolidation.jsonl` to restore safety, physical commonsense, concise summarization, classification, comparison, and negation behavior.
- Curriculum: 1,085 targeted examples plus 1,000 broad-retention examples.
- Source: scaled 10/46 candidate; two epochs at learning rate 5e-6.
- Result: 9/46. Commonsense recovered to 2/3, but multi-step, language, and science gains regressed.
- Decision: reject consolidation checkpoint and retain the scaled checkpoint as the best candidate.

## Semantic audit and stopping decision

Manual review of every machine-accepted answer showed that the keyword scorer
substantially overstated all checkpoint scores. Representative false positives:

- edge-case answer asserted that an empty string contains a space, but passed because it mentioned `empty`, `space`, and `no`
- toaster-safety answers ambiguously or incorrectly endorsed a metal fork, but passed because they also contained `unplug` and `safe`
- science answers mentioned `tilt`, `gravity`, or `mass` while giving incorrect causal explanations
- ambiguity answers invented a team or game and passed merely because a keyword such as `team` appeared
- multi-step logic sometimes began with the right yes/no token but contradicted the premise in its explanation

Therefore, the numeric 10/46 result is not a valid promotion result. The only
reliably retained strength is short basic factual recall; grammar correction is
occasionally correct. Arithmetic, multi-step calculation, exact formatting,
JSON, coding, date/spatial reasoning, calibration, concise summarization, and
causal commonsense remain broadly unreliable.

Seven iterations covered:

1. compact generalized curriculum with too few updates
2. high-update targeted curriculum, rejected for destructive overfit
3. lower-rate retention curriculum
4. 1,000-row broad blend
5. fresh two-epoch SFT over the full 87,945-row dataset
6. scaled 614-example weak-class blend
7. 1,085-example consolidation curriculum at 5e-6

None produced semantically broad held-out improvement without regressions. The
original `sft_best.safetensors` remains untouched and no candidate is promoted.
Further progress requires an external-state change: a materially stronger
pretraining checkpoint/model capacity, or a new pretraining run whose base
representations support these reasoning and structured-output skills. Additional
small SFT variations on the current weights are no longer evidence-backed.

## QA expansion experiments

The blocked goal was resumed to test whether materially more QA supervision
could unlock general capability.

### Local QA audit

- TriviaQA was already fully present in `data/chat/train.jsonl` (3,000/3,000 exact overlap).
- SciQ was already present (5,999/6,000 overlap), but inspection found at least one clearly incorrect label (`inductive`-reasoning question answered as `deductive`), so 5,978 exact SciQ conversations were excluded from new retention samples.
- SQuAD and BoolQ had zero exact overlap and supplied 7,952 usable new examples after language and length filtering.

### Extractive/context QA iteration

- New raw batches:
  - `qa_reading_comprehension_squad.jsonl`
  - `qa_boolean_context_reasoning_boolq.jsonl`
- Curriculum: 9,037 targeted/generalized rows plus 20,000 broad-retention rows; 27,106 unique conversations after deduplication.
- Training: original `sft_best`, exactly two epochs, learning rate 1e-5.
- Best validation loss: 1.8920, improved from the original checkpoint metadata value of 2.0146.
- Held-out result: the original general suite regressed to 6/46 machine-scored; a new ten-question context-QA slice remained about 4/10 on manual semantic review, essentially unchanged from baseline.
- Decision: reject; lower validation loss did not transfer to general or unseen context-QA capability.

### Science/commonsense reasoning QA iteration

- Downloaded full official training splits:
  - ARC Challenge: 1,119 rows
  - OpenBookQA: 4,957 rows
- New raw batches:
  - `qa_science_reasoning_arc_challenge.jsonl`
  - `qa_science_commonsense_openbookqa.jsonl`
- After filtering: 6,045 reasoning-QA examples and 1,079 generalized/retention examples; blended with 20,000 broad rows for 27,124 unique conversations.
- Training: original `sft_best`, exactly two epochs, learning rate 1e-5.
- Best validation loss: 1.8785.
- Added a ten-question held-out multiple-choice reasoning slice with no copied training questions.
- Manual result: baseline and reasoning-QA checkpoint were both about 1/10 on multiple-choice reasoning; context QA stayed about 4/10; original arithmetic, coding, formatting, date/spatial, and causal-science classes remained unreliable.
- Decision: reject; the model mostly chose option A or repeated prompt fragments and did not learn the QA reasoning pattern.

Both QA experiments confirm that lower in-distribution validation loss is not
evidence of broader capability for this checkpoint. No checkpoint was promoted,
and `checkpoints/180m/sft_best.safetensors` remains untouched.

## Novel broad general-assistant expansion

After QA-specific datasets failed to transfer, the remaining large SFT lever was
the unused portion of the local SmolTalk pool.

- Raw SmolTalk pool: 149,983 unique conversations.
- After the repository's system/identity/tool/refusal/language and 512-token filters: 46,411 usable conversations.
- Exact conversations already present in `data/chat/train.jsonl`: 30,031.
- Genuinely unseen usable conversations: 16,380.
- New raw batch: `general_assistant_instruction_qa_smoltalk_expansion.jsonl`.
- Final curriculum: 16,380 novel conversations plus 20,000 clean broad-retention conversations; 36,380 unique rows.
- Training: original `sft_best`, exactly two epochs, batch 64, accumulation 4, learning rate 1e-5.
- Epoch-1 validation loss: 1.5741.
- Epoch-2 validation loss: 1.5686, the strongest in-distribution result of all iterations.
- Expanded held-out result: 13/66 mechanically, but manual semantic review found no broad improvement. Apparent passes again included contradictions, wrong causal explanations, or answers containing both correct and incorrect options.
- Genuine retained skills remained limited to short factual recall, one grammar correction pattern, a few simple context lookups, and an isolated multiple-choice item.
- Decision: reject and do not promote.

This experiment rules out the hypothesis that the previous failures were simply
caused by too little unseen general-assistant SFT data. Across extractive QA,
reasoning QA, and 16,380 novel broad conversations, validation loss improved
dramatically while held-out semantic capability did not. The limiting factor is
now the pretrained representation/model, not SFT dataset volume or source mix.
