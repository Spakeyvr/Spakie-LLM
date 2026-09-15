# Spakie inference investigation — 14 September 2026

## Main conclusion

The current checkpoints have demonstrably weak factual knowledge, arithmetic, and instruction following. There is also a concrete numerical problem: large negative output logits lose substantial resolution in BF16 and interact badly with the chat repetition penalty. This warrants investigation before spending substantially more compute. The evidence does not establish that architecture or corpus percentages are the primary cause.

## Scope and reproducibility

- Real local MLX/Metal inference with strict model loading and checkpoint/tokenizer compatibility validation.
- `pretrain_interrupt.safetensors`: 94,392,576 model parameters, BF16 stored weights, step 61,338, 8,039,694,336 tokens processed toward 8.8B.
- `sft_interrupt.safetensors`: step 219, epoch index 0, microbatch offset 876, training dataset size 207,907, batch 16, accumulation 4. Approximately 14,016 examples processed, or 6.7% of one epoch. This is an unfinished SFT checkpoint, predating the latest pretrain checkpoint; comparisons are not a controlled before/after SFT experiment.
- Checkpoint metadata and SHA-256 hashes are saved alongside this report.
- 16 diagnostic cases, both checkpoints, raw continuation and no-system chat, repetition penalties 1.0 and 1.2: 128 generations, greedy, 64-token cap. Cases test facts, arithmetic, context use, ordering, code, format following, uncertainty, and prose.
- 12 additional cache-free generation controls in BF16 and float32, with 16-token caps.
- 36 additional generations over six cases and three seeds (14, 29, 71), using production defaults: base continuation temperature .8/top-k 50/top-p .9/penalty 1; SFT chat .1/1/1/1.2. Top-k 1 can retain multiple equal logits; these settings are not necessarily deterministic when BF16 produces ties.
- These are targeted diagnostic probes, not 176 independent benchmark questions. They were newly assembled for this investigation but not exhaustively decontaminated against training data.
- Rerun `run.py`, then `numerics.py`, then `sampling.py` with the repository virtualenv and Metal access. `controls.jsonl` appends; remove that diagnostic output before a complete rerun if deduplication is needed. Scripts catch Ctrl+C and preserve completed JSONL rows.

## 1. Confirmed numerical degradation

For four prompts, full float32 output logits were approximately -210 to -278. Casting/computing the model in BF16 left only 31–42 unique output values across 24,576 vocabulary entries; float32 retained over 24,000. For “The capital of France is”, five BF16 logits tied for the maximum and the top 20 logits occupied only three values. Water's top-token choice changed between BF16 and float32.

The common negative offset does not itself change softmax in exact arithmetic. However, BF16 spacing at these magnitudes is approximately 1–2 logit units, so meaningful relative distinctions disappear. Casting the already-computed BF16 logits to float32 during sampling cannot restore them.

The sign-aware repetition penalty multiplies negative logits by 1.2. A repeated token at -230 therefore loses 46 logit units, even though top-token differences can be below 1. This makes the chat penalty extremely aggressive for these weights. The base continuation default disables it; this issue cannot explain all base failures.

The MLX custom training loss also computes the head and softmax in BF16 for BF16 inputs, promoting only float16. This is a concrete path to audit, not proof that it caused the checkpoint's offset. No gradient experiment or retraining has established the historical cause. Priorities are accurate output projection/loss numerics and a decoding penalty calibrated to actual logits. Increasing training tokens before resolving this would confound the diagnosis.

## 2. Capability failures persist beyond decoding controls

Representative greedy outputs:

| Probe | Base, raw continuation | SFT, no-system chat |
|---|---|---|
| France's capital | “the city of Aubusson” | Paris; some variants add false geography |
| Japan's capital | Kyoto | Osaka |
| Water's formula | repeated zeros / invented chemistry | “water” |
| 17 + 26 | 11 or 10 | repeats “17 + 26” |
| Largest planet | Pluto | Earth |
| Uppercase lantern | unrelated prose | a sentence about “lemon” |

The base's float32 cache-free outputs still call France's capital Chartres, invent water chemistry, and produce 100 for 17 + 26. Changing sampling seeds also fails the basic fact/math probes. Therefore BF16, the cache, repetition penalty, and chat-template choice are not sufficient explanations.

There are narrow successes: extracting the green key's color, repeating the fictional capital Luma, and SFT recalling Paris. The model often continues beyond correct details into unsupported claims. It has learned recognizable language patterns without reliable factual or task execution behavior on these probes.

## 3. Runtime controls

The actual base checkpoint was also loaded strictly into Torch CPU float32. For four prompts, Torch and MLX selected the same next token; maximum absolute logit differences were .033–.112. MLX cached versus full-prefix float32 checks also selected the same next token, with maximum differences .086–.104. BF16 differences were larger (up to 2 in the single diagnostic cache probe). These checks reduce the likelihood of a gross backend/masking error but do not prove full-sequence parity or rule out close-ranking differences.

Fresh explicit-float32 CE on the same 16 uniformly sampled 512-token validation windows (seed 1409): base 2.6699, older SFT 2.9006. This is only 8,192 validation tokens and not the same sequence length/sampling as the training monitor. It confirms meaningful next-token learning; it does not imply broad capability, nor does the SFT/base loss difference isolate SFT damage.

## 4. Data evidence

Current `data/chat/train.jsonl`: 218,849 rows, 218,849 distinct exact message arrays; no exact duplicate conversations. Median total message length 818 characters. This does not measure semantic duplication, answer accuracy, or source balance.

FineMath spot audit: first 100 rows from each of 10 evenly spaced shards, 1,000 rows total, current heuristic filter only. Results: 508 kept, 428 rejected for duplicate 5-grams, 58 repeated lines, 5 link farms, 1 symbol-heavy. This is a systematic convenience sample, not a random estimate of corpus quality.

Inspected rejected samples are mixed: useful worked trigonometry and educational explanation are rejected alongside repeated question listings and noisy pages. This supports examining false positives and cleaning useful text before filtering, but does not justify disabling the filter or calling all 428 examples valuable. The existing corpus's math share is 9.9% versus a planned 15%; there is no controlled mix experiment demonstrating that raising it improves this model.

## Recommended order

1. Investigate and correct head/loss precision and repetition-penalty interaction; rerun these exact probes and held-out loss. Do not claim that fixes will restore knowledge already absent from the weights.
2. Evaluate a completed SFT run from a clearly recorded base checkpoint using fresh held-out tasks. The available SFT interrupt file is not evidence of the model's finished instruction-tuning potential.
3. Test targeted data-quality changes at matched compute, including rejection false positives and simple accurate worked examples.
4. Decide on more tokens or a larger model from those results. Architecture redesign is not supported as the first move by this investigation.

No production code, training data, or checkpoints were modified. Only diagnostic scripts and evidence artifacts were added. This was not a full repository test run or an architecture ablation.
