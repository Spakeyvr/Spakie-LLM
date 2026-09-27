# Spakie-LLM

Spakie-LLM is a GPT-style language model project with parallel PyTorch and MLX runtime paths. It includes tokenizer training, corpus download/scraping and preprocessing, pretraining, SFT fine-tuning, checkpointed chat inference, and basic evaluation tooling from the same codebase.

## Setup

### Numerical precision and recovery runs

Both backends compute the output projection and cross-entropy in float32,
including when transformer weights/activations use BF16. Promoting before the
projection preserves token-score differences when trained logits have a large
common offset. This uses more head compute and memory than BF16 projection.
The shared repetition penalty divides repeated tokens' unnormalized probability
by the specified factor (`1.2` means a factor of `1/1.2`), once per unique token.
It no longer depends on the sign or absolute offset of logits.

Use `scripts/finetune.py --output-dir checkpoints/92m/my_run` to isolate a fresh
SFT run's checkpoints and status. Specify `--source-checkpoint` explicitly when
comparing runs. An interrupted checkpoint is not a completed SFT result; compare
held-out answers as well as validation loss after a full epoch.

Data preparation counts overlapping repeated five-word spans only once (since schema 5).
Existing arrays remain unchanged. New filtering requires a fresh preparation
run; old partial shards must not be mixed with the revised filtering contract.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is the main dependency for the torch backend. On Apple Silicon, MLX is installed via the conditional `mlx>=0.31.1` requirement; on CUDA machines, install a PyTorch wheel that matches your CUDA toolkit if the default wheel is not appropriate.

## Project Layout

- `scripts/` - training, preprocessing, scraping, data download, evaluation, benchmark, and pipeline entry points
- `training/` - dataset, optimizer, pretraining, and fine-tuning logic
- `model/` - PyTorch and MLX Transformer implementations
- `runtime/` - device, precision, and MLX runtime helpers
- `tokenizer/` - SentencePiece tokenizer training and wrapper code
- `inference/` - chat and generation loops for both backends
- `configs/default.py` and `configs/default.yaml` - preset definitions, paths, optimizer defaults, and corpus planning defaults
- `tests/` - `unittest` coverage for runtime resolution, config scaling, Muon, and MLX/PyTorch parity

## Quick Start

1. Download or add pretraining data:

```bash
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only
```

2. Train the 24K tokenizer from cleaned files under `data/raw/`:

```bash
python3 tokenizer/train_tokenizer.py
```

Tokenizer samples pass the canonical domain-aware filters and are streamed in
deterministic weighted-fair order using the final corpus plan. The five-million
sample cap therefore follows the intended mix instead of lexical path order.
Samples retain indentation, tabs, and line breaks through JSONL spooling and
SentencePiece's iterator interface; repeated-space removal is disabled. These
settings apply to newly trained tokenizers. Existing checkpoints must keep their
original tokenizer: changing it requires new token arrays and a separately
planned training or migration experiment.
The inspection-only `.vocab` file escapes control characters using JSON string
escapes (without surrounding quotes), so every token occupies one TSV row;
the `.model` file retains the actual characters used by both model backends.
Python-Edu downloading and preparation preserve string contents, including
HTML tags, navigation-like lines, trailing spaces, and blank lines. Only source
line endings are normalized. Preparation schema 6 rejects partial shards from
older cleaning rules. Existing downloaded code may already have lost whitespace;
resuming downloads preserves those old records, so a fully corrected fresh
corpus requires fresh code downloads into a separate directory. Existing model
checkpoints and processed arrays are not rewritten by this change.
Tokenizer training and preparation reject downloader metadata that identifies
retained output from the old code cleaner, even after its cursor is migrated.
Use the downloader's `--output-dir` for a separate fresh corpus. The tokenizer
accepts `--raw-data-dir` and `--output-prefix`; preparation accepts
`--raw-data-dir`, `--tokenizer-prefix`, and `--output-dir` (which keeps its
arrays, shards, and report together). Training accepts `--tokenizer-prefix`
and `--processed-data-dir`, alongside its existing checkpoint `--output-dir`.

3. Prepare fresh pretraining arrays:

```bash
python3 scripts/prepare_data.py
```

4. Train a model:

```bash
python3 scripts/train.py --preset 300m --backend mlx --precision auto
python3 scripts/train.py --preset 300m --backend torch --device auto --precision auto
```

For approximately 360M parameters, select `--preset 360m`. It has 362,869,248
parameters with the canonical vocabulary, 2,048-token context, microbatch 8,
and six accumulation steps (98,304 tokens per update), accumulated
sequentially. The starting recipe retains Muon with ten
Newton–Schulz steps, a `6e-4` peak learning rate, and the trapezoid schedule.
This is a conservative starting point; short pilots do not establish the best
full-run schedule or eventual capabilities.

Before a fresh long run, rebuild any lossy raw code, train the tokenizer from
the corrected source-balanced corpus, and prepare arrays with the current
provenance and quality gates. Keep these assets separate from old checkpoints.
Then use `--smoke --max-steps 12` with the intended preset and isolated output
directory to check memory and throughput on the actual machine. Test Ctrl+C and resume
separately, and check the reported peak memory rather than assuming the requested
MLX memory limit is a hard process-memory cap.
Estimate the full token budget from that measured throughput before launching.
Compare held-out BASE facts, language, and arithmetic as training progresses;
evaluate instruction following again after a separately validated SFT stage.

5. Build SFT data and fine-tune:

```bash
python3 scripts/download_sft_data.py
python3 scripts/build_sft_seed_data.py
python3 scripts/prepare_sft.py
python3 scripts/finetune.py --backend mlx --precision auto
```

6. Chat with a checkpoint:

```bash
python3 scripts/chat.py --backend mlx --precision auto
python3 scripts/chat.py --backend torch --device auto --precision auto
```

`scripts/train.py` defaults to `--backend mlx --preset 92m`. The shared config default preset and pipeline default are `300m`.

## Runtime Selection

Torch entry points support:

```bash
--device {auto,cuda,mps,cpu}
--precision {auto,fp32,fp16,bf16}
```

Default runtime behavior:

- `--device auto` prefers `cuda`, then `mps`, then `cpu`
- `--precision auto` resolves to `bf16` on CUDA, `bf16` on MPS, and `fp32` on CPU

Common MLX runtime flags:

```bash
--mlx-compile / --no-mlx-compile
--mlx-prefetch / --no-mlx-prefetch
--mlx-memory-gb <value>
--mlx-wired-gb <value>
--mlx-cache-gb <value>
--mlx-profile
```

## Data Sources

You can add local `.md`, `.txt`, and `.jsonl` files under `data/raw/`, or use the built-in download and scrape scripts:

```bash
python3 scripts/scrape_wiki.py
python3 scripts/scrape_dictionary.py --max 5000
python3 scripts/scrape_open_corpus.py
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only
```

The downloader streams Gutenberg, Stack Exchange, and arXiv from bulk corpus
snapshots rather than rate-limited public APIs. Python-Edu uses a pinned
pre-materialized Parquet copy of the original corpus, avoiding millions of
individual Software Heritage requests. Migrating an older `--resume` run keeps
committed output and token accounting, resets only the obsolete input cursor,
and skips replayed documents through their blob IDs. By default, each Hugging
Face source keeps four input shards active
(`--hf-workers-per-source`). Hugging Face transfers use a 60-second timeout
unless `HF_HUB_DOWNLOAD_TIMEOUT` is already set. A failed source is resumed
from its durable cursor up to three times (`--source-retries`). New progress
files store exact Hugging Face stream state, so `--resume` continues at the
saved input shard instead of replaying every earlier row. Near-complete progress
files from the old row-counter format
skip their multi-million-row replay and fill the small remaining tail from a
source with a direct cursor. Progress is labelled `Accepted corpus`; the
displayed rate is accepted estimated tokens over the last 15 seconds, so
retries, filtering, and other zero-progress time reduce it instead of leaving
an earlier burst rate on screen. If a requested source is exhausted or
unavailable, its shortfall is filled from another requested streaming source;
use `--no-redistribute-shortfall` to preserve strict per-source quotas instead.
Ctrl+C gives active workers five seconds to flush their checkpoints, then exits
without waiting for blocked HTTP retries; pressing Ctrl+C again skips the grace
period. Sources already at their saved target are skipped before their
potentially large resume indexes are loaded.

`scripts/prepare_data.py` streams documents from `data/raw/`, including `data/raw/large_corpus/<source>/`, applies separate prose/math/code filters, source-appropriate language ID, and MinHash/LSH near-deduplication, tokenizes in deterministic input order, writes token shards under `data/processed/shards/`, and transactionally merges them into `data/processed/train.npy` and `data/processed/val.npy`. A `processed_data_manifest.json` commit marker is published only after both arrays are complete and durable. Full-corpus runs enforce configured limits on total corpus completion, source-kind coverage and mix, and the share of unplanned sources. Individual source shortfalls and source-mix deviations are recorded as warnings in `corpus_report.json`; they do not independently block publication. Inspect its actual token counts and warnings before selecting a training recipe—a passing gate does not guarantee the planned source proportions. It records the exact tokenizer, preparation settings, raw-input generation, token-ID bounds, and array file identity; training refuses stale or unverifiable arrays by default.

Useful prepare commands:

```bash
python3 scripts/prepare_data.py --resume
python3 scripts/prepare_data.py --dry_run
python3 scripts/prepare_data.py --target_train_tokens 100000000
python3 scripts/prepare_data.py --source_dirs large_corpus,wiki
python3 scripts/prepare_data.py --workers 1
# Diagnostic/ablation escape hatch; do not use for a canonical full run:
python3 scripts/prepare_data.py --allow-incomplete-corpus
```

Resume requires the shard-generation provenance and accepted-document journal
from the interrupted run. A compatible resume verifies the raw prefix while
skipping repeated SentencePiece and MinHash work for accepted documents;
changed raw files, tokenizer, or preparation settings are rejected instead of
being mixed into old shards.

## Pretraining and SFT

Pretraining:

```bash
python3 scripts/train.py --preset 92m --backend mlx --precision auto --smoke
python3 scripts/train.py --preset 300m --backend mlx --precision auto --mlx-profile
python3 scripts/train.py --preset 300m --backend torch --device auto --precision auto
```

Useful training options:

```bash
--max-steps <steps>
--target_tokens <tokens>
--resume
--resume-from <checkpoint>
--additional-steps <steps>
--output-dir <dir>
--eval-interval <steps>
--eval-batches <batches>
--checkpoint-interval <steps>
```

Training writes a live status file to `checkpoints/<preset>/training_status.json`
and automatically starts a background monitor on port `8765`. Without a
password it binds only to `127.0.0.1`. To open it from another device on your
LAN, set `MONITOR_PASSWORD` before training; the authenticated monitor then
binds to the LAN address printed at startup. The page shows step/token progress, loss, throughput, ETA, checkpoint
paths, MLX memory when available, and a prompt box for querying the best
available checkpoint. Pretrain checkpoints use raw continuation mode; SFT
checkpoints use the same chat template path as `scripts/chat.py`. A monitor
process started by training is stopped automatically when that training process
exits.

For a public IPv6 address, wrap the address in square brackets:

```text
http://[2001:db8::1234]:8765
```

Public access still requires the network path to allow inbound port `8765`
(router/firewall/ISP), and password protection does not encrypt plain HTTP.

You can also start the monitor manually:

```bash
python3 scripts/monitor_training.py
```

Use `--status-file <path>` to pin the monitor to a specific run, or
`--checkpoint-dir <dir>` to scan a different checkpoint tree. Set
`SPAKIE_MONITOR=0` to disable training autostart, or `SPAKIE_MONITOR_PORT=9000`
to use a different port. Set `SPAKIE_MONITOR_PROMPT_TIMEOUT=300` if prompt
generation needs more than the default 180 seconds.

For password protection, set the password before starting training:

```bash
MONITOR_PASSWORD="choose-a-long-password" python3 scripts/train.py --backend mlx
```

Manual monitor starts also accept a flag:

```bash
python3 scripts/monitor_training.py --host :: --password "choose-a-long-password"
```

When password protection is enabled, the monitor serves only the login page or a
401 response until the password is accepted. If you expose this through a public
IP, put HTTPS/VPN in front of it; plain HTTP still sends the password over the
network unencrypted.

Pipeline runner:

```bash
python3 scripts/run_pipeline.py --preset 300m --backend mlx --max-steps 1000
python3 scripts/run_pipeline.py --preset 180m --backend torch --device auto --precision auto
python3 scripts/run_pipeline.py --preset 300m --backend mlx --skip-sft
```

Fine-tuning expects chat-style JSONL records like:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Assistant messages may set `"train": false` to remain in the conversation as
context without contributing labels. If omitted, assistant turns are supervised
as before. The Nemotron Chat v3 importer uses this for all but the final target.

System messages are optional. The default SFT merge omits them, which is usually
better for the smaller presets; pass `--system "..."` only when you intend to
train and infer with that extra control turn.

SFT source data is downloaded to `data/chat_raw/` and merged into `data/chat/train.jsonl`:

```bash
python3 scripts/download_sft_data.py
python3 scripts/build_sft_seed_data.py
python3 scripts/prepare_sft.py
python3 scripts/prepare_sft.py --system "Answer clearly and factually."
python3 scripts/finetune.py --backend mlx --precision auto
python3 scripts/finetune.py --backend torch --device auto --precision auto
```

Fine-tuning validates every rendered conversation before training, drops rows
that would require truncating a turn, and caches compact unpadded token arrays by
default (`--no-pretokenize-sft` disables the cache). If interrupted, resume the
exact optimizer/sampler position with `python3 scripts/finetune.py --resume`; an
explicit path can be supplied with `--resume-from PATH`.

The default SFT download uses the enabled sources in `configs/default.yaml`.
The canonical merge is capped at 100,000 examples and uses concise
`smol-smoltalk`, grounded QA/science sources, and small capped Nemotron and
OpenHermes samples. Unconfigured files are disabled by default, so ad-hoc or
benchmark-derived curricula cannot silently enter training. Every exported row
records canonical source provenance, a stable example ID, and rendered token
length. Refusals are accepted only from explicitly configured safety sources.
Limits are applied to usable converted examples rather than raw rows.

`build_sft_seed_data.py` writes the small permanent local sources separately so
they remain easy to inspect and version: `spakie_identity.jsonl`,
`assistant_behavior.jsonl`, `anti_echo.jsonl`, and `factual_repairs.jsonl`.
They live in `data/chat_raw/` and are merged exactly like downloaded sources.
Identity seeds use the size-independent name Spakie and make no parameter-count
claims, so the same data can serve different presets. The historical
`spakie_180m_identity` source is disabled; rerun the seed builder and merge to
replace old identity targets in a new training dataset. Existing datasets and
checkpoints are not rewritten.

For a small targeted SFT/eval set instead of downloaded SFT sources:

```bash
python3 scripts/build_targeted_data.py
```

This writes repeated teaching examples directly to `data/chat/train.jsonl` and
related diagnostics to `data/eval/`. Those diagnostics test the taught material;
their scores do not establish held-out generalization.

Useful SFT options:

```bash
--train-jsonl <path>
--source-checkpoint <checkpoint>
--output-name <filename>
--max-examples <count>
--epochs <count>
--lr <value>
--list-models
--no-model-prompt
```

On MLX, SFT batches are length-bucketed with right-padding bucket trim by
default. This keeps the transformer math dense and unchanged while avoiding a
large amount of padded-token work on chat datasets. The conservative defaults
are `--sft-sampler sortish`, `--sft-bucket-multiple 128`, and no SFT packing or
varlen attention.

The permanent assistant-behavior and identity sources anchor greetings,
Spakie identity questions, direct answers, and simple factual responses so
a lightly fine-tuned model is less likely to continue pretraining-style web
text or invent a human occupation.

## Optimizer

Pretraining defaults to Muon. SFT deliberately defaults to one gentler AdamW
epoch at `1.5e-5` to reduce catastrophic forgetting:

```bash
python3 scripts/train.py --optimizer muon
python3 scripts/train.py --optimizer adamw --allow-adamw-fallback
python3 scripts/verify_muon.py
```

Muon options include `--muon-adjust-lr-fn {match_rms_adamw,original,none}`, `--muon-ns-steps`, `--muon-momentum`, `--muon-nesterov / --no-muon-nesterov`, `--muon-qkv-split / --no-muon-qkv-split`, `--muon-ns-polish-steps`, and `--muon-split-gate-up / --no-muon-split-gate-up`.

Two settings follow the DeepSeek-V4 Muon recipe. `--muon-ns-polish-steps`
(default 2) runs the last iterations of the ten Newton–Schulz steps with the
`(2, -1.5, 0.5)` polish coefficients. The quintic alone leaves singular values
spread across roughly 0.7–1.1; the polish iterations settle them at 1 for the
same number of matrix multiplies. `--muon-split-gate-up` (default off)
orthogonalizes the fused SwiGLU gate and up projections as two independent
matrices instead of one.

In the 360m pilot (seed 42, about 103M tokens per arm), the polish schedule
lowered final validation loss by 0.009, led at all 20 evaluations from step
100, and reached the baseline's final loss with about 1.5% fewer tokens, at the
same throughput. The gate/up split alone did not differ from the baseline, so
it stays off.

The Muon pilot compares these settings against the current recipe. It runs a
paired 2×2 factorial (`baseline`, `paper`, `polish`, `split`): every arm of a
seed shares initialization, data order, schedule, and validation batches.
Runs finish on an evaluation step, completed arms are skipped on re-run, and
interrupted arms resume. The summary reports each arm's final validation-loss
change against the same seed's baseline, the mean change over the last quarter
of evaluations, and token efficiency: the baseline's step count divided by the
step at which the arm first reaches the baseline's final loss.

```bash
python3 scripts/run_muon_pilot.py                     # preview the 360m matrix
python3 scripts/run_muon_pilot.py --execute
python3 scripts/run_muon_pilot.py --summarize --summary-json evaluations/muon_pilot/summary.json
python3 scripts/run_muon_pilot.py --arms baseline paper --seeds 42 7 --execute
```

Before committing to a long pretraining run, preview the fair 100M-token
cosine-vs-trapezoid sweep across three learning rates, then execute it:

```bash
python3 scripts/run_pretrain_ablations.py --preset 300m --backend mlx
python3 scripts/run_pretrain_ablations.py --preset 300m --backend mlx --execute
```

## Checkpoints and Chat

Checkpoints live under `checkpoints/<preset>/`. Smoke-test outputs live under `smoke_pretrain/` and `smoke_sft/` subdirectories. Torch and MLX checkpoints are self-contained and store the exact current configuration schema plus tokenizer identity; pretraining checkpoints also bind resume state to the processed-data generation. Every model load validates the schema, provenance, tensor keys, and shapes. Torch always uses PyTorch's restricted loader, and MLX requires metadata embedded in the safetensors file. Older or incomplete checkpoints are rejected rather than migrated or guessed. `pretrain_interrupt.*` is the rolling resume checkpoint; every successful run also atomically publishes `pretrain_final.*`, even when it ends before an evaluation boundary.

Useful commands:

```bash
python3 scripts/chat.py --list-models --device auto --precision auto
python3 scripts/chat.py --model 1 --backend mlx --no-model-prompt
python3 scripts/chat.py --checkpoint checkpoints/300m/pretrain_final.safetensors --backend mlx
python3 scripts/chat.py --json_mode --system "Answer as JSON."
python3 scripts/finetune.py --list-models --backend mlx
python3 scripts/train.py --resume
python3 scripts/finetune.py --resume
```

The default chat tokenizer path is `tokenizer/spakie.model`.

For pretraining, `--max-steps` and `--additional-steps` also change the horizon
used by the learning-rate schedule. They are not independent early-stop limits:
shortening a resumed run can change its learning rate immediately. A bounded
experiment intended to preserve the original schedule must keep that horizon
fixed and control its stopping budget separately. Record whether optimizer
state, batch size, context length, and schedule were preserved when comparing
continuation runs.

## Evaluation and Tests

For reproducible pretraining comparisons, pass `--seed` to `scripts/train.py`.
It controls fresh model initialization and data sampling in either backend;
saved checkpoint state takes precedence on resume. The ablation runner passes
the same seed to every recipe (default 42); use a separate output root when
repeating the matrix with another seed. Matching seeds does not guarantee
bitwise equivalence across different hardware or backends.
GPU backward kernels can also introduce small numerical differences between
otherwise identical runs on the same machine.

Tokenizer sampling defaults to weighting chunks bounded to 4 KiB. For an experimental
comparison, `tokenizer/train_tokenizer.py --sampling-unit bytes --max-bytes
4194304 --stop-on-source-exhaustion` weights cleaned UTF-8 bytes instead and caps
the fitting sample at 4 MiB. Supply isolated input/output paths as above. Byte
weights approximate corpus token proportions; they are not exact token weights.
The final chunk may be shortened at a UTF-8 boundary. The collection log reports
actual bytes per source and which sources ran out. Without the exhaustion flag,
remaining sources receive the exhausted source's share. These options do not
establish a better vocabulary; compare held-out byte-normalized prediction,
roundtrip fidelity and downstream capabilities before selecting one.

For lightweight, from-scratch MLX recipe diagnostics:

```bash
python3 scripts/probe_pretrain_recipe.py prepare --assets /path/to/read-only-assets --output evaluations/pilot_data
python3 scripts/probe_pretrain_recipe.py run --assets /path/to/read-only-assets --data evaluations/pilot_data --output evaluations/pilot_seed17 --seed 17
```

The probe uses separate raw shards for training/development/test samples from
five sources, a vocabulary-preserving whitespace normalization copy, and an
18.9M-parameter model by default. It caps runs at 512 updates and 180 training
seconds. `--scale 360m --steps 64` tests a 362.9M-parameter configuration with
short contexts and small batches, not the full production training setup.
Use `--independent-source-rng` when comparing different tokenizations or corpus
lengths. It separates source selection from window-offset randomness, keeping
source draws matched at the same seed and weights when offset ranges differ.
The default retains the legacy
random stream for replaying earlier probes; mixing these modes is not matched.
`--final-test` additionally evaluates the separate test split; keep it unused
while choosing recipes on development results. Outputs must be new directories
outside the read-only asset tree. Ctrl+C preserves partial metrics and exits
cleanly; this diagnostic does not save resumable model checkpoints. Its
per-domain losses measure early learning, not reasoning or instruction quality.

Before extending a pilot's token budget, check each source's sampled tokens
against the length of its prepared training stream. Equal-sized source samples
with unequal sampling weights can repeatedly expose the model to a small part
of the intended corpus. Expand the heavily reused sources before comparing
longer runs, and record exposure alongside the loss. Hold the vocabulary,
initialization, source draws, schedule, and training-token budget fixed; confirm
an improvement with another seed and fresh held-out documents. Better held-out
likelihood alone does not establish factual accuracy or useful generation.

For an explicit larger pilot, `--production-mix` prepares all enabled sources
and applies the production MinHash/LSH filter within the sample. Pass it to both
preparation and training. `--long-context-pilot` uses 2,048-token sequences and
allows up to 900 training seconds, still capped at 512 updates. Target-size
microbatches remain limited to two sequences. For example:

```bash
python3 scripts/probe_pretrain_recipe.py prepare --assets /path/to/read-only-assets --output evaluations/production_pilot --production-mix
python3 scripts/probe_pretrain_recipe.py run --assets /path/to/read-only-assets --data evaluations/production_pilot --output evaluations/target_pilot --production-mix --long-context-pilot --scale 360m --steps 256 --seconds 900 --capabilities --save-model
```

`--save-model` writes an **inference-only** BASE snapshot; it cannot resume
training. `--capabilities` runs frozen development diagnostics for factual
completions, arithmetic in two formats, grounding, exact-format instructions,
grammar preference, and continuation repetition. Evaluate a BASE checkpoint
independently with:

```bash
python3 scripts/eval_base_readiness.py --checkpoint checkpoints/92m/pretrain_interrupt.safetensors --tokenizer tokenizer/spakie.model --output evaluations/base_readiness.json
```

Reserve `--split test` for a final comparison after choosing on development
results. These are small diagnostic suites, not public benchmark scores or
proof of chat quality. Full-run readiness also requires current-schema data,
a freshly trained whitespace-preserving tokenizer, recovery checks, and a
budget adequate for the intended capabilities. The pilot's effective batch
differs from the full preset, so its winning learning rate remains provisional.

Check whether a capability test distinguishes the checkpoints being compared.
If every short run scores zero, that test cannot rank their recipes. Inspect
actual completions and use answer likelihood or forced-choice scores only as
diagnostics; improving those scores does not establish correct answers or
instruction following. Calibration against an external model can expose an
insensitive test, but differences in architecture, tokenizer, data, and training
schedule prevent treating it as a matched recipe comparison.

For an inference-only check of copying, prepare one repeat-order protocol from
all compared tokenizers, then run it on each checkpoint:

```bash
python3 scripts/eval_repeat_order.py prepare --tokenizers /path/to/first.model /path/to/second.model --output evaluations/repeat_protocol.json
python3 scripts/eval_repeat_order.py run --protocol evaluations/repeat_protocol.json --checkpoint /path/to/base.safetensors --tokenizer /path/to/first.model --output evaluations/repeat_first.json
```

It compares repeated random sequences with shuffled sequences made from the
same single-token word pieces, excluding original adjacent transitions from
the shuffled control. The first token of each sequence is excluded from scoring.
Results report full-vocabulary prediction accuracy and loss by sequence length;
positive `shuffle_minus_repeat_nll` means ordered repetition was easier.
This artificial task measures a narrow copying behavior, not useful retrieval
or general reasoning. Size, training history and vocabulary remain confounded
between unrelated checkpoints. Use matched controls before attributing changes
to a recipe. The default time budget is 120 seconds, checked between cases;
Ctrl+C saves partial results and exits with code 130.

Readiness output preserves the original numeric-prefix scores and additionally
reports `grounding_diagnostic` for each grounding answer. This narrow parser
recognizes a correct first assertion such as “The badge code for Tarin is 431”;
it exposes trailing text for review rather than certifying later claims.
`whole_response_supported` requires no trailing text and a known generation stop
before the token limit. An unterminated number at an unknown or exhausted output
budget has an undetermined (`null`) first-assertion result. These supplemental fields
do not replace frozen experiment gates. `generation_budget_reached` flags answers
that used the entire output allowance; inspect these for truncation before
concluding that the model lacks the tested ability.

Verified synthetic BASE curricula can be prepared independently of any local
experiment folder:

```bash
python3 scripts/build_reasoning_curriculum.py --tokenizer tokenizer/spakie.model --output evaluations/reasoning_data --broad-numbers --decomposed-addition
```

The generator writes plain next-token training documents, token arrays,
development tasks, a separate context test set, and a provenance manifest.
Context entities and numeric values are separated across splits; addition
documents exclude held-out operand pairs, including intermediate equations.
`--retrieval-only` replaces the context comparison exercises with retrieval
examples. These are experimental data options, not production defaults.

Evaluate explained addition on any compatible MLX BASE checkpoint with:

```bash
python3 scripts/eval_reasoning_probe.py --checkpoint checkpoints/92m/pretrain_interrupt.safetensors --tokenizer tokenizer/spakie.model --output evaluations/reasoning_dev.json
```

Every checkpoint receives the same worked demonstration. Final-answer accuracy
and correctness of all intermediate calculations are reported separately.
A supplementary addition audit scans the entire output, including text after
the final answer, for wrong numeric `+`/`=` and `plus`/`equals` claims. The
`_no_detected_addition_error` metric requires the expected answer and steps
with no detected contradictory addition claim. It does not certify other
operations, number words, scientific notation, or the truth of all prose.
The original answer and intermediate-step metrics remain unchanged.
Reserve `--split test` for a selected recipe; its operand pairs and question
wording differ. `--tasks evaluations/reasoning_data/tasks.json` evaluates the
generated context/development tasks instead. Optional `--validation-array`
measures general-text loss on fixed windows; `--validation-split test` uses
separate windows. Windows use the smaller of 2,048 tokens and the checkpoint's
context limit, recorded as `validation_seq_len`; compare language losses only
at the same window length and with the same tokenizer. The array must provide
128 non-overlapping input windows plus a final target token (32 development
windows and 96 test windows).
Small arithmetic gains do not establish general reasoning,
factual knowledge, instruction following, or transfer to another model size.
After using a test set to diagnose a failure, reserve new cases before tuning
again. `--tasks` accepts frozen external cases; `--validation-seed` can select
new language windows. Check their overlap and freeze the seed before comparing.

`scripts/probe_base_learning.py` compares bounded BASE continuation against
matched corpus replay. Its default asset guard targets the frozen historical
92M checkpoint; it is not a general production training entry point.
`--extended-pilot` explicitly permits at most 2,048 updates and 600 training
seconds. Alternatively, `--token-pilot` permits at most 4,096 updates
(4,194,304 tokens at the standard probe batch size), with the same 600-second
training limit. These flags are mutually exclusive; neither bypasses the time
limit, and a time-limited run is not a completed comparison.
Outputs are inference-only snapshots with fresh optimizer state;
they cannot resume optimizer state. `--initialize-from PATH` can use a completed,
unblended BASE probe to start another bounded stage with a fresh optimizer.
It checks root lineage, model and tokenizer contracts, records the starting
checkpoint, and preserves cumulative token/update counts. The replay teacher
remains the original frozen base. Count every stage when comparing budgets.
Compare equal update/token budgets, replay
sampling schedules, and seeds before selecting a data mix. Keep the parent
checkpoint and original tokenizer/data unchanged.

The probe uses a constant learning rate by default. `--lr-schedule cosine`
decays from `--lr` on the first update to 10% of it on the final planned
update. Actual rates are logged; an interrupted or time-limited run may stop
before reaching that floor. This option does not change production schedules.

For continuation experiments, `--curriculum-sampling fixed` gives an exact
number of curriculum rows per batch (`--curriculum-fraction 0.25` means one of
four). `--replay-loss kl --replay-kl-weight 1` replaces replay's hard-label loss
with KL divergence from the frozen parent; only curriculum rows retain normal
next-token cross entropy. Use the identical replay objective in the control.
The experimental KL coefficient accepts finite values from 0 through 20;
larger values strengthen replay anchoring, not the curriculum loss.
The logged token budget includes both curriculum and distillation tokens.
Differentiated and inference forwards may differ numerically even at identical
weights, so pure KL replay is not necessarily a zero-update control. Compare
its measured drift alongside the curriculum arm and the unchanged parent.
This is an opt-in MLX probe feature, not a production pretraining default.

`--sequence-length` selects 256 (default), 512, 1024, or 2048 training tokens,
up to the source checkpoint's context limit. Match it between comparison arms:
each update presents four times this many tokens, including replay. The update
and wall-time caps still apply. The fixed sanity probe retains its 128-token
sequences. General validation windows remain unchanged across these settings.

`--curriculum-window document` moves each sampled curriculum offset backward
to its document start. It verifies every JSONL document against the packed
tokens and requires each document to fit in the training length plus one. The first
document is complete; later documents may still be cut off. Sampling remains
biased by document length and changes the mix of examples actually seen, so
compare logged offsets and exposure at the same token budget. The default
remains `random`; general replay sampling is unchanged.

To investigate factual answer flips with continuous measurements:

```bash
python3 scripts/eval_fact_confidence.py --parent PARENT_BASE --checkpoint CANDIDATE_BASE --tokenizer tokenizer/spakie.model --output evaluations/fact_confidence.json
```

This scores every token of one canonical answer per fact and compares it with
rival tokens fixed by the parent. It supplements greedy checks; a low canonical
probability can reflect a valid paraphrase and is not a semantic correctness
judgment. Repeat `--checkpoint` to compare controls and candidates together.

For an explicit inference-only rollback experiment,
`scripts/blend_base_probe.py --base PARENT --donor PROBE --fraction 0.875 --output NEW_FILE`
retains the requested fraction of a probe update. It verifies the exact parent
hash and model/tokenizer contracts and refuses already blended donors.
The fraction is an experimental choice, not a recommended default.
`scripts/rescore_base_probe.py --records RECORDS_JSONL --output NEW_JSON`
re-scores preserved completions without training or generation and reports
changed decisions and the scoring version.

Run the basic QA evaluator:

```bash
python3 scripts/build_targeted_data.py
python3 scripts/eval_basic_qa.py --backend mlx --preset 300m
python3 scripts/eval_basic_qa.py --backend torch --device auto --precision auto
```

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Notable tests:

- `tests/test_muon.py` - Muon optimizer math, parameter classification, and BF16 tolerance bounds
- `tests/test_mlx_parity.py` - numerical parity between PyTorch and MLX transformer paths
- `tests/test_scaling.py` - preset/config invariants and data preparation behavior
- `tests/test_runtime.py` - device/precision auto-resolution

## Model Presets

The repo currently supports these presets:

| Preset | Layers | `d_model` | Q heads | KV heads | MLP | Pretrain batch | Grad accum | SFT batch | SFT grad accum | Notes |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `92m` | 12 | 768 | 12 | 4 | SwiGLU hidden 2048 | 16 | 4 | 16 | 4 | Default modern small preset; RoPE + QK norm; blocked attention |
| `180m` | 24 | 768 | 12 | 4 | SwiGLU hidden 2304 | 12 | 4 | 8 | 4 | ~184M parameters, RoPE + QK norm; blocked attention |
| `300m` | 24 | 1024 | 16 | 4 | SwiGLU hidden 3072 | 16 | 3 | 4 | 2 | ~315M parameters, RoPE + QK norm; sequential accumulation; blocked attention |
| `360m` | 28 | 1024 | 16 | 4 | SwiGLU hidden 3072 | 8 | 6 | 4 | 2 | ~363M parameters; same token batch as 300m, sequential accumulation; blocked attention |

Shared model defaults:

- `vocab_size = 24576`
- `max_seq_len = 2048`
- `dropout = 0.0`
- `bias = false`
- RoPE (`theta=100000`) with QK norm for every preset
- weight-tied LM head
- SwiGLU MLPs
- scaled dot-product attention, with grouped-query attention where configured
- activation checkpointing disabled by default for all current presets

MLX training on every preset computes causal attention in query blocks of 512
tokens (`attention_query_block`, overridable with `--attention-query-block`;
0 disables it). Each block attends to
keys up to its own end. MLX 0.31 fuses the attention forward pass but not its
backward, which otherwise builds the full 2,048 × 2,048 score matrix; blocking
skips the masked upper half. The result matches dense attention up to bf16
rounding. On the M5 Max it raised training throughput by 15–23% and cut peak
memory by 7–11 GB on `92m`, `180m`, and `360m`. Evaluation, generation, and
packed SFT batches with segment masks always use dense attention.
At `300m`, blocked attention also keeps the B16 × 2,048 microbatch at a 76 GB
peak (7,000–9,000 tok/s); dense attention pushes it to memory pressure on a
128 GB machine.

## Balanced Pretraining Corpus

The default corpus target is 10B training tokens (about 10.53B processed with
the 95/5 split), with `max_seq_len=2048`. The source plan is intentionally
balanced by capability domain:

| Domain | Target share |
|---|---:|
| FineWeb-Edu + filtered-web supplements | 45% |
| Wikipedia + books | 15% |
| Math (`FineMath-4+` + OpenWebMath) | 15% |
| Educational Python code (score 4+) | 15% |
| arXiv + StackExchange + Cosmopedia | 10% |

Download and prepare a fresh generation; existing processed arrays retain the
old mixture until rebuilt:

```bash
python3 scripts/download_pretrain_corpus.py --sources all --resume --english-only
python3 scripts/prepare_data.py
```

Before a full run, compare the presets at the same token budget:

```bash
python3 scripts/train.py --preset 180m --backend mlx --target-tokens 300000000
python3 scripts/train.py --preset 300m --backend mlx --target-tokens 300000000
```

Use `scripts/eval_general_capability.py` after each matched run. Its fixed suite
reports results by category (math, factual recall, instruction following,
formatting, and edge cases), so architecture selection is not based on aggregate
validation loss alone.

## Architecture Notes

The codebase has two parallel backends that share configs, tokenizer, checkpoints, and CLI entry points.

| Concern | Torch | MLX |
|---|---|---|
| Model | `model/transformer.py` | `model/transformer_mlx.py` |
| Dataset | `training/dataset.py` | `training/dataset_mlx.py` |
| Pretrain loop | `training/pretrain.py` | `training/pretrain_mlx.py` |
| SFT loop | `training/finetune.py` | `training/finetune_mlx.py` |
| Optimizer | `training/optimizers.py` | `training/optimizers_mlx.py` |
| Chat / generate | `inference/chat.py`, `inference/generate.py` | `inference/chat_mlx.py`, `inference/generate_mlx.py` |

When changing model behavior or training logic, keep the Torch and MLX paths aligned and re-run the parity tests.

## Mac Troubleshooting

If torch+MPS hits unsupported ops or memory pressure on macOS, these environment variables can help:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Chat Template

```text
<|system|>You are Spakie, a helpful AI language model.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```

The system turn is optional. `scripts/chat.py` omits it by default unless
`--system` is provided.
