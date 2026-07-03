"""Merge SFT JSONL files from data/chat_raw/ into a single training file.

Each *.jsonl file in the raw dir is treated as one source — the filename stem
is the source name, used to look up an optional per-source cap from
`SpakieConfig.sft_source_limits`. Any custom JSONL you drop into data/chat_raw/
is picked up automatically; sources without a cap entry are taken in full.

System messages are stripped by default, which works better for small models
where every control token has to earn its keep. Pass --system to inject exactly
one system message into every example.

Examples are length-filtered against the model's context window: any conversation
whose rendered token length (role tokens + content + eos for every turn) exceeds
``--max-seq-len`` is dropped rather than silently truncated. Truncation used to
cut assistant answers mid-sentence — the model never saw the closing ``<eos>`` and
learned to ramble without stopping. ``--max-assistant-tokens`` additionally caps
how verbose a single answer may be, biasing the mix toward concise replies.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from tokenizer.train_tokenizer import SpakieTokenizer


# Templated prefixes injected by scripts/download_sft_data.py (squad, sciq, boolq, arc, openbookqa).
# Matched only at the start of a user message so we never eat legitimate content.
_TEMPLATE_PREFIX_RE = re.compile(r"^\s*(?:Question|Q)\s*:\s*", re.IGNORECASE)
_TEMPLATE_BLOCK_RE = re.compile(
    r"(?:\n+\s*|\s+)(?:Context|Reference|Passage|Choices)\s*:\s*",
    re.IGNORECASE,
)
_TEMPLATE_TRAILER_RE = re.compile(
    r"\n*\s*(?:Answer\s+(?:yes\s+or\s+no\s+)?(?:clearly|using\s+the\s+context|the\s+question)\.?|"
    r"Select\s+the\s+correct\s+answer\.?)\s*$",
    re.IGNORECASE,
)

DISALLOWED_SFT_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
    "Action:",
    "Observation:",
    "Final Answer:",
    "You are an expert in composing functions",
)


# Natural-language phrasing swaps used to diversify the behavior seeds. Repeating
# a single exact prompt string (e.g. "What is the capital of France?") dozens of
# times teaches the model to memorize that literal rather than generalize, so it
# breaks on "What's the capital of France?". Expanding each seed into contraction
# and punctuation variants makes the anchored behavior phrasing-robust.
_PHRASING_SWAPS = (
    ("What is", "What's"),
    ("What are", "What're"),
    ("Who is", "Who's"),
    ("How is", "How's"),
    ("That is", "That's"),
    ("Where is", "Where's"),
    ("It is", "It's"),
)


def expand_user_phrasings(user_text: str) -> list[str]:
    """Return phrasing variants of a seed prompt (contractions, trailing '?')."""
    variants = {user_text}
    for full, short in _PHRASING_SWAPS:
        for current in list(variants):
            if full in current:
                variants.add(current.replace(full, short))
            if short in current:
                variants.add(current.replace(short, full))
    # Both with and without a trailing question mark, so the model does not tie
    # the answer to the presence of punctuation.
    for current in list(variants):
        if current.endswith("?"):
            variants.add(current[:-1])
    return sorted(variants)


def expand_seed_pairs(pairs: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Expand (user, assistant) seeds across user phrasing variants, deduped."""
    expanded: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for user_text, assistant_text in pairs:
        for variant in expand_user_phrasings(user_text):
            key = (variant, assistant_text)
            if key not in seen:
                seen.add(key)
                expanded.append(key)
    return tuple(expanded)


_ASSISTANT_BEHAVIOR_SEED_PAIRS = (
    ("Hi", "Hello! How can I help you today?"),
    ("Hello", "Hello! How can I help?"),
    ("Hey", "Hi! What would you like help with?"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("Are you there?", "Yes, I am here and ready to help."),
    ("Can you help me?", "Yes. Tell me what you need help with, and I will do my best."),
    ("Who are you?", "I am Spakie, a helpful AI assistant."),
    ("What are you?", "I am an AI assistant designed to answer questions and help with tasks."),
    ("Are you human?", "No. I am an AI assistant, not a human."),
    ("How old are you?", "I do not have an age. I am an AI assistant."),
    ("Where do you live?", "I do not live anywhere. I run as software."),
    ("Do you have a husband?", "No. I am an AI assistant and do not have personal relationships."),
    ("Tell me about yourself.", "I am Spakie, an AI assistant that gives clear and useful answers."),
    ("What do you like to cook?", "I do not cook or have personal hobbies, but I can help with recipes."),
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is Python?", "Python is a popular programming language used for software, data, automation, and AI."),
    ("Explain photosynthesis simply.", "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make food and oxygen."),
    ("What is gravity?", "Gravity is the force that pulls objects with mass toward each other."),
    ("If you do not know an answer, what should you do?", "I should say that I do not know instead of making up an answer."),
    ("Answer briefly: what is an operating system?", "An operating system manages a computer's hardware and helps programs run."),
    ("Say hello in one short sentence.", "Hello! How can I help you today?"),
    ("I only said hi. What should you say back?", "I should greet you and ask how I can help."),
    ("Please answer as an assistant, not as a fictional character.", "Understood. I will answer directly as an AI assistant."),
)

_ANTI_ECHO_SEED_PAIRS = (
    ("What's Python", "Python is a popular programming language used for software, data, automation, and AI."),
    ("Tell me about Python", "Python is a programming language used for scripting, web development, data analysis, automation, and AI."),
    ("What's gravity", "Gravity is the force that pulls objects with mass toward each other."),
    ("What are atoms?", "Atoms are tiny building blocks of matter."),
    ("Explain sleep", "Sleep helps the body rest and repair itself. It also supports memory, focus, mood, and energy."),
    ("How do I study?", "Start by reviewing the main ideas, then practice with questions. Short, focused study sessions usually work better than cramming."),
    ("Hi", "Hello! How can I help?"),
    ("Hello", "Hello! How can I help?"),
)

# Public seed sets: each base pair fanned out across phrasing variants so the
# anchored behavior generalizes across contractions and punctuation.
ASSISTANT_BEHAVIOR_SEEDS = expand_seed_pairs(_ASSISTANT_BEHAVIOR_SEED_PAIRS)
ANTI_ECHO_SEEDS = expand_seed_pairs(_ANTI_ECHO_SEED_PAIRS)


def strip_question_template(content: str) -> str:
    """Strip 'Question: ... Context: ... Answer clearly.' scaffolding from a user turn.

    Reorders to '<context>\n\n<question>' so the model still learns context-aware Q&A
    without learning that 'Question:' is a required trigger word.
    """
    if not _TEMPLATE_PREFIX_RE.match(content):
        return content

    body = _TEMPLATE_PREFIX_RE.sub("", content, count=1)
    body = _TEMPLATE_TRAILER_RE.sub("", body)

    match = _TEMPLATE_BLOCK_RE.search(body)
    if match is None:
        return body.strip()

    question = body[: match.start()].strip()
    context = body[match.end():].strip()
    if not question or not context:
        return body.strip()
    return f"{context}\n\n{question}"


def contains_disallowed_sft_marker(messages: object) -> bool:
    """Return True when any message content contains tool/reasoning scaffolding."""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and any(marker in content for marker in DISALLOWED_SFT_MARKERS):
            return True
    return False


def normalize_example(raw: dict, system_prompt: str | None) -> dict | None:
    """Return a clean {messages: [...]} dict or None if malformed.

    system_prompt=None means strip any system message; a string (empty allowed)
    means prepend exactly one system message with that content.
    """
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return None

    cleaned: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        if role == "system":
            continue
        content = content.strip()
        if role == "user":
            content = strip_question_template(content)
        if not content:
            return None
        cleaned.append({"role": role, "content": content})

    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return None

    if system_prompt is not None:
        cleaned.insert(0, {"role": "system", "content": system_prompt})

    return {"messages": cleaned}


def build_assistant_seed_examples(system_prompt: str | None, repeats: int) -> list[dict]:
    """Return repeated assistant-behavior examples for SFT anchoring."""
    if repeats <= 0:
        return []

    examples: list[dict] = []
    for _ in range(repeats):
        for user_text, assistant_text in ASSISTANT_BEHAVIOR_SEEDS:
            messages: list[dict] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            examples.append({"messages": messages})
    return examples


def build_pair_seed_examples(
    pairs: tuple[tuple[str, str], ...],
    system_prompt: str | None,
    repeats: int,
) -> list[dict]:
    """Return repeated single-turn user/assistant seed examples."""
    if repeats <= 0:
        return []

    examples: list[dict] = []
    for _ in range(repeats):
        for user_text, assistant_text in pairs:
            messages: list[dict] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            examples.append({"messages": messages})
    return examples


def rendered_token_lengths(messages: list[dict], tokenizer: SpakieTokenizer) -> tuple[int, int]:
    """Return (total_tokens, assistant_tokens) for a rendered conversation.

    Mirrors how ``ChatSFTDataset``/``ChatSFTDatasetMLX`` tokenize each turn:
    one role token + the content tokens + one ``<eos>``. Used to decide whether
    a conversation fits the context window without truncating any answer.
    """
    total = 0
    assistant = 0
    for msg in messages:
        n = len(tokenizer.encode(msg.get("content", ""))) + 2  # role token + eos
        total += n
        if msg.get("role") == "assistant":
            assistant += n
    return total, assistant


def fits_context(
    messages: list[dict],
    tokenizer: SpakieTokenizer,
    max_seq_len: int,
    max_assistant_tokens: int,
) -> bool:
    """True if the conversation fits the window and respects the answer-length cap.

    The dataset builds ``input_ids[: max_seq_len + 1]`` then shifts to length
    ``max_seq_len``, so a conversation of exactly ``max_seq_len + 1`` tokens is
    still fully supervised (its final ``<eos>`` survives the shift).
    """
    if max_seq_len <= 0:
        return True
    total, assistant = rendered_token_lengths(messages, tokenizer)
    if total > max_seq_len + 1:
        return False
    if max_assistant_tokens > 0 and assistant > max_assistant_tokens:
        return False
    return True


def signature(example: dict) -> tuple:
    # Dedup on user/assistant content only — the system prompt is uniform.
    return tuple(
        (msg["role"], msg["content"])
        for msg in example["messages"]
        if msg["role"] != "system"
    )


def load_source(
    path: str,
    system_prompt: str | None,
    limit: int,
    seed: int,
    tokenizer: SpakieTokenizer | None = None,
    max_seq_len: int = 0,
    max_assistant_tokens: int = 0,
) -> list[dict]:
    examples: list[dict] = []
    malformed = 0
    filtered = 0
    too_long = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if contains_disallowed_sft_marker(raw.get("messages")):
                filtered += 1
                continue
            example = normalize_example(raw, system_prompt)
            if example is None:
                malformed += 1
                continue
            if tokenizer is not None and not fits_context(
                example["messages"], tokenizer, max_seq_len, max_assistant_tokens
            ):
                too_long += 1
                continue
            examples.append(example)
    if malformed:
        print(f"    warning: skipped {malformed} malformed lines in {os.path.basename(path)}")
    if filtered:
        print(f"    filtered {filtered} tool/template artifact examples in {os.path.basename(path)}")
    if too_long:
        print(
            f"    dropped {too_long} examples over the {max_seq_len}-token window "
            f"in {os.path.basename(path)} (would truncate an answer)"
        )
    # Length-filter before applying the cap so each source contributes `limit`
    # usable examples rather than `limit` rows of which some are then discarded.
    if limit > 0 and len(examples) > limit:
        random.Random(seed).shuffle(examples)
        examples = examples[:limit]
    return examples


def main() -> None:
    config = SpakieConfig()
    parser = argparse.ArgumentParser(description="Merge data/chat_raw/*.jsonl into data/chat/train.jsonl")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=config.chat_raw_dir,
        help=f"Directory of per-source JSONL files (default: {config.chat_raw_dir})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(config.chat_data_dir, "train.jsonl"),
        help="Output path for the merged JSONL",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt to inject into every example",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Global cap on total merged examples (0 = no cap)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=config.max_seq_len,
        help=(
            "Drop conversations whose rendered token length exceeds this window "
            f"(default: model max_seq_len = {config.max_seq_len}). 0 disables length filtering."
        ),
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=0,
        help=(
            "Drop examples whose combined assistant turns exceed this many tokens "
            "(0 = no cap). Biases the mix toward concise answers."
        ),
    )
    parser.add_argument(
        "--assistant-seed-repeats",
        type=int,
        default=20,
        help=(
            "Repeat a small built-in assistant-behavior seed set this many times "
            "(0 disables). Helps greetings, identity questions, and short factual prompts."
        ),
    )
    parser.add_argument(
        "--anti-echo-seed-repeats",
        type=int,
        default=80,
        help=(
            "Repeat direct-answer anti-echo seed examples this many times "
            "(0 disables). Helps small models answer instead of continuing/repeating prompts."
        ),
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Optional comma-separated source names (filename stems) to include; default = all *.jsonl",
    )
    args = parser.parse_args()

    if args.system is None:
        system_prompt: str | None = None
        system_label = "(no system message)"
    else:
        system_prompt = args.system
        system_label = "(empty system message)" if system_prompt == "" else repr(system_prompt)

    if not os.path.isdir(args.input_dir):
        print(f"Input dir not found: {args.input_dir}")
        print("Run `python3 scripts/download_sft_data.py` first, or drop your own *.jsonl files in there.")
        sys.exit(2)

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    if not paths:
        print(f"No *.jsonl files found in {args.input_dir}/.")
        sys.exit(2)

    requested = {s.strip() for s in args.sources.split(",") if s.strip()}
    if requested:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in requested]
        missing = requested - {os.path.splitext(os.path.basename(p))[0] for p in paths}
        if missing:
            print(f"Missing sources in {args.input_dir}/: {', '.join(sorted(missing))}")
            sys.exit(2)

    print(f"System prompt: {system_label}")

    tokenizer: SpakieTokenizer | None = None
    if args.max_seq_len > 0:
        tokenizer_model = config.tokenizer_prefix + ".model"
        if not os.path.exists(tokenizer_model):
            print(
                f"Tokenizer not found at {tokenizer_model}; cannot length-filter. "
                "Train the tokenizer first or pass --max-seq-len 0."
            )
            sys.exit(2)
        tokenizer = SpakieTokenizer(tokenizer_model)
        cap_note = (
            f", assistant<= {args.max_assistant_tokens}" if args.max_assistant_tokens > 0 else ""
        )
        print(f"Length filter: total<= {args.max_seq_len} tokens{cap_note}")
    else:
        print("Length filter: disabled")

    all_examples: list[dict] = []
    counts: Counter[str] = Counter()
    for path in paths:
        source_name = os.path.splitext(os.path.basename(path))[0]
        if not config.sft_source_enabled(source_name):
            print(f"  {source_name}: disabled")
            continue
        limit = config.sft_source_limit(source_name)
        examples = load_source(
            path,
            system_prompt,
            limit,
            args.seed,
            tokenizer,
            args.max_seq_len,
            args.max_assistant_tokens,
        )
        counts[source_name] = len(examples)
        cap_note = f" (cap {limit:,})" if limit > 0 else ""
        print(f"  {source_name}: {len(examples):,} examples{cap_note}")
        all_examples.extend(examples)

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for example in all_examples:
        sig = signature(example)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(example)
    dropped = len(all_examples) - len(deduped)
    if dropped:
        print(f"Removed {dropped:,} duplicate examples")

    assistant_seed = build_assistant_seed_examples(system_prompt, args.assistant_seed_repeats)
    if assistant_seed:
        deduped.extend(assistant_seed)
        counts["assistant_seed"] = len(assistant_seed)
        print(f"Added {len(assistant_seed):,} assistant-behavior seed examples")

    anti_echo_seed = build_pair_seed_examples(
        ANTI_ECHO_SEEDS,
        system_prompt,
        args.anti_echo_seed_repeats,
    )
    if anti_echo_seed:
        deduped.extend(anti_echo_seed)
        counts["anti_echo_seed"] = len(anti_echo_seed)
        print(f"Added {len(anti_echo_seed):,} anti-echo seed examples")

    random.Random(args.seed).shuffle(deduped)
    if args.max > 0 and len(deduped) > args.max:
        deduped = deduped[: args.max]
        print(f"Capped output at {args.max:,} examples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_path = args.output + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for example in deduped:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(tmp_path, args.output)

    print(f"\nSaved {len(deduped):,} examples to {args.output}")
    for source_name in sorted(counts):
        print(f"  {source_name}: {counts[source_name]:,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while preparing SFT data.")
        sys.exit(130)
