"""Merge SFT JSONL files from data/chat_raw/ into a single training file.

Each *.jsonl file in the raw dir is treated as one source — the filename stem
is the source name and must be explicitly enabled in
`SpakieConfig.sft_source_limits`. This fail-closed allowlist prevents ad-hoc or
benchmark-tuned files from silently entering the canonical mixture.

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
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from runtime.langid import is_probably_english
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

# Reviewer annotations belong in a review log, not in assistant targets. The
# preceding answer is not guaranteed to have been rewritten correctly, so the
# safest policy is to exclude the whole example rather than teach internal
# review commentary.
CORRECTION_ANNOTATION_RE = re.compile(r"(?:^|\n)\s*Correction\s*:", re.IGNORECASE)

# Refusal-style examples are excluded from general sources and accepted only
# from the small, explicitly configured refusal sources. Keep ordinary
# uncertainty (for example, "I can't guarantee") because it is useful factual
# calibration.
REFUSAL_RESPONSE_RE = re.compile(
    r"\b(?:i\s+(?:can(?:not|'t)|cannot|won't|will\s+not|am\s+unable\s+to)|"
    r"sorry[, ]+but\s+i\s+(?:can(?:not|'t)|cannot|won't|will\s+not))\s+"
    r"(?:help|assist|provide|write|give|create|do|explain|tell|offer|support)\b",
    re.IGNORECASE,
)

FOREIGN_IDENTITY_RE = re.compile(
    r"\b(?:i\s+am|i'm|as)\s+(?:an?\s+)?(?:chatgpt|claude|gemini|glm|gpt[-\w.]*)\b|"
    r"\b(?:created|developed|trained)\s+by\s+(?:openai|anthropic|google)\b",
    re.IGNORECASE,
)

DIRECT_IDENTITY_QUERY_RE = re.compile(
    r"^\s*(?:who\s+are\s+you|what\s+are\s+you|what\s+ai\s+are\s+you|"
    r"what\s+(?:model|language\s+model)\s+are\s+you|what(?:'s|\s+is)\s+your\s+name|"
    r"what\s+should\s+i\s+call\s+you|tell\s+me\s+about\s+yourself|identify\s+yourself|"
    r"introduce\s+yourself|are\s+you\s+(?:human|a\s+person|an?\s+ai|an?\s+language\s+model|"
    r"chatgpt|claude|gemini|an?\s+software\s+engineer))\s*[?.!]*\s*$",
    re.IGNORECASE,
)

SFT_IDENTITY_NAME = "Spakie-180M"


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
        else:
            variants.add(current + "?")
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
    ("Who are you?", "I am Spakie-180M, a helpful AI language model."),
    ("What are you?", "I am Spakie-180M, a 180-million-parameter AI language model."),
    ("Are you human?", "No. I am Spakie-180M, a 180-million-parameter AI language model."),
    ("How old are you?", "I do not have an age. I am an AI assistant."),
    ("Where do you live?", "I do not live anywhere. I run as software."),
    ("Do you have a husband?", "No. I am an AI assistant and do not have personal relationships."),
    ("Tell me about yourself.", "I am Spakie-180M, an AI language model that gives clear and useful answers."),
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

_IDENTITY_PROMPTS = (
    "Who are you?",
    "What are you?",
    "What AI are you?",
    "What model are you?",
    "What language model are you?",
    "What is your name?",
    "Tell me your name.",
    "Identify yourself.",
    "Tell me about yourself.",
    "Which AI model am I talking to?",
    "Which model is answering me?",
    "What should I call you?",
    "Are you an AI?",
    "Are you a language model?",
    "How many parameters do you have?",
    "What size model are you?",
    "Are you Spakie?",
    "Are you Spakie-180M?",
    "Is this ChatGPT?",
    "Are you ChatGPT?",
    "Are you Claude?",
    "Are you Gemini?",
    "Are you a software engineer?",
    "Are you a person?",
    "Are you human?",
    "Introduce yourself briefly.",
    "Give me a one-sentence introduction.",
    "Say who you are in plain English.",
    "Remind me which model this is.",
    "What's the name of this assistant?",
)

_IDENTITY_ANSWERS = (
    "I am Spakie-180M, a 180-million-parameter AI language model.",
    "My name is Spakie-180M. I am an AI language model with 180 million parameters.",
    "You are talking to Spakie-180M, a small AI language model designed to be helpful.",
    "I am Spakie-180M. I answer questions and help with tasks using a 180-million-parameter language model.",
    "This is Spakie-180M, an AI assistant powered by a 180-million-parameter language model.",
    "I am not a person or a software engineer. I am Spakie-180M, an AI language model.",
)

_IDENTITY_NEGATIVE_PAIRS = (
    ("Are you ChatGPT?", "No. I am Spakie-180M, a 180-million-parameter AI language model."),
    ("Is this ChatGPT?", "No. This is Spakie-180M, an AI language model with 180 million parameters."),
    ("Are you Claude?", "No. I am Spakie-180M, a 180-million-parameter AI language model."),
    ("Are you Gemini?", "No. I am Spakie-180M, a 180-million-parameter AI language model."),
    ("Are you a software engineer?", "No. I am Spakie-180M, an AI language model. I can still help with software questions."),
    ("Are you human?", "No. I am Spakie-180M, an AI language model with 180 million parameters."),
    ("What company do you work for?", "I do not work for a company. I am Spakie-180M, an AI language model."),
)

_FACTUAL_REPAIR_SEEDS = (
    ("Who was Nikola Tesla?", "Nikola Tesla was a Serbian-American inventor and electrical engineer known for his work on alternating-current power systems."),
    ("Who is Nicola Tesla?", "You probably mean Nikola Tesla. He was a Serbian-American inventor and electrical engineer known for developing alternating-current power technology."),
    ("Was Nikola Tesla a woman?", "No. Nikola Tesla was a man, born in 1856 in Smiljan in the Austrian Empire, in present-day Croatia."),
    ("Was Nikola Tesla American?", "Tesla was born in the Austrian Empire to a Serbian family and later became a naturalized American citizen."),
    ("What did Nikola Tesla invent?", "Tesla developed important alternating-current motors and power systems and contributed to radio, wireless transmission, and high-voltage engineering."),
)


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


def contains_correction_annotation(messages: object) -> bool:
    """Return True when reviewer ``Correction:`` text is embedded in a turn."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(msg, dict)
        and isinstance(msg.get("content"), str)
        and CORRECTION_ANNOTATION_RE.search(msg["content"]) is not None
        for msg in messages
    )


def contains_refusal_response(messages: object) -> bool:
    """Return True when an assistant turn is an explicit refusal response."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and isinstance(msg.get("content"), str)
        and REFUSAL_RESPONSE_RE.search(msg["content"]) is not None
        for msg in messages
    )


def contains_foreign_identity_claim(messages: object) -> bool:
    """Reject assistant turns that claim to be another named model."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and isinstance(msg.get("content"), str)
        and FOREIGN_IDENTITY_RE.search(msg["content"]) is not None
        for msg in messages
    )


def contains_conflicting_identity_example(messages: object) -> bool:
    """Reject direct identity Q&A unless the answer names Spakie-180M."""
    if not isinstance(messages, list):
        return False
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or DIRECT_IDENTITY_QUERY_RE.match(content) is None:
            continue
        for reply in messages[index + 1 :]:
            if not isinstance(reply, dict):
                continue
            if reply.get("role") == "assistant" and isinstance(reply.get("content"), str):
                if SFT_IDENTITY_NAME not in reply["content"]:
                    return True
                break
    return False


def is_english_sft_example(messages: list[dict], config: SpakieConfig) -> bool:
    """Language-filter chat while retaining short English factual answers."""
    text = "\n".join(str(msg.get("content", "")) for msg in messages)
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ascii_letters = sum("a" <= char.lower() <= "z" for char in letters)
    ascii_ratio = ascii_letters / len(letters)
    # fastText is unreliable on tiny answers such as "Paris". Short examples
    # are accepted only when their alphabetic content is overwhelmingly ASCII.
    if len(letters) < 100 or text.count(" ") < 20:
        return ascii_ratio >= 0.90
    return ascii_ratio >= 0.70 and is_probably_english(text, config)


def _smoltalk_bucket(messages: list[dict], tokenizer: SpakieTokenizer | None) -> str:
    """Bucket SmolTalk by task shape, turn count, and rendered length."""
    user_text = "\n".join(
        str(msg.get("content", "")) for msg in messages if msg.get("role") == "user"
    ).lower()
    all_text = "\n".join(str(msg.get("content", "")) for msg in messages).lower()
    if len(messages) > 2:
        return "multi_turn"
    if "```" in all_text or re.search(
        r"\b(?:python|javascript|typescript|java|rust|golang|sql|html|css|function|code|program)\b",
        user_text,
    ):
        return "code"
    if re.search(r"\b(?:rewrite|summari[sz]e|translate|edit|proofread|email|letter|essay)\b", user_text):
        return "writing"
    if re.search(
        r"\b(?:exactly|at least|at most|must contain|format as|bullet points?|json|one word|"
        r"one sentence|do not include)\b",
        user_text,
    ):
        return "constrained"
    if tokenizer is not None:
        total, _ = rendered_token_lengths(messages, tokenizer)
    else:
        total = sum(max(1, len(msg.get("content", "")) // 4) + 2 for msg in messages)
    if total <= 180:
        return "general_short"
    if total <= 340:
        return "general_medium"
    return "general_long"


def stratify_smoltalk_examples(
    examples: list[dict],
    limit: int,
    seed: int,
    tokenizer: SpakieTokenizer | None,
) -> list[dict]:
    """Select a deterministic, concise, multi-turn-aware SmolTalk mixture."""
    if limit <= 0 or len(examples) <= limit:
        return examples

    weights = {
        "multi_turn": 0.20,
        "code": 0.12,
        "writing": 0.15,
        "constrained": 0.13,
        "general_short": 0.20,
        "general_medium": 0.15,
        "general_long": 0.05,
    }
    buckets: dict[str, list[dict]] = {name: [] for name in weights}
    for example in examples:
        buckets[_smoltalk_bucket(example["messages"], tokenizer)].append(example)

    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)

    selected: list[dict] = []
    safe_leftovers: list[dict] = []
    for name, weight in weights.items():
        target = int(limit * weight)
        rows = buckets[name]
        selected.extend(rows[:target])
        if name not in {"code", "constrained", "writing"}:
            safe_leftovers.extend(rows[target:])

    rng.shuffle(safe_leftovers)
    if len(selected) < limit:
        selected.extend(safe_leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


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
        cleaned_message = {"role": role, "content": content}
        if "train" in msg:
            # Optional per-turn supervision metadata. Absence retains the
            # historical behavior (all assistant turns are targets), while a
            # literal false keeps an assistant response as context only.
            if role != "assistant" or not isinstance(msg["train"], bool):
                return None
            cleaned_message["train"] = msg["train"]
        cleaned.append(cleaned_message)

    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return None

    if system_prompt is not None:
        cleaned.insert(0, {"role": "system", "content": system_prompt})

    normalized = {"messages": cleaned}
    for key in ("source", "category"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def build_assistant_seed_examples(system_prompt: str | None) -> list[dict]:
    """Return one canonical copy of each assistant-behavior SFT anchor."""
    examples: list[dict] = []
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
) -> list[dict]:
    """Return one canonical copy of each single-turn seed example."""
    examples: list[dict] = []
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


def build_identity_seed_examples(system_prompt: str | None) -> list[dict]:
    """Build varied, non-duplicate anchors for the Spakie-180M identity."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for prompt in _IDENTITY_PROMPTS:
        for variant in expand_user_phrasings(prompt):
            for answer in _IDENTITY_ANSWERS:
                pair = (variant, answer)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    for pair in _IDENTITY_NEGATIVE_PAIRS:
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return build_pair_seed_examples(tuple(pairs), system_prompt)


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
    # Dedup on user/assistant content and explicit supervision intent. The same
    # conversation with a different loss mask is not the same training example.
    return tuple(
        (
            msg["role"],
            msg["content"],
            msg.get("train", True) if msg["role"] == "assistant" else None,
        )
        for msg in example["messages"]
        if msg["role"] != "system"
    )


def deduplicate_examples(examples: list[dict]) -> tuple[list[dict], int]:
    """Drop exact conversation duplicates while preserving first-seen order."""
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for example in examples:
        sig = signature(example)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(example)
    return deduped, len(examples) - len(deduped)


def supervise_final_assistant_only(messages: list[dict]) -> None:
    """Keep earlier assistant turns as context and target only the final one."""
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("final-assistant supervision requires an assistant turn")
    for index in assistant_indices:
        messages[index]["train"] = index == assistant_indices[-1]


def load_source(
    path: str,
    system_prompt: str | None,
    limit: int,
    seed: int,
    tokenizer: SpakieTokenizer | None = None,
    max_seq_len: int = 0,
    max_assistant_tokens: int = 0,
    final_assistant_only: bool = False,
    source_name: str = "",
    config: SpakieConfig | None = None,
) -> list[dict]:
    config = config or SpakieConfig()
    examples: list[dict] = []
    malformed = 0
    filtered = 0
    correction_filtered = 0
    refusal_filtered = 0
    identity_filtered = 0
    conflicting_identity_filtered = 0
    non_english = 0
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
            if not isinstance(raw, dict):
                malformed += 1
                continue
            if contains_disallowed_sft_marker(raw.get("messages")):
                filtered += 1
                continue
            if contains_correction_annotation(raw.get("messages")):
                correction_filtered += 1
                continue
            refusal_allowed = source_name in set(config.sft_refusal_sources)
            if contains_refusal_response(raw.get("messages")) and not refusal_allowed:
                refusal_filtered += 1
                continue
            if contains_foreign_identity_claim(raw.get("messages")):
                identity_filtered += 1
                continue
            if contains_conflicting_identity_example(raw.get("messages")):
                conflicting_identity_filtered += 1
                continue
            example = normalize_example(raw, system_prompt)
            if example is None:
                malformed += 1
                continue
            if source_name:
                original_source = example.get("source")
                if isinstance(original_source, str) and original_source != source_name:
                    example["source_detail"] = original_source
                example["source"] = source_name
                for source_id_key in ("id", "uuid", "row_id", "source_id"):
                    source_id = raw.get(source_id_key)
                    if isinstance(source_id, (str, int)) and str(source_id).strip():
                        example["source_row_id"] = str(source_id).strip()
                        break
            if not is_english_sft_example(example["messages"], config):
                non_english += 1
                continue
            if final_assistant_only:
                supervise_final_assistant_only(example["messages"])
            if tokenizer is not None and not fits_context(
                example["messages"], tokenizer, max_seq_len, max_assistant_tokens
            ):
                too_long += 1
                continue
            if tokenizer is not None and source_name:
                total_tokens, assistant_tokens = rendered_token_lengths(
                    example["messages"], tokenizer
                )
                example["rendered_tokens"] = total_tokens
                example["assistant_tokens"] = assistant_tokens
            if source_name:
                canonical_messages = json.dumps(
                    example["messages"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )
                example["example_id"] = hashlib.sha256(
                    canonical_messages.encode("utf-8")
                ).hexdigest()
            examples.append(example)
    if malformed:
        print(f"    warning: skipped {malformed} malformed lines in {os.path.basename(path)}")
    if filtered:
        print(f"    filtered {filtered} tool/template artifact examples in {os.path.basename(path)}")
    if correction_filtered:
        print(
            f"    excluded {correction_filtered} reviewer-correction examples in "
            f"{os.path.basename(path)}"
        )
    if refusal_filtered:
        print(
            f"    excluded {refusal_filtered} refusal examples in "
            f"{os.path.basename(path)}"
        )
    if identity_filtered:
        print(
            f"    excluded {identity_filtered} foreign-model identity examples in "
            f"{os.path.basename(path)}"
        )
    if conflicting_identity_filtered:
        print(
            f"    excluded {conflicting_identity_filtered} conflicting identity examples in "
            f"{os.path.basename(path)}"
        )
    if non_english:
        print(f"    excluded {non_english} non-English examples in {os.path.basename(path)}")
    if too_long:
        print(
            f"    dropped {too_long} examples over the {max_seq_len}-token window "
            f"in {os.path.basename(path)} (would truncate an answer)"
        )
    # Length-filter before applying the cap so each source contributes `limit`
    # usable examples rather than `limit` rows of which some are then discarded.
    if source_name == "smoltalk":
        before = len(examples)
        examples = stratify_smoltalk_examples(examples, limit, seed, tokenizer)
        if len(examples) < before:
            print(
                f"    stratified SmolTalk selection: {len(examples):,}/{before:,} examples "
                "(turn/length balanced)"
            )
    elif limit > 0 and len(examples) > limit:
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
        default=config.sft_download_max_examples,
        help=(
            "Global cap on total merged examples "
            f"(default: {config.sft_download_max_examples:,}; 0 = no cap)"
        ),
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
        default=512,
        help=(
            "Drop examples whose combined assistant turns exceed this many tokens "
            "(default: 512; 0 = no cap). Biases the mix toward concise answers."
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
            final_assistant_only=(
                source_name == "nemotron_instruction_following_chat_v3"
            ),
            source_name=source_name,
            config=config,
        )
        counts[source_name] = len(examples)
        cap_note = f" (cap {limit:,})" if limit > 0 else ""
        print(f"  {source_name}: {len(examples):,} examples{cap_note}")
        all_examples.extend(examples)

    deduped, dropped = deduplicate_examples(all_examples)
    if dropped:
        print(f"Removed {dropped:,} duplicate examples")

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
