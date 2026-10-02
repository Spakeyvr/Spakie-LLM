"""Teacher-forced choice log-probabilities for BASE probes (MLX)."""

from __future__ import annotations


def continuation_ids(tokenizer, prompt, choice, separator=' '):
    """Prompt ids and the ids of ``separator + choice``; the prompt must tokenize as a stable prefix."""
    prompt_ids = tokenizer.encode(prompt)
    full = tokenizer.encode(prompt + separator + choice)
    if full[:len(prompt_ids)] != prompt_ids or len(full) <= len(prompt_ids):
        raise ValueError(f'Unstable tokenization boundary for choice {choice!r}')
    return prompt_ids, full[len(prompt_ids):]


def choice_logprobs(model, tokenizer, prompt, choices, separator=None):
    """Total log-probability of each choice continuing ``prompt``.

    ``separator`` defaults to a space, or nothing when the prompt already ends
    in whitespace (a code line ending in a newline, for example).
    """
    import mlx.core as mx
    import numpy as np
    if separator is None:
        separator = '' if prompt[-1:].isspace() else ' '
    logprobs = []
    for choice in choices:
        prompt_ids, target = continuation_ids(tokenizer, prompt, choice, separator)
        ids = prompt_ids + target
        logits, _, _ = model(mx.array([ids[:-1]]))
        rows = np.array(logits[0, len(prompt_ids) - 1:].astype(mx.float32))
        rows = rows - rows.max(axis=-1, keepdims=True)
        log_norm = np.log(np.exp(rows).sum(axis=-1))
        logprobs.append(float(sum(rows[i, t] - log_norm[i] for i, t in enumerate(target))))
    return logprobs


def likelihood_record(model, tokenizer, row):
    """Likelihood fields for a probe row, or None when it has no choices.

    The correct answer is ranked against its rivals; the margin is its
    log-probability minus the best rival's.
    """
    from scripts._base_probe_cases import likelihood_choices
    choices = likelihood_choices(row)
    if not choices:
        return None
    logprobs = choice_logprobs(model, tokenizer, row['prompt'], choices)
    best_rival = max(logprobs[1:])
    return {'likelihood_choices': choices, 'likelihood_logprobs': logprobs,
            'likelihood_correct': logprobs[0] > best_rival,
            'likelihood_margin': logprobs[0] - best_rival}
