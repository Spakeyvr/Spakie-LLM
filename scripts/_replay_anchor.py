"""Opt-in prediction anchoring for the bounded MLX BASE continuation probe."""
from __future__ import annotations

import math


def replay_kl(student_logits, teacher_logits, replay_rows):
    """KL(parent || student), masked to replay, averaged over all batch tokens.

The coefficient therefore has the same meaning per replay token in a mixed
batch and a pure replay control. Curriculum rows receive exactly zero anchor
gradient; teacher predictions are always detached. All arithmetic is FP32.
"""
    import mlx.core as mx
    import mlx.nn as nn
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError('Teacher/student logits must share shape (batch, sequence, vocab)')
    if replay_rows.shape != (student_logits.shape[0],):
        raise ValueError('Replay mask must contain exactly one entry per batch row')
    teacher_logits = mx.stop_gradient(teacher_logits.astype(mx.float32))
    teacher_probs = mx.stop_gradient(mx.softmax(teacher_logits, axis=-1))
    # The direct softmax target gives an exactly zero student gradient when
    # logits match. exp(log_softmax) leaves tiny residuals that a normalized
    # optimizer can amplify even at the supposedly stationary parent.
    # This is a logit-level guarantee: differentiated model forwards can differ
    # numerically from inference forwards even with identical stored weights.
    cross_entropy = nn.losses.cross_entropy(student_logits.astype(mx.float32), teacher_probs)
    entropy = mx.stop_gradient(nn.losses.cross_entropy(teacher_logits, teacher_probs))
    token_kl = cross_entropy-entropy
    return mx.where(replay_rows[:, None], token_kl, 0.).mean()


def build_anchored_step(model, teacher, weight, *, replace_replay_ce=False):
    import mlx.core as mx
    import mlx.nn as nn
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Replay anchor weight must be finite and nonnegative')
    if teacher.training:
        raise ValueError('The frozen replay teacher must be in evaluation mode')

    def objective(student, x, y, teacher_logits, replay_rows):
        logits, _, _ = student(x)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), y)
        if replace_replay_ce:
            ce = mx.where(replay_rows[:,None], 0., ce)
        ce = ce.mean()
        kl = replay_kl(logits, teacher_logits, replay_rows)
        return ce + weight*kl, ce, kl

    value_and_grad = nn.value_and_grad(model, objective)

    def step(x, y, replay_rows):
        # Materialize the teacher forward before constructing student backward
        # state. No teacher activations or gradients are retained for backward.
        teacher_logits = mx.stop_gradient(teacher(x)[0])
        mx.eval(teacher_logits)
        return value_and_grad(model, x, y, teacher_logits, replay_rows)

    return step
