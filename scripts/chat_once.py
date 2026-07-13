"""Generate one response from a Spakie checkpoint.

This is the non-interactive equivalent of scripts/chat.py. It intentionally
uses the same auto mode defaults: pretrain_* checkpoints do raw continuation,
while SFT checkpoints use the chat template.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import (
    checkpoint_search_dirs,
    get_preset_config,
    inherit_attention_shape_from_tensors,
    inherit_mlp_shape_from_tensors,
)
from inference.continuation import decode_prefilled_continuation
from runtime import DEVICE_CHOICES, PRECISION_CHOICES
from runtime.checkpoint_io import (
    config_from_checkpoint_payload,
    discard_training_state,
    load_mlx_checkpoint_config,
    load_mlx_checkpoint_meta,
    load_mlx_model_weights_strict,
    load_torch_checkpoint,
    validate_checkpoint_tokenizer,
)
from scripts.chat import (
    apply_mode_defaults,
    infer_checkpoint_mode,
    list_available_checkpoints,
    resolve_checkpoint_path,
)


_TORCH_EXTS = (".pt",)
_MLX_EXTS = (".safetensors",)


def _generate_torch_once(args: argparse.Namespace, config, ckpt_path: str) -> str:
    from inference.chat import build_prompt_ids
    from inference.generate import generate, generate_json
    from model.transformer import SpakieGPT
    from runtime import resolve_runtime_settings
    from tokenizer.train_tokenizer import SpakieTokenizer

    runtime = resolve_runtime_settings(args.device, args.precision)
    ckpt = load_torch_checkpoint(
        ckpt_path,
        map_location=runtime.device,
        trust_checkpoint=getattr(args, "trust_checkpoint", False),
    )
    if "config" in ckpt:
        config = config_from_checkpoint_payload(
            ckpt["config"], source=ckpt_path,
            schema_version=ckpt.get("config_schema_version"),
        )
    elif not getattr(args, "allow_legacy_config", False):
        raise ValueError("checkpoint has no full config; use --allow-legacy-config only for a verified legacy file")
    tokenizer_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    validate_checkpoint_tokenizer(
        ckpt,
        tokenizer_path,
        source=ckpt_path,
        allow_unverified=getattr(args, "allow_unverified_tokenizer", False),
    )
    discard_training_state(ckpt)

    model = SpakieGPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(runtime.device)
    model.eval()

    tokenizer = SpakieTokenizer(tokenizer_path)
    if args.mode == "continue":
        prompt_ids = tokenizer.encode(args.prompt)
        prompt_ids = prompt_ids[-max(1, config.max_seq_len - args.max_new_tokens):]
        response_ids = generate(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            runtime=runtime,
            stop_on_special_tokens=False,
            ban_special_tokens=False,
        )
        return decode_prefilled_continuation(
            tokenizer,
            prompt_ids,
            response_ids,
            show_special_tokens=args.show_special_tokens,
        )

    prompt_ids = build_prompt_ids(
        tokenizer,
        [{"role": "user", "content": args.prompt}],
        args.system,
    )
    if args.json_mode:
        response = generate_json(
            model,
            tokenizer,
            prompt_ids,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            runtime=runtime,
        )
        return response if response is not None else "(failed to generate valid JSON)"

    response_ids = generate(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        runtime=runtime,
    )
    return tokenizer.decode(response_ids)


def _generate_mlx_once(args: argparse.Namespace, config, ckpt_path: str) -> str:
    from inference.chat_mlx import _build_prompt_ids
    from inference.generate_mlx import generate, generate_json
    from model.transformer_mlx import SpakieGPTMLX
    from runtime.mlx_backend import load_safetensors, resolve_mlx_runtime
    from tokenizer.train_tokenizer import SpakieTokenizer

    runtime = resolve_mlx_runtime(args.precision)
    flat = load_safetensors(ckpt_path)
    model_flat = {k[len("model."):]: v for k, v in flat.items() if k.startswith("model.")}
    if not model_flat:
        raise ValueError(f"no 'model.*' tensors found in {ckpt_path}")

    saved_config = load_mlx_checkpoint_config(
        ckpt_path, allow_legacy_config=getattr(args, "allow_legacy_config", False)
    )
    if saved_config is not None:
        config = saved_config
    else:
        config = inherit_attention_shape_from_tensors(config, model_flat)
        config = inherit_mlp_shape_from_tensors(config, model_flat)

    tokenizer_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    validate_checkpoint_tokenizer(
        load_mlx_checkpoint_meta(ckpt_path),
        tokenizer_path,
        source=ckpt_path,
        allow_unverified=getattr(args, "allow_unverified_tokenizer", False),
    )

    model = SpakieGPTMLX(config)
    load_mlx_model_weights_strict(model, flat, path=ckpt_path)
    del flat, model_flat
    if runtime.dtype is not None:
        model.set_dtype(runtime.dtype)
    model.eval()

    tokenizer = SpakieTokenizer(tokenizer_path)
    if args.mode == "continue":
        prompt_ids = tokenizer.encode(args.prompt)
        prompt_ids = prompt_ids[-max(1, config.max_seq_len - args.max_new_tokens):]
        response_ids = generate(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_on_special_tokens=False,
            ban_special_tokens=False,
        )
        return decode_prefilled_continuation(
            tokenizer,
            prompt_ids,
            response_ids,
            show_special_tokens=args.show_special_tokens,
        )

    prompt_ids = _build_prompt_ids(
        tokenizer,
        [{"role": "user", "content": args.prompt}],
        args.system,
    )
    if args.json_mode:
        response = generate_json(
            model,
            tokenizer,
            prompt_ids,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return response if response is not None else "(failed to generate valid JSON)"

    response_ids = generate(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    return tokenizer.decode(response_ids)


def _available_models(backend: str, preset: str) -> list[tuple[str, str]]:
    config = get_preset_config(preset)
    return [
        (preset, path)
        for path in list_available_checkpoints(checkpoint_search_dirs(config), backend)
    ]


def generate_once(args: argparse.Namespace) -> dict[str, Any]:
    available_models = _available_models(args.backend, args.preset)
    selected_preset, ckpt_path = resolve_checkpoint_path(
        available_models=available_models,
        checkpoint_arg=args.checkpoint,
        model_arg=None,
        interactive=False,
        preset_name=args.preset,
    )
    if selected_preset is None:
        raise ValueError("could not determine checkpoint preset")

    if args.mode == "auto":
        args.mode = infer_checkpoint_mode(ckpt_path)
    apply_mode_defaults(args)

    config = get_preset_config(selected_preset)
    with contextlib.redirect_stdout(io.StringIO()):
        if args.backend == "mlx":
            response = _generate_mlx_once(args, config, ckpt_path)
        else:
            response = _generate_torch_once(args, config, ckpt_path)

    return {
        "ok": True,
        "backend": args.backend,
        "preset": selected_preset,
        "mode": args.mode,
        "checkpoint": ckpt_path,
        "response": response,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one Spakie response")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx")
    parser.add_argument("--mode", choices=("auto", "chat", "continue"), default="auto")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--show-special-tokens", action="store_true")
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--system", default="")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unsafe Python pickle loading for a trusted legacy Torch checkpoint")
    parser.add_argument("--allow-legacy-config", action="store_true",
                        help="Allow shape guessing for verified legacy checkpoints without full config metadata")
    parser.add_argument("--allow-unverified-tokenizer", action="store_true",
                        help="Allow a legacy checkpoint without tokenizer identity metadata")
    args = parser.parse_args()

    try:
        payload = generate_once(args)
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "message": "interrupted"}))
        return 130
    except Exception as exc:
        print(json.dumps({"ok": False, "message": str(exc)}))
        return 1

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
