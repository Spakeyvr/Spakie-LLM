"""Export a conservative MLX BASE probe update as an inference-only checkpoint.

The donor must record the SHA256 of its exact parent. Model and tokenizer
contracts must match; optimizer/sampler state is never exported or resumable.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_lineage(base_meta, donor_meta, parent_sha, *, base_path=None):
    from scripts._base_probe_cases import is_base_checkpoint
    if not is_base_checkpoint(base_meta, base_path) or donor_meta.get('stage') != 'pretrain':
        raise ValueError('Both checkpoints must be BASE pretrain checkpoints')
    experiment = donor_meta.get('experiment', {})
    if experiment.get('parent_sha256') != parent_sha or not experiment.get('inference_only'):
        raise ValueError('Donor must be an inference-only probe derived from this exact parent')
    for key in ('config', 'tokenizer'):
        if not base_meta.get(key) or donor_meta.get(key) != base_meta[key]:
            raise ValueError(f'Checkpoint {key} contracts differ or are missing')
    if experiment.get('blend'):
        raise ValueError('Use the raw probe donor, not an already blended checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fraction', type=float, required=True,
                        help='Fraction of the learned update to retain, strictly between 0 and 1')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in (args.base.resolve(), args.donor.resolve()):
        parser.error('Output must be a new file, separate from both input checkpoints')
    if not math.isfinite(args.fraction) or not 0 < args.fraction < 1:
        parser.error('Fraction must be finite and strictly between zero and one')
    from scripts.probe_base_learning import digest
    from runtime.checkpoint_io import load_mlx_checkpoint_meta
    from runtime.mlx_backend import load_safetensors, save_safetensors_checkpoint
    import mlx.core as mx
    parent_sha = digest(args.base)
    base_meta = load_mlx_checkpoint_meta(str(args.base))
    donor_meta = load_mlx_checkpoint_meta(str(args.donor))
    validate_lineage(base_meta, donor_meta, parent_sha, base_path=args.base)
    weights = {k:v for k,v in load_safetensors(str(args.base)).items() if k.startswith('model.')}
    donor = load_safetensors(str(args.donor))
    if not weights or weights.keys() != donor.keys():
        raise ValueError('Donor must contain exactly the same model tensors as its parent')
    blended = {}
    for name, base in weights.items():
        other = donor[name]
        if base.shape != other.shape:
            raise ValueError(f'Tensor shape mismatch: {name}')
        blended[name] = ((1-args.fraction)*base.astype(mx.float32)
                         + args.fraction*other.astype(mx.float32)).astype(base.dtype)
    mx.eval(blended)
    donor_meta['experiment']['blend'] = {
        'donor_sha256':digest(args.donor), 'base_sha256':parent_sha,
        'fraction':args.fraction, 'arithmetic':'FP32 interpolation rounded to original storage dtype',
        'script_sha256':digest(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_checkpoint(str(args.output), blended, donor_meta)
    print(f'Saved inference-only blend: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
