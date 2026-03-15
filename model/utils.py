"""Model utilities: parameter counting and summary."""

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    total = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total:,} ({total / 1e6:.1f}M)")
    print()
    seen = set()
    for name, module in model.named_children():
        params = 0
        for p in module.parameters():
            if id(p) not in seen:
                params += p.numel()
                seen.add(id(p))
        print(f"  {name}: {params:,}")
