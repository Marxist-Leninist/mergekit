# SPDX-License-Identifier: LGPL-3.0-only
"""Shape-preserving layer remapping for custom in-memory state dictionaries.

This checks storage compatibility, not functional or architectural compatibility.
Callers must evaluate remapped models. Inputs are never modified or aliased.
"""
import re
from collections.abc import Mapping, Sequence
import torch


def remap_layers(state: Mapping[str, torch.Tensor], order: Sequence[int], *, prefix: str = 'blocks') -> dict[str, torch.Tensor]:
    """Copy each donor layer in ``order`` into consecutive original layer slots.

    The depth remains unchanged; donor repetition is allowed. Non-layer weights
    are copied unchanged. Every destination must have matching tensor names,
    shape and dtype. This deliberately does not resize model configurations.
    """
    if not prefix or not order or any(type(i) is not int or i < 0 for i in order):
        raise ValueError('Nonempty prefix/order and nonnegative integer donors required')
    pattern = re.compile(r'^' + re.escape(prefix) + r'\.(\d+)\.(.+)$')
    layers = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Expected tensor: {key}')
        m = pattern.match(key)
        if m:
            layers.setdefault(int(m[1]), {})[m[2]] = value
    if set(layers) != set(range(len(order))):
        raise ValueError('Order must preserve contiguous existing layer count')
    for dest, donor in enumerate(order):
        if donor not in layers or layers[dest].keys() != layers[donor].keys():
            raise ValueError('Missing donor or mismatched layer tensor names')
        for suffix, target in layers[dest].items():
            source = layers[donor][suffix]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f'Incompatible tensor: {dest}.{suffix}')
    output = {}
    for key, value in state.items():
        m = pattern.match(key)
        source = layers[order[int(m[1])]][m[2]] if m else value
        output[key] = source.detach().clone()
    return output
