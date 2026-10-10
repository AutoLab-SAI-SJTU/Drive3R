"""Load complete inference weights while preserving optional model branches."""

from collections.abc import Mapping
from pathlib import Path

import torch


def load_state_dict(path):
    """Read tensor weights and normalize an optional wrapper / DDP prefix."""
    path = Path(path)
    if path.suffix.lower() == '.safetensors':
        from safetensors.torch import load_file

        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, Mapping):
        for name in ('state_dict', 'model', 'module'):
            if isinstance(state.get(name), Mapping):
                state = state[name]
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint must contain a non-empty tensor state dictionary.')

    normalized = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Checkpoint state dictionary must map string names to tensors.')
        key = name[7:] if name.startswith('module.') else name
        if key in normalized:
            raise ValueError(f'Duplicate checkpoint key after removing module prefix: {key}')
        normalized[key] = value
    return normalized


def load_model_weights(model, path):
    """Validate required keys and shapes before copying any model parameters."""
    state = load_state_dict(path)
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    mismatched = sorted(name for name in expected.keys() & state.keys()
                        if expected[name].shape != state[name].shape)
    if missing or mismatched:
        details = []
        if missing:
            details.append(f'{len(missing)} missing required keys (first 8): {missing[:8]}')
        if mismatched:
            details.append(f'{len(mismatched)} incompatible tensor shapes (first 8): {mismatched[:8]}')
        raise RuntimeError(
            'Checkpoint does not match the selected Drive3R inference configuration: '
            + '; '.join(details) + '. Use complete Drive3R weights and matching model options.'
        )
    result = model.load_state_dict(state, strict=False)
    return len(state), result
