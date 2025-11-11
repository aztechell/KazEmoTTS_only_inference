"""Utility functions required for Grad-TTS inference."""

import torch


def sequence_mask(length, max_length=None):
    """Return a mask with ones up to each sequence length."""
    if max_length is None:
        max_length = length.max()
    positions = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return positions.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    """Pad a length so it is divisible by the UNet downsampling factor."""
    while length % (2**num_downsamplings_in_unet):
        length += 1
    return length


def convert_pad_shape(pad_shape):
    """Convert a nested list pad description into torch.nn.functional.pad format."""
    reversed_shape = pad_shape[::-1]
    return [item for sublist in reversed_shape for item in sublist]


def generate_path(duration, mask):
    """Expand token durations into an alignment path constrained by ``mask``."""
    device = duration.device

    batch, token_steps, mel_steps = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)
    path = torch.zeros(batch, token_steps, mel_steps, dtype=mask.dtype, device=device)

    flat_cum_duration = cum_duration.view(batch * token_steps)
    path = sequence_mask(flat_cum_duration, mel_steps).to(mask.dtype)
    path = path.view(batch, token_steps, mel_steps)
    path = path - torch.nn.functional.pad(
        path, convert_pad_shape([[0, 0], [1, 0], [0, 0]])
    )[:, :-1]
    return path * mask
