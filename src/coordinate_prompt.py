"""Coordinate-prompt construction for Thayer-Select.

Coordinates use image pixel convention ``(x, y)`` with the origin at the
upper-left pixel center.  Empty prompts are explicit zero maps, not a magic
off-image coordinate.  Coordinates may intentionally be wrong at evaluation
time; the prompt builder has no access to target pixels or oracle metrics.
"""

from __future__ import annotations

import torch


def gaussian_coordinate_prompt(
    coordinates_xy: torch.Tensor,
    height: int,
    width: int,
    *,
    sigma_pixels: float = 3.0,
    valid: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return peak-normalized Gaussian prompts with shape ``(B, 1, H, W)``.

    Parameters
    ----------
    coordinates_xy:
        Tensor of shape ``(B, 2)`` containing continuous ``(x, y)`` pixels.
    height, width:
        Spatial output dimensions.
    sigma_pixels:
        Positive Gaussian standard deviation in pixels.
    valid:
        Optional boolean vector of shape ``(B,)``. False entries produce exact
        zero maps and therefore represent empty prompts. Non-finite coordinates
        are also treated as empty.
    """
    if coordinates_xy.ndim != 2 or coordinates_xy.shape[1] != 2:
        raise ValueError("coordinates_xy must have shape (batch, 2)")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be positive")

    if dtype is None:
        dtype = coordinates_xy.dtype if coordinates_xy.is_floating_point() else torch.float32
    if device is None:
        device = coordinates_xy.device
    coords = coordinates_xy.to(device=device, dtype=dtype)
    finite = torch.isfinite(coords).all(dim=1)
    in_frame = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] <= width - 1)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] <= height - 1)
    )
    coordinate_valid = finite & in_frame
    if valid is None:
        valid_mask = coordinate_valid
    else:
        if valid.ndim != 1 or valid.shape[0] != coords.shape[0]:
            raise ValueError("valid must have shape (batch,)")
        valid_mask = valid.to(device=device, dtype=torch.bool) & coordinate_valid

    # Replace non-finite values before arithmetic; the validity mask then makes
    # the corresponding output exactly zero and keeps gradients finite.
    safe_coords = torch.where(torch.isfinite(coords), coords, torch.zeros_like(coords))
    x = torch.arange(width, dtype=dtype, device=device).view(1, 1, 1, width)
    y = torch.arange(height, dtype=dtype, device=device).view(1, 1, height, 1)
    cx = safe_coords[:, 0].view(-1, 1, 1, 1)
    cy = safe_coords[:, 1].view(-1, 1, 1, 1)
    squared_distance = (x - cx).square() + (y - cy).square()
    prompt = torch.exp(-0.5 * squared_distance / float(sigma_pixels**2))

    # Continuous subpixel centers need not land exactly on a pixel. Normalize
    # each non-empty map so its discrete maximum is exactly one.
    maxima = prompt.amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    prompt = prompt / maxima
    return prompt * valid_mask.to(dtype=dtype).view(-1, 1, 1, 1)


def concatenate_image_and_prompt(
    image_grz: torch.Tensor, prompt: torch.Tensor
) -> torch.Tensor:
    """Validate and concatenate three scientific channels with one prompt."""
    if image_grz.ndim != 4 or image_grz.shape[1] != 3:
        raise ValueError("image_grz must have shape (batch, 3, height, width)")
    if prompt.ndim != 4 or prompt.shape[1] != 1:
        raise ValueError("prompt must have shape (batch, 1, height, width)")
    if prompt.shape[0] != image_grz.shape[0] or prompt.shape[2:] != image_grz.shape[2:]:
        raise ValueError("image and prompt batch/spatial shapes must match")
    return torch.cat((image_grz, prompt.to(image_grz)), dim=1)
