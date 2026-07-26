import math

import torch
import torch.nn.functional as F


def _normalized_bbox(refer, index, count, margin=0.04):
    bbox = refer.get("bbox") if isinstance(refer, dict) else None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
            raise ValueError("refer bbox must use normalized [0, 1] coordinates")
    else:
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        row, column = divmod(index, columns)
        x0, x1 = column / columns, (column + 1) / columns
        y0, y1 = row / rows, (row + 1) / rows

    x0 = min(max(x0 + margin, 0.0), 1.0)
    y0 = min(max(y0 + margin, 0.0), 1.0)
    x1 = min(max(x1 - margin, 0.0), 1.0)
    y1 = min(max(y1 - margin, 0.0), 1.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"refer bbox collapses after applying margin={margin}")
    return x0, y0, x1, y1


def compose_joint_refer_latent(
    refers,
    num_frames,
    batch_size,
    dtype,
    device,
    history_latent=None,
    history_strength=1.0,
    refer_alpha=1.0,
    margin=0.04,
):
    """Build one spatially coherent latent canvas before materializing refer K/V.

    Each reference occupies a spatial region in every temporal slot. Unoccupied
    regions optionally retain a clean historical latent, allowing the temporary
    forward pass to jointly recache old scene identity and new objects.
    """
    valid_refers = [
        refer for refer in (refers or [])
        if isinstance(refer, dict) and torch.is_tensor(refer.get("latent"))
    ]
    if not valid_refers:
        return None, []

    first = valid_refers[0]["latent"]
    height, width = first.shape[-2:]
    if history_latent is None:
        canvas = torch.zeros(
            (batch_size, num_frames, first.shape[2], height, width),
            device=device,
            dtype=dtype,
        )
    else:
        history = history_latent.to(device=device, dtype=dtype)
        if history.shape[0] == 1 and batch_size > 1:
            history = history.repeat(batch_size, 1, 1, 1, 1)
        indices = torch.arange(num_frames, device=device) % history.shape[1]
        canvas = history.index_select(1, indices).clone() * history_strength

    for index, refer in enumerate(valid_refers):
        latent = refer["latent"].to(device=device, dtype=dtype)
        if latent.shape[0] == 1 and batch_size > 1:
            latent = latent.repeat(batch_size, 1, 1, 1, 1)
        frame_indices = torch.arange(num_frames, device=device) % latent.shape[1]
        latent = latent.index_select(1, frame_indices)

        x0, y0, x1, y1 = _normalized_bbox(refer, index, len(valid_refers), margin)
        left, right = round(x0 * width), round(x1 * width)
        top, bottom = round(y0 * height), round(y1 * height)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, max(left + 1, right)), min(height, max(top + 1, bottom))

        patch = latent.flatten(0, 1)
        patch = F.interpolate(
            patch,
            size=(bottom - top, right - left),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (batch_size, num_frames))
        destination = canvas[:, :, :, top:bottom, left:right]
        destination.lerp_(patch, refer_alpha)

    return canvas, valid_refers
