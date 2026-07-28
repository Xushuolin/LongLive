import math


def progressive_refer_alpha(step, num_steps, start_alpha, end_alpha, schedule):
    """Return an absolute sink blend weight for one chunk in a recache window."""
    if schedule == "constant":
        return float(end_alpha)
    if schedule not in {"linear", "cosine"}:
        raise ValueError(f"Unsupported refer sink schedule={schedule!r}")
    if num_steps <= 1:
        progress = 1.0
    else:
        progress = min(max(step / (num_steps - 1), 0.0), 1.0)
    if schedule == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(start_alpha + (end_alpha - start_alpha) * progress)
