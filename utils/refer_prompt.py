"""Utilities for binding reference-image metadata into text prompts."""


def _format_refer_label(refer, index):
    role = str(refer.get("role") or refer.get("name") or refer.get("subject") or f"object {index}").strip()
    if not role:
        role = f"object {index}"
    return role


def _format_layout_clause(refer):
    layout = refer.get("layout") or refer.get("position")
    if layout:
        return f" {layout}"
    bbox = refer.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return f" in bbox {bbox}"
    return ""


def build_refer_binding_text(chunk_refers, *, joint_scene=False, use_layout=True):
    """Build a compact text binding clause from a chunk's refer metadata."""
    valid_refers = [r for r in (chunk_refers or []) if isinstance(r, dict)]
    if not valid_refers:
        return ""

    clauses = []
    for idx, refer in enumerate(valid_refers, start=1):
        role = _format_refer_label(refer, idx)
        layout_clause = _format_layout_clause(refer) if use_layout else ""
        description = refer.get("description") or refer.get("desc") or refer.get("prompt")
        if description:
            role = f"{role}, {description}"
        clauses.append(
            f"Reference image {idx} is the {role}{layout_clause}; preserve its appearance and identity."
        )

    if joint_scene and len(valid_refers) > 1:
        return (
            "Use the references as one coherent multi-object visual anchor. "
            + " ".join(clauses)
            + " Keep the references distinct and preserve their spatial relationship."
        )
    return " ".join(clauses)


def append_refer_binding_to_prompts(prompts, refers_by_chunk, *, joint_scene=False, use_layout=True):
    """Append refer binding text to each chunk prompt that has refer metadata."""
    rewritten = []
    for idx, prompt in enumerate(prompts):
        chunk_refers = refers_by_chunk[idx] if refers_by_chunk is not None and idx < len(refers_by_chunk) else []
        binding = build_refer_binding_text(chunk_refers, joint_scene=joint_scene, use_layout=use_layout)
        rewritten.append(f"{prompt} {binding}".strip() if binding else prompt)
    return rewritten


def build_refer_kv_prompts(prompts, refers_by_chunk, *, use_layout=True):
    """Create prompts specialized for materializing refer K/V caches."""
    rewritten = []
    for idx, prompt in enumerate(prompts):
        chunk_refers = refers_by_chunk[idx] if refers_by_chunk is not None and idx < len(refers_by_chunk) else []
        binding = build_refer_binding_text(chunk_refers, joint_scene=True, use_layout=use_layout)
        if binding:
            rewritten.append(f"{prompt} {binding}".strip())
        else:
            rewritten.append(prompt)
    return rewritten
