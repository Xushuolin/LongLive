def recent_cache_region(local_end, global_tokens, requested_tokens):
    """Return a protected-prefix-safe recent-cache slice and its token length."""
    available = max(0, int(local_end) - int(global_tokens))
    num_tokens = min(max(0, int(requested_tokens)), available)
    return int(local_end) - num_tokens, num_tokens
