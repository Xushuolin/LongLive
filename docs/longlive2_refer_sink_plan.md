# LongLive2 refer-sink integration notes

This repository cannot currently fetch GitHub from the execution sandbox, so the
LongLive2 source is expected to be vendored locally with
`scripts/fetch_longlive2.sh` from a machine that has GitHub access. Keep
LongLive2 isolated under `third_party/longlive2/LongLive` so ShotStream changes
can be reviewed separately from upstream code.

## Why vendor/fork LongLive2 here?

LongLive2 already has the two pieces we want for the next experiment:

1. I2V conditioning, so a reference image can be encoded by an upstream-supported
   conditioning path instead of being forced through a T2V-only interface.
2. A multi-shot sink policy with a long-lived global sink and a shot-level sink
   that is rebound at scene cuts.

The refer-frame experiment should therefore be implemented as a small sink-swap
extension on top of LongLive2, not as a rewrite of ShotStream's current causal
pipeline.

## Preferred source layout

```text
third_party/longlive2/
  README.md                 # short pointer to this plan
  LongLive/                 # vendored upstream NVlabs/LongLive main branch
```

Use the helper script:

```bash
bash scripts/fetch_longlive2.sh --vendor
```

The script clones the upstream main branch into a temporary directory, removes
the nested `.git`, and copies it into `third_party/longlive2/LongLive`. Removing
the nested Git metadata keeps this repository self-contained and avoids
accidental submodule/gitlink commits.

If you prefer to keep LongLive2 as a true independent repository for local
experiments, run:

```bash
bash scripts/fetch_longlive2.sh --worktree
```

That mode preserves LongLive2's `.git` directory and should not be staged into
the parent repository unless a submodule/gitlink is intentional.

## Proposed refer-sink patch design for LongLive2

LongLive2 should keep its existing multi-shot sink semantics:

- **global sink**: preserves long-range identity and should not be overwritten
  completely at shot start;
- **shot sink**: is rebound at scene cuts and is a better target for refer
  injection after the shot has started;
- **local window**: continues to slide normally.

The refer injection should be delayed and partial, matching the lesson from the
ShotStream experiments: do not put the reference image into every sink slot
before any generated shot anchor exists.

### Suggested config knobs

Add these under LongLive2's inference config:

```yaml
refer_sink_swap: false
refer_sink_after_chunks: 1        # wait until this many chunks of the shot exist
refer_sink_injection_chunks: 1    # keep the subject KV swapped for this many chunks
refer_sink_start_slot: 0          # SPAWN-aligned default; ablate against slot 1
refer_sink_num_slots: 2           # with sink_size=8, swap slots 0 and 1
refer_sink_mode: cycle            # cycle | repeat_first
refer_sink_rope_start_frame: 0    # compact RoPE id for swapped refer slots
refer_sink_target: shot           # shot | global
refer_sink_restore: true          # restore original sink KV after the injection window
```

For the first experiment, use:

```yaml
multi_shot_sink: true
sink_size: 8
refer_sink_swap: true
refer_sink_after_chunks: 1
refer_sink_injection_chunks: 1
refer_sink_start_slot: 0
refer_sink_num_slots: 2
```

This keeps LongLive2's default `sink_size=8` and starts with a two-slot
SPAWN-style subject KV swap. Ablate against the safer ShotStream setting
(`refer_sink_start_slot: 1`) to test whether preserving the first generated
sink slot is more stable than replacing the first two anchor slots.

### Suggested prompt / metadata format

Keep LongLive2's existing multi-shot prompt folder format and add optional
`refers` metadata inside each shot JSON:

```json
{
  "caption": "The same woman enters a quiet train carriage...",
  "refers": [
    {
      "image_path": "refs/woman_identity.png",
      "role": "identity",
      "swap_after_chunks": 1,
      "sink_start_slot": 1,
      "sink_num_slots": 2
    }
  ]
}
```

The metadata should be resolved relative to the JSON file directory, matching
the ShotStream refer-path behavior.

### Implementation checkpoints inside LongLive2

1. **Dataset / prompt loader**
   - Parse optional `refers` from each numbered shot JSON.
   - Expand shot-level refer metadata to chunk-level metadata using
     `shot_durations.txt`, just like prompts are expanded.
   - Preserve backward compatibility when no `refers` field exists.

2. **I2V / refer image encoder**
   - Load refer images through LongLive2's existing image preprocessing path.
   - Encode each refer image into the same latent/cache representation used by
     the sink write path.

3. **Pipeline sink swap**
   - At scene cut, allow LongLive2 to create/rebind the normal shot sink first.
   - After `refer_sink_after_chunks`, encode refer frames into a temporary cache.
   - Copy only `refer_sink_start_slot : refer_sink_start_slot + refer_sink_num_slots`
     into the active shot/global sink cache.
   - Do not advance global/local cache pointers while copying swapped slots.
   - Treat the swapped content as the subject image's KV cache, not raw pixels:
     the reference image is first encoded as a latent, then recached with
     compact RoPE, and only its generated K/V tensors are copied into the sink.
   - When `refer_sink_restore` is true, keep the subject KV swapped only for
     `refer_sink_injection_chunks` chunks, then restore the original sink KV.

4. **RoPE / text routing**
   - Re-apply compact RoPE to swapped refer sink slots using
     `refer_sink_rope_start_frame` or the existing sink RoPE offset.
   - Keep the current shot prompt as the cross-attention text condition unless a
     dedicated refer prompt is explicitly added later.

5. **Debug logging**
   - Print shot index, chunk index, refer image path, swapped sink slot range,
     and active sink sizes.
   - Add an option to dump selected refer frames and first generated sink frames
     for visual sanity checks.

## First ablation matrix

1. LongLive2 baseline I2V, no refer sink swap.
2. LongLive2 multi-shot sink only.
3. Refer sink swap at shot start, all sink slots (expected to over-condition).
4. Delayed refer sink swap after one generated chunk, slots `0:2`.
5. Delayed refer sink swap after one generated chunk, slots `1:3` only.
6. Refer in global sink only vs shot sink only.

The safest default to start with is experiment 4.
