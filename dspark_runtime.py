"""Runtime knobs for the LFM2.5-VL-3B DSpark drafter under mlx-vlm >= 0.7.2.

Lossless: the target still verifies every drafted token, so greedy output is
unchanged; these knobs only change how much the drafter sees and how wide it drafts.

  configure(draft_model, window=2048, policy="fixed")

window  - sliding draft window: the drafter attends only to the last `window`
          context tokens (mlx-vlm `draft_window_size`), INCLUDING the first round,
          where stock mlx-vlm hands the whole prompt to the drafter.
policy  - "fixed" (verify the requested block every round) or "adaptive"
          (mlx-vlm's acceptance-driven block controller).
"""
from __future__ import annotations


def configure(draft_model, *, window: int | None = None, policy: str = "fixed",
              initial_block: int | None = None):
    if policy not in ("fixed", "adaptive"):
        raise ValueError(policy)
    draft_model.prefer_requested_block_size = policy == "fixed"
    draft_model.dflash_initial_block_size = (initial_block or 4) if policy == "adaptive" else None

    draft_model.config.draft_window_size = int(window) if window else None
    orig = getattr(draft_model, "_dspark_orig_hidden", None) or draft_model._hidden
    draft_model._dspark_orig_hidden = orig

    if not window:
        draft_model._hidden = orig
        return draft_model

    W = int(window)

    def _hidden(inputs, target_hidden, cache):
        # First round: the drafter cache is empty and receives the whole prompt.
        # Keep only the last W positions. RoPE is relative, so positions of the
        # kept context and the draft block stay consistent with each other.
        if getattr(cache[0], "offset", 0) == 0 and target_hidden.shape[1] > W:
            target_hidden = target_hidden[:, -W:]
        return orig(inputs, target_hidden, cache)

    draft_model._hidden = _hidden
    return draft_model
