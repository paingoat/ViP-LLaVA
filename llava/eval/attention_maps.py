"""Training-free attention maps from the LLM of a LLaVA-style model onto its image tokens.

Methods:
- Visual sink filtering: "See What You Are Told: Visual Attention Sink in Large Multimodal Models" (ICLR 2025).
- Answer-token attention: mean attention of the answer-producing positions (as in LVLM-Interpret, CVPR 2024 W).
- Relative attention: "MLLMs Know Where to Look: Training-free Perception of Small Visual Details" (ICLR 2025).
- Localization heads: "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding"
  (CVPR 2025).

Attention tensors used here have shape [num_layers, num_heads, num_image_tokens] (float32, CPU), i.e. one
query (or a mean over queries) per head, restricted to the image-token keys.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from llava.constants import IMAGE_TOKEN_INDEX


def find_image_span(input_ids: torch.Tensor, num_image_tokens: int) -> Tuple[int, int]:
    """[start, end) of the image features in the expanded sequence built by `prepare_inputs_labels_for_multimodal`."""
    positions = (input_ids.flatten() == IMAGE_TOKEN_INDEX).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one image token, found {len(positions)}")
    start = int(positions[0])
    return start, start + num_image_tokens


def expanded_length(input_ids: torch.Tensor, num_image_tokens: int) -> int:
    return input_ids.numel() - 1 + num_image_tokens


def default_layer_band(num_layers: int) -> List[int]:
    return list(range(num_layers // 4, 3 * num_layers // 4))


def parse_layer_band(spec: Optional[str], num_layers: int) -> List[int]:
    """'10-29' (inclusive) or '8,12,16'; None -> middle half of the network."""
    if not spec:
        return default_layer_band(num_layers)
    layers = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    bad = [l for l in layers if not 0 <= l < num_layers]
    if bad:
        raise ValueError(f"Layers {bad} out of range [0, {num_layers})")
    return sorted(set(layers))


@torch.no_grad()
def image_attention(attentions: Sequence[torch.Tensor], query_rows: Sequence[int],
                    image_span: Tuple[int, int]) -> torch.Tensor:
    """Mean over `query_rows` of the attention to image tokens -> [L, H, N]."""
    start, end = image_span
    rows = torch.as_tensor(list(query_rows), device=attentions[0].device)
    return torch.stack([a[0][:, rows, start:end].float().mean(dim=1) for a in attentions]).cpu()


@torch.no_grad()
def detect_visual_sinks(hidden_states: Sequence[torch.Tensor], image_span: Tuple[int, int], layers: Sequence[int],
                        bos_index: int = 0, num_sink_dims: int = 2, tau: float = 20.0) -> Dict:
    """Visual sink tokens = image tokens whose residual stream shows the BOS-style massive activation.

    Sink dimensions are the `num_sink_dims` largest |h_d| / RMS(h) of the BOS token. An image token is a sink when
    phi(h) = max_{d in sink dims} |h_d| / RMS(h) >= tau, averaged over `layers` (`hidden_states[l]` is the input of
    decoder layer l).
    """
    start, end = image_span

    def rms_ratio(h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        return h.abs() / h.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

    bos_ratio = torch.stack([rms_ratio(hidden_states[l][0, bos_index]) for l in layers]).mean(dim=0)
    sink_dims = bos_ratio.topk(num_sink_dims).indices
    phi = torch.stack([rms_ratio(hidden_states[l][0, start:end])[:, sink_dims].max(dim=-1).values
                       for l in layers]).mean(dim=0)
    indices = (phi >= tau).nonzero().flatten()
    return {
        "indices": indices.tolist(),
        "sink_dims": sink_dims.tolist(),
        "phi_max": float(phi.max()),
        "tau": tau,
        "num_sink_dims": num_sink_dims,
    }


def _grid(values: torch.Tensor) -> np.ndarray:
    side = int(math.isqrt(values.numel()))
    if side * side != values.numel():
        raise ValueError(f"{values.numel()} image tokens do not form a square grid")
    return values.reshape(side, side).numpy()


def _drop_sinks(values: torch.Tensor, sinks: Sequence[int]) -> torch.Tensor:
    values = values.clone()
    if len(sinks):
        values[..., list(sinks)] = 0
    return values


def answer_token_attention(answer_attn: torch.Tensor, layers: Sequence[int], sinks: Sequence[int]) -> np.ndarray:
    """Mean over the layer band and all heads of the answer-token attention -> [side, side]."""
    return _grid(_drop_sinks(answer_attn[list(layers)].mean(dim=(0, 1)), sinks))


def relative_attention(task_attn: torch.Tensor, generic_attn: torch.Tensor, layers: Sequence[int],
                       sinks: Sequence[int], eps_ratio: float = 0.01) -> np.ndarray:
    """A(task prompt) / A(generic prompt) for the last input token, averaged over the layer band and heads.

    `eps_ratio * mean(A_generic)` is added to the denominator so tokens the generic prompt ignores don't explode.
    """
    task = task_attn[list(layers)].mean(dim=(0, 1))
    generic = generic_attn[list(layers)].mean(dim=(0, 1))
    return _grid(_drop_sinks(task / (generic + eps_ratio * generic.mean()), sinks))


def spatial_entropy(grid: np.ndarray) -> float:
    """Entropy of connected-component sizes after thresholding at the mean (8-connectivity); lower = more focused."""
    labels, num = ndimage.label(grid > grid.mean(), structure=np.ones((3, 3), dtype=int))
    if num == 0:
        return float("inf")
    sizes = np.bincount(labels.ravel())[1:].astype(np.float64)
    p = sizes / sizes.sum()
    return float(-(p * np.log(p)).sum())


def localization_heads_attention(last_attn: torch.Tensor, sinks: Sequence[int], topk: int = 3, min_layer: int = 2,
                                 candidate_frac: float = 0.2) -> Tuple[np.ndarray, List[Dict]]:
    """Average map of the `topk` heads with the lowest spatial entropy among the heads (layer >= min_layer) in the
    top `candidate_frac` by total attention on image tokens from the last input token."""
    num_layers, num_heads, _ = last_attn.shape
    image_sum = last_attn.sum(dim=-1)
    image_sum[:min_layer] = -1
    num_candidates = max(topk, int(round(candidate_frac * (num_layers - min_layer) * num_heads)))
    candidates = image_sum.flatten().topk(num_candidates).indices.tolist()

    scored = []
    for flat in candidates:
        layer, head = divmod(flat, num_heads)
        grid = _grid(_drop_sinks(last_attn[layer, head], sinks))
        if grid.sum() <= 0:
            continue
        scored.append((spatial_entropy(grid), -float(image_sum[layer, head]), layer, head, grid))
    scored.sort(key=lambda s: (s[0], s[1]))
    selected = scored[:topk]
    if not selected:
        raise RuntimeError("No localization head candidates with non-zero image attention")

    combined = np.mean([grid / grid.sum() for *_, grid in selected], axis=0)
    heads = [{"layer": layer, "head": head, "image_attention_sum": -neg_sum, "spatial_entropy": entropy}
             for entropy, neg_sum, layer, head, _ in selected]
    return combined, heads


def to_image_space(grid: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Map a grid over the `expand2square`-padded image back onto the original (W, H) image, normalized to [0, 1]."""
    width, height = image_size
    side = max(width, height)
    upsampled = F.interpolate(torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32))[None, None],
                              size=(side, side), mode="bicubic", align_corners=False)[0, 0]
    top, left = (side - height) // 2, (side - width) // 2
    heatmap = upsampled[top:top + height, left:left + width].clamp_min(0).numpy()
    lo, hi = heatmap.min(), heatmap.max()
    return (heatmap - lo) / (hi - lo) if hi > lo else np.zeros_like(heatmap)


def peak_xy(heatmap: np.ndarray) -> List[int]:
    y, x = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    return [int(x), int(y)]
