"""Pre-download ViP-LLaVA checkpoints and the CLIP vision tower into the HF cache ($HF_HOME).

Usage:
    python runpod/download_weights.py                      # repos from $MODEL_PATHS
    python runpod/download_weights.py mucai/vip-llava-13b  # explicit repos
"""
import os
import sys

from huggingface_hub import snapshot_download

# Hard-coded in llava/model/multimodal_encoder/clip_4layer_encoder.py (mm_vision_tower: clip_4layers_336)
VISION_TOWER = "openai/clip-vit-large-patch14-336"

# Only PyTorch weights are loaded; skip TF/Flax/ONNX copies.
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "*.tflite", "tf_model*", "flax_model*", "rust_model*"]


def main():
    repos = sys.argv[1:] or [p.strip() for p in os.environ.get("MODEL_PATHS", "mucai/vip-llava-7b").split(",")]
    repos = [r for r in repos if r and not os.path.isdir(r)]
    repos.append(VISION_TOWER)

    print(f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')} "
          f"| HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")
    for repo in dict.fromkeys(repos):
        print(f"\n[download] {repo}")
        path = snapshot_download(repo_id=repo, ignore_patterns=IGNORE_PATTERNS)
        print(f"[done] {repo} -> {path}")


if __name__ == "__main__":
    main()
