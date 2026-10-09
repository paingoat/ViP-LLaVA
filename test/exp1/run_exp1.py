"""Exp1: deictic visual cue reasoning with ViP-LLaVA + attention maps.

Every image in --input-dir (visual prompt already overlaid) is paired with every prompt in --prompts; each pair is
one independent run (fresh conversation). Each run writes into test/exp1/output/<YYYY-MM-DD_HH-mm-ss>/:
    <image_stem>__<prompt_id>.png   input + 3 attention maps, answer as caption
    <image_stem>__<prompt_id>.json  run configuration and attention metadata

Runs on the GPU pod (see runpod/run_exp1.sh):
    python test/exp1/run_exp1.py [--model-path mucai/vip-llava-13b] [--limit 1]
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR))

from llava.constants import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.eval import attention_maps as am  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

GENERIC_PROMPT = "Write a general description of the image."
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RELATIVE_EPS_RATIO = 0.01
LOC_HEADS_MIN_LAYER = 2
OVERLAY = {"upsample": "bicubic", "colormap": "turbo", "alpha": 0.5}
PANEL_TITLES = {
    "answer_token": "Answer-token attention",
    "relative": "Relative attention (task / generic)",
    "localization_heads": "Localization heads",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="mucai/vip-llava-13b")
    parser.add_argument("--load-mode", choices=["fp16", "8bit", "4bit"], default=os.environ.get("LOAD_MODE", "fp16"))
    parser.add_argument("--input-dir", type=Path, default=EXP_DIR / "input")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "output")
    parser.add_argument("--prompts", type=Path, default=EXP_DIR / "prompts.json")
    parser.add_argument("--layers", default=None,
                        help="LLM layer band for answer-token/relative attention and sink detection, e.g. '10-29' "
                             "(default: middle half of the network)")
    parser.add_argument("--topk-heads", type=int, default=3)
    parser.add_argument("--candidate-frac", type=float, default=0.2,
                        help="Fraction of heads (by image attention) considered as localization-head candidates")
    parser.add_argument("--sink-tau", type=float, default=20.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tz-offset", type=float, default=7.0, help="UTC offset (hours) for the batch folder name")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many runs (smoke test)")
    return parser.parse_args()


def infer_conv_mode(model_name):
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "v1" in name or "vip-llava" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def build_prompt(question, conv_mode, mm_use_im_start_end):
    image_token = DEFAULT_IMAGE_TOKEN
    if mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], image_token + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def encode(prompt, tokenizer, device):
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


@torch.no_grad()
def generate_answer(model, tokenizer, input_ids, image_tensor, max_new_tokens):
    """Greedy decoding. Returns answer token ids, without the leading BOS or the EOS.

    LlavaLlamaForCausalLM.generate forwards only inputs_embeds. Transformers 4.37 then
    records a one-token BOS prefix in the returned ids. skip_special_tokens hides that
    prefix in the decoded string, but leaving it in the ids inserts a fake token between
    the prompt and the answer, so the answer-token attention rows are off by one.
    """
    output_ids = model.generate(input_ids, images=image_tensor, do_sample=False, num_beams=1,
                                max_new_tokens=max_new_tokens, use_cache=True)[0]
    if len(output_ids) == 0 or int(output_ids[0]) != tokenizer.bos_token_id:
        first = int(output_ids[0]) if len(output_ids) else None
        raise RuntimeError(
            "generate() did not prefix a single BOS token "
            f"(first id={first}, bos_token_id={tokenizer.bos_token_id}). "
            "Refusing to treat that id as an answer token."
        )
    output_ids = output_ids[1:]
    eos = (output_ids == tokenizer.eos_token_id).nonzero().flatten()
    return output_ids[:int(eos[0])] if len(eos) else output_ids


@torch.no_grad()
def forward_with_attention(model, input_ids, image_tensor):
    return model(input_ids=input_ids, images=image_tensor, output_attentions=True, output_hidden_states=True,
                 use_cache=False, return_dict=True)


def format_layers(layers):
    if layers == list(range(layers[0], layers[-1] + 1)):
        return f"{layers[0]}-{layers[-1]}"
    return ",".join(map(str, layers))


def wrap_text(text, width):
    paragraphs = text.splitlines() or [""]
    return "\n".join(textwrap.fill(p, width) if p.strip() else "" for p in paragraphs)


def render_figure(image, heatmaps, subtitles, input_subtitle, title, answer, path):
    num_panels = 1 + len(heatmaps)
    panel_w = 5.0
    panel_h = panel_w * image.height / image.width
    caption = wrap_text("Answer: " + (answer or "(empty answer)"), width=190)
    title_h = 1.4
    text_h = 0.2 * (caption.count("\n") + 1) + 0.4
    fig_w, fig_h = panel_w * num_panels, title_h + panel_h + text_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(2, num_panels, height_ratios=[panel_h, text_h], left=0.005, right=0.995,
                            bottom=0.005, top=1 - title_h / fig_h, hspace=0.05, wspace=0.02)
    axes = [fig.add_subplot(grid[0, i]) for i in range(num_panels)]

    axes[0].imshow(image)
    axes[0].set_title(f"Input (visual prompt overlaid)\n{input_subtitle}", fontsize=10)
    for ax, (key, heatmap) in zip(axes[1:], heatmaps.items()):
        ax.imshow(image)
        ax.imshow(heatmap, cmap=OVERLAY["colormap"], alpha=OVERLAY["alpha"], vmin=0, vmax=1)
        x, y = am.peak_xy(heatmap)
        ax.plot(x, y, marker="x", color="white", markersize=10, markeredgewidth=2.5)
        ax.set_title(f"{PANEL_TITLES[key]}\n{subtitles[key]}", fontsize=10)
    for ax in axes:
        ax.axis("off")

    text_ax = fig.add_subplot(grid[1, :])
    text_ax.axis("off")
    text_ax.text(0.005, 1.0, caption, va="top", ha="left", fontsize=10)
    fig.suptitle(title, fontsize=12, y=1 - 0.12 / fig_h, va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def repo_relative(path):
    try:
        return path.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(path)


def environment_info():
    def git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=REPO_DIR, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    tz = timezone(timedelta(hours=args.tz_offset))
    batch_name = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = args.output_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)
    print(f"[batch] {batch_dir}")

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name,
        load_8bit=args.load_mode == "8bit", load_4bit=args.load_mode == "4bit",
        attn_implementation="eager",
    )
    model.eval()

    conv_mode = infer_conv_mode(model_name)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_image_tokens = model.get_vision_tower().num_patches
    layers = am.parse_layer_band(args.layers, num_layers)
    layer_str = format_layers(layers)
    crop = image_processor.crop_size["height"]
    environment = environment_info()

    total = len(images) * len(prompts) if args.limit is None else min(args.limit, len(images) * len(prompts))
    done = 0
    for image_path in images:
        if done >= total:
            break
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        generic_ids = encode(build_prompt(GENERIC_PROMPT, conv_mode, mm_use_im_start_end), tokenizer, model.device)
        out = forward_with_attention(model, generic_ids, image_tensor)
        generic_last = am.image_attention(out.attentions,
                                          [am.expanded_length(generic_ids, num_image_tokens) - 1],
                                          am.find_image_span(generic_ids, num_image_tokens))
        del out
        torch.cuda.empty_cache()

        for prompt in prompts:
            if done >= total:
                break
            run_id = f"{image_path.stem}__{prompt['id']}"
            templated = build_prompt(prompt["text"], conv_mode, mm_use_im_start_end)
            input_ids = encode(templated, tokenizer, model.device)

            t0 = time.perf_counter()
            answer_ids = generate_answer(model, tokenizer, input_ids, image_tensor, args.max_new_tokens)
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            t1 = time.perf_counter()

            full_ids = torch.cat([input_ids, answer_ids[None].to(input_ids.device)], dim=1)
            out = forward_with_attention(model, full_ids, image_tensor)
            span = am.find_image_span(input_ids, num_image_tokens)
            prompt_len = am.expanded_length(input_ids, num_image_tokens)
            assert out.attentions[0].shape[-1] == prompt_len + len(answer_ids)
            # Row prompt_len - 1 + t is the position that predicts answer token t.
            answer_rows = list(range(prompt_len - 1, prompt_len - 1 + max(1, len(answer_ids))))
            answer_attn = am.image_attention(out.attentions, answer_rows, span)
            last_attn = am.image_attention(out.attentions, [prompt_len - 1], span)
            sinks = am.detect_visual_sinks(out.hidden_states, span, layers, tau=args.sink_tau)
            del out
            torch.cuda.empty_cache()

            sink_idx = sinks["indices"]
            loc_grid, loc_heads = am.localization_heads_attention(
                last_attn, sink_idx, topk=args.topk_heads, min_layer=LOC_HEADS_MIN_LAYER,
                candidate_frac=args.candidate_frac)
            grids = {
                "answer_token": am.answer_token_attention(answer_attn, layers, sink_idx),
                "relative": am.relative_attention(last_attn, generic_last, layers, sink_idx, RELATIVE_EPS_RATIO),
                "localization_heads": loc_grid,
            }
            heatmaps = {key: am.to_image_space(grid, image.size) for key, grid in grids.items()}
            peaks = {key: am.peak_xy(heatmap) for key, heatmap in heatmaps.items()}
            t2 = time.perf_counter()

            subtitles = {
                "answer_token": f"LLM layers {layer_str}, all heads, {len(answer_ids)} answer tokens",
                "relative": f"last input token, LLM layers {layer_str}",
                "localization_heads": "last input token, heads " + ", ".join(
                    f"L{h['layer']}H{h['head']}" for h in loc_heads),
            }
            title = f"{image_path.stem}  |  {prompt['name']} ({prompt['id']})\nQ: {prompt['text']}"
            figure_name = f"{run_id}.png"
            render_figure(image, heatmaps, subtitles, f"{len(sink_idx)} visual sink tokens masked", title, answer,
                          batch_dir / figure_name)

            config = {
                "run_id": run_id,
                "batch": batch_name,
                "timestamp": datetime.now(tz).isoformat(timespec="seconds"),
                "image": {
                    "path": repo_relative(image_path),
                    "width": image.width,
                    "height": image.height,
                    "preprocess": f"{getattr(model.config, 'image_aspect_ratio', 'square')} -> {crop}x{crop}",
                },
                "prompt": {"id": prompt["id"], "name": prompt["name"], "text": prompt["text"],
                           "templated": templated},
                "model": {
                    "model_path": args.model_path,
                    "model_name": model_name,
                    "conv_mode": conv_mode,
                    "load_mode": args.load_mode,
                    "dtype": "float16",
                    "attn_implementation": "eager",
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "num_image_tokens": num_image_tokens,
                },
                "decoding": {"do_sample": False, "temperature": 0.0, "num_beams": 1,
                             "max_new_tokens": args.max_new_tokens, "seed": args.seed},
                "answer": answer,
                "num_answer_tokens": len(answer_ids),
                "image_token_span": list(span),
                "attention": {
                    "layer_band": layers,
                    "visual_sinks": {
                        "method": "See What You Are Told: Visual Attention Sink in LMMs (ICLR 2025)",
                        **sinks,
                    },
                    "answer_token": {
                        "method": "Answer-token attention (LVLM-Interpret, CVPR 2024 W)",
                        "query": "positions predicting each answer token",
                        "aggregation": "mean over queries, all heads, layer band; sinks zeroed",
                        "peak_xy": peaks["answer_token"],
                    },
                    "relative": {
                        "method": "Relative attention (MLLMs Know Where to Look, ICLR 2025)",
                        "query": "last input token",
                        "generic_prompt": GENERIC_PROMPT,
                        "aggregation": "mean over all heads, layer band; A_task / (A_generic + eps_ratio * "
                                       "mean(A_generic)); sinks zeroed",
                        "eps_ratio": RELATIVE_EPS_RATIO,
                        "peak_xy": peaks["relative"],
                    },
                    "localization_heads": {
                        "method": "Localization heads (Only Needs A Few Attention Heads For Visual Grounding, "
                                  "CVPR 2025)",
                        "query": "last input token",
                        "min_layer": LOC_HEADS_MIN_LAYER,
                        "candidate_frac": args.candidate_frac,
                        "topk": args.topk_heads,
                        "selected_heads": loc_heads,
                        "peak_xy": peaks["localization_heads"],
                    },
                    "overlay": OVERLAY,
                },
                "runtime_sec": {"generate": round(t1 - t0, 3), "attention": round(t2 - t1, 3)},
                "outputs": {"figure": figure_name},
                "environment": environment,
            }
            (batch_dir / f"{run_id}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")

            done += 1
            print(f"[{done}/{total}] {run_id} ({t2 - t0:.1f}s): {answer[:100]!r}")

    print(f"[done] {done} runs -> {batch_dir}")


if __name__ == "__main__":
    main()
