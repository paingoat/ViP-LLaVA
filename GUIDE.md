# Hướng dẫn chạy ViP-LLaVA trên RunPod (CUDA 12.8, RTX A6000)

## 1. Tạo pod

| Mục | Giá trị khuyến nghị |
| --- | --- |
| Template | RunPod PyTorch (CUDA 12.8) — bất kỳ image Ubuntu có driver ≥ 12.1 đều chạy được |
| GPU | 1× RTX A6000 (48GB) |
| Volume disk (`/workspace`) | ≥ 90GB cho 13B |
| Container disk | ≥ 20GB |
| Expose HTTP port | `7860` (tuỳ chọn, để dùng link RunPod proxy) |

Conda, môi trường Python và trọng số đều nằm trong `/workspace`, nên vẫn còn sau khi restart pod.

## 2. Setup (lần đầu)

```bash
cd /workspace
git clone <repo-cua-ban> ViP-LLaVA && cd ViP-LLaVA

cp .env.example .env
nano .env            # điền HF_TOKEN (tuỳ chọn), chọn MODEL_PATHS, LOAD_MODE

bash setup_runpod.sh
```

`setup_runpod.sh` chạy lần lượt các bước sau:

1. Cài Miniconda vào `/workspace/miniconda3`.
2. Tạo env `vip-llava` (Python 3.10), cài PyTorch 2.1.2 (cu121) và các thư viện đã pin trong `runpod/requirements.txt`, rồi `pip install -e .`.
3. Tạo `/workspace/data/{huggingface,logs}` và tải trọng số bằng `hf_transfer` vào `/workspace/data/huggingface`: `mucai/vip-llava-13b` (~26GB) và `openai/clip-vit-large-patch14-336` (~1.7GB).

Các tuỳ chọn: `--skip-install`, `--skip-download`.

Chạy demo sau khi setup xong:

```bash
bash runpod/start_gradio.sh
```

Script khởi động controller, model worker và Gradio, in ra **Public link** (`https://xxxx.gradio.live`), rồi hiển thị log ở foreground: mỗi lần người dùng tương tác (gửi câu hỏi, câu trả lời của model, regenerate, clear, vote) đều in ra terminal. Bấm **Ctrl+C** để dừng toàn bộ server.

## 3. Sau khi restart pod

```bash
cd /workspace/ViP-LLaVA
bash runpod/start_gradio.sh            # start/restart demo, rồi hiện log (Ctrl+C dừng server)
bash runpod/start_gradio.sh --detach   # chạy nền, terminal được trả lại
bash runpod/start_gradio.sh logs       # xem log của server đang chạy (Ctrl+C chỉ thoát xem log)
bash runpod/start_gradio.sh stop       # dừng
```

## 4. Cấu hình `.env`

| Biến | Ý nghĩa |
| --- | --- |
| `HF_TOKEN` | Token Hugging Face (không bắt buộc với repo `mucai/*`) |
| `DATA_DIR`, `HF_HOME` | Nơi cache trọng số (mặc định `/workspace/data/huggingface`) |
| `MODEL_PATHS` | Danh sách model, cách nhau bằng dấu phẩy, ví dụ `mucai/vip-llava-7b,mucai/vip-llava-13b` |
| `LOAD_MODE` | `fp16` (mặc định), `8bit` hoặc `4bit` |
| `GRADIO_SHARE` | `1` để tạo link public (hết hạn sau 72 giờ) |
| `GRADIO_PORT`, `CONTROLLER_PORT`, `WORKER_BASE_PORT` | Các port dịch vụ |

## 5. Dùng demo

1. Upload ảnh, rồi dùng cọ đỏ (color-sketch) khoanh hoặc trỏ vào vùng cần hỏi.
2. Gõ câu hỏi, ví dụ "What is the object within the red circle?", rồi bấm **Send**.
3. Khi đổi sang ảnh mới, bấm **Clear** trước.

## 6. Thí nghiệm exp1: deictic visual cue + attention map

Mỗi ảnh trong `test/exp1/input` (đã overlay sẵn vòng khoanh + mũi tên) được ghép với từng prompt trong `test/exp1/prompts.json`. Mỗi cặp là 1 run độc lập (hội thoại mới, greedy decoding). Kết quả nằm trong `test/exp1/output/YYYY-MM-DD_HH-mm-ss/` (giờ UTC+7), mỗi run gồm 2 file:

- `<ảnh>__<prompt_id>.png`: ảnh gốc + 3 attention map (dấu `x` trắng là đỉnh của map), câu trả lời làm caption.
- `<ảnh>__<prompt_id>.json`: cấu hình chạy (prompt, model, decoding, layer band, sink token, head được chọn, toạ độ đỉnh, phiên bản thư viện, git commit).

Ba attention map (đều không cần huấn luyện, code ở `llava/eval/attention_maps.py`):

| Panel | Phương pháp | Query |
| --- | --- | --- |
| Answer-token attention | Trung bình attention lên image token qua các layer giữa và mọi head (LVLM-Interpret, CVPR'24 W) | Các vị trí sinh ra từng token câu trả lời |
| Relative attention | Attention với prompt thí nghiệm chia cho attention với prompt chung "Write a general description of the image." (ICLR'25 "MLLMs Know Where to Look") | Token cuối của input |
| Localization heads | Top-k head có spatial entropy thấp nhất trong nhóm head chú ý nhiều nhất vào ảnh (CVPR'25 "Only Needs A Few Attention Heads") | Token cuối của input |

Mọi map đều bỏ các visual sink token (ICLR'25 "See What You Are Told"), cắt phần padding của `expand2square` rồi upsample về kích thước ảnh gốc.

Cách chạy (toàn bộ trên pod):

```bash
# Local: commit code + ảnh input, push branch attention
# Trên pod:
cd /workspace/ViP-LLaVA
git fetch && git checkout attention
git config --global user.name "<tên>" && git config --global user.email "<email>"
# git push cần GitHub Personal Access Token (dùng làm password, hoặc: git config --global credential.helper store)

bash runpod/run_exp1.sh --limit 1 --no-push   # smoke test 1 run, không commit
bash runpod/run_exp1.sh                       # 15 run (5 ảnh x 3 prompt), commit + push batch folder
```

Sau đó chạy `git pull` ở máy local để lấy batch folder. Script sẽ tắt demo Gradio để giải phóng VRAM; bật lại bằng `bash runpod/start_demo.sh`. Các tham số thêm của `test/exp1/run_exp1.py` (truyền qua `run_exp1.sh`): `--layers 10-29`, `--topk-heads 3`, `--candidate-frac 0.2`, `--sink-tau 20`, `--max-new-tokens 512`, `--tz-offset 7`.

## 7. Xử lý sự cố

| Triệu chứng | Cách xử lý |
| --- | --- |
| Không thấy public link | `grep gradio.live /workspace/data/logs/gradio.out`; hoặc dùng `https://<POD_ID>-7860.proxy.runpod.net` (cần expose port 7860) |
| Dropdown không có model | Reload trang; xem `/workspace/data/logs/worker_*.out` |
| Hết VRAM | Đặt `LOAD_MODE=8bit` hoặc bớt model trong `MODEL_PATHS` |
| Lỗi khi cài lại thư viện | `conda env remove -n vip-llava` rồi chạy lại `bash setup_runpod.sh` |

## 8. Ghi chú về phiên bản thư viện

- `gradio==3.35.2` (không phải 4.16.0 như trong `pyproject.toml` gốc): `llava/serve/gradio_web_server.py` dùng API của Gradio 3.x (`gr.Button.update`, `tool="color-sketch"`, `concurrency_count`). Vì vậy phải pin kèm `pydantic<2`, `fastapi==0.104.1`, `websockets==11.0.3`.
- `numpy==1.26.4`: torch 2.1.2 và scikit-learn 1.2.2 không chạy được với NumPy 2.
- `huggingface_hub==0.25.2`: transformers 4.37.2 yêu cầu `<1.0`; phiên bản này tải qua `hf_transfer` khi `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Demo Gradio chỉ hỗ trợ các checkpoint dùng Vicuna (`vip-llava-7b`, `vip-llava-13b`, `*-base`). Các bản Llama-3/Phi-3 cần template hội thoại khác.
- Không cài `flash-attn`/`deepspeed` vì chỉ cần cho training.
