# Hướng dẫn chạy ViP-LLaVA trên RunPod (CUDA 12.8, RTX A6000)

## 1. Tạo pod

| Mục | Giá trị khuyến nghị |
| --- | --- |
| Template | RunPod PyTorch (CUDA 12.8) — bất kỳ image Ubuntu có driver ≥ 12.1 đều chạy được |
| GPU | 1× RTX A6000 (48GB) |
| Volume disk (`/workspace`) | ≥ 60GB cho 7B, ≥ 90GB nếu chạy thêm 13B |
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
3. Tạo `/workspace/data/{huggingface,logs}` và tải trọng số bằng `hf_transfer` vào `/workspace/data/huggingface`: `mucai/vip-llava-7b` (~13.5GB) và `openai/clip-vit-large-patch14-336` (~1.7GB).
4. Khởi động controller, model worker và Gradio, rồi in ra **Public link** (`https://xxxx.gradio.live`).

Các tuỳ chọn: `--skip-install`, `--skip-download`, `--no-launch`.

## 3. Sau khi restart pod

```bash
cd /workspace/ViP-LLaVA
bash runpod/start_demo.sh          # start/restart toàn bộ demo
bash runpod/start_demo.sh stop     # dừng
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

## 6. Xử lý sự cố

| Triệu chứng | Cách xử lý |
| --- | --- |
| Không thấy public link | `grep gradio.live /workspace/data/logs/gradio.out`; hoặc dùng `https://<POD_ID>-7860.proxy.runpod.net` (cần expose port 7860) |
| Dropdown không có model | Reload trang; xem `/workspace/data/logs/worker_*.out` |
| Hết VRAM | Đặt `LOAD_MODE=8bit` hoặc bớt model trong `MODEL_PATHS` |
| Lỗi khi cài lại thư viện | `conda env remove -n vip-llava` rồi chạy lại `bash setup_runpod.sh` |

## 7. Ghi chú về phiên bản thư viện

- `gradio==3.35.2` (không phải 4.16.0 như trong `pyproject.toml` gốc): `llava/serve/gradio_web_server.py` dùng API của Gradio 3.x (`gr.Button.update`, `tool="color-sketch"`, `concurrency_count`). Vì vậy phải pin kèm `pydantic<2`, `fastapi==0.104.1`, `websockets==11.0.3`.
- `numpy==1.26.4`: torch 2.1.2 và scikit-learn 1.2.2 không chạy được với NumPy 2.
- `huggingface_hub==0.25.2`: transformers 4.37.2 yêu cầu `<1.0`; phiên bản này tải qua `hf_transfer` khi `HF_HUB_ENABLE_HF_TRANSFER=1`.
- Demo Gradio chỉ hỗ trợ các checkpoint dùng Vicuna (`vip-llava-7b`, `vip-llava-13b`, `*-base`). Các bản Llama-3/Phi-3 cần template hội thoại khác.
- Không cài `flash-attn`/`deepspeed` vì chỉ cần cho training.
