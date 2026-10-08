# Exp1: suy luận theo deictic visual cue

Kiểm tra ViP-LLaVA-13B (`mucai/vip-llava-13b`, template `llava_v1`) có bám đúng dấu hiệu hình học đã vẽ sẵn trên ảnh khi trả lời, và đang nhìn vào đâu lúc đó.

## Thiết kế

Ảnh nằm trong `test/exp1/input`, đã overlay vòng tròn đỏ và mũi tên đỏ. Mỗi ảnh được ghép với từng prompt trong `test/exp1/prompts.json`. Một cặp ảnh–prompt là một run, hội thoại mới, không dùng lại câu trả lời của run trước.

Ba prompt tăng dần độ khó:

1. **Source Object Recognition.** Vật nằm trong vòng tròn đỏ là gì.
2. **Target Location Understanding.** Đầu mũi tên chỉ vào đâu so với các vật xung quanh.
3. **Compositional Spatial Reasoning.** Nếu đưa vật trong vòng tròn tới đầu mũi tên, vị trí mới của nó so với các vật xung quanh là gì.

Với 5 ảnh hiện có, một batch đầy đủ là 15 run. Decoding là greedy (`temperature` 0, `num_beams` 1, tối đa 512 token mới).

## Một run làm gì

Ảnh được pad thành hình vuông rồi resize về 336×336, thành 576 image token trên lưới 24×24. Model load một lần với attention `eager` (bắt buộc để lấy trọng số attention; demo Gradio phải tắt trước vì 13B fp16 chiếm khoảng 27GB).

Với mỗi ảnh, script chạy trước một lượt prompt chung `"Write a general description of the image."` và giữ attention của token input cuối. Lượt này dùng chung cho cả 3 prompt của ảnh đó.

Với mỗi prompt:

1. Sinh câu trả lời.
2. Cho model đọc lại prompt cộng câu trả lời đó (`output_attentions`, `output_hidden_states`) để lấy attention từ các vị trí sinh câu trả lời lên 576 image token.
3. Vẽ heatmap rồi ghi kết quả.

## Ba attention map

Cả ba map đều bỏ visual sink token, cắt phần padding của `expand2square`, upsample bicubic về kích thước ảnh gốc, rồi chuẩn hóa về [0, 1]. Code ở `llava/eval/attention_maps.py`.

**Lọc sink** (ICLR 2025, "See What You Are Told"). Lấy 2 chiều mà token BOS có `|h| / RMS(h)` lớn nhất. Image token nào có cùng kiểu kích hoạt đó, trung bình trên dải layer, với ngưỡng `tau` mặc định 20, bị gán attention bằng 0 trên mọi map.

**Answer-token attention** (LVLM-Interpret, CVPR 2024 Workshop). Trung bình attention của mọi vị trí sinh ra token câu trả lời, mọi head, trên nửa layer giữa của LLM (mặc định layer 10–29 với model 40 layer). Cho biết model nhìn đâu trong lúc viết câu trả lời.

**Relative attention** (ICLR 2025, "MLLMs Know Where to Look"). Attention của token input cuối với prompt thí nghiệm, chia cho attention của cùng vị trí đó với prompt mô tả chung. Mẫu số cộng thêm `0.01 × mean(A_generic)` để token mà prompt chung gần như bỏ qua không bị phóng đại. Phần còn lại là vùng mà riêng câu hỏi thí nghiệm kéo sự chú ý tới.

**Localization heads** (CVPR 2025, "Only Needs A Few Attention Heads For Visual Grounding"). Từ token input cuối, lấy các head từ layer 2 trở đi nằm trong top 20% theo tổng attention lên ảnh. Trong nhóm đó chọn 3 head có spatial entropy thấp nhất (entropy kích thước các vùng liên thông sau khi ngưỡng tại giá trị trung bình), rồi trung bình 3 map đã chuẩn hóa. Head được chọn riêng cho từng run.

## Kết quả

Mỗi batch là một folder `test/exp1/output/YYYY-MM-DD_HH-mm-ss` (giờ UTC+7). Mỗi run có hai file `<tên ảnh>__<prompt_id>`:

- `.png`: ảnh input và 3 heatmap (colormap `turbo`, alpha 0.5, dấu `x` trắng là đỉnh), câu trả lời là caption.
- `.json`: prompt đã gắn template, câu trả lời, dải layer, chỉ số sink, head được chọn, tọa độ đỉnh từng map, thời gian chạy, commit git, GPU và phiên bản thư viện.

Chạy trên pod bằng `bash runpod/run_exp1.sh`. Chi tiết ở mục 6 của `GUIDE.md`.
