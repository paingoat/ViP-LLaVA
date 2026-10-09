# Phản biện: attention map có bất khả thi để lấy bbox đích hay không

Tài liệu này kiểm tra ba luận điểm cho rằng pipeline sau là bất khả thi:

1. ViP-LLaVA-13B đọc một visual cue (vòng quanh vật nguồn, mũi tên trỏ tới chỗ trống).
2. Attention map của LLM cho biết model đang “nhìn” chỗ đích.
3. Map đó được chuyển thành bbox, rồi thành mask, để inpainting vật vào đúng vị trí và đúng phối cảnh.

Kết luận ngắn: pipeline nguyên dạng (attention → bbox → mask inpainting) không đứng vững. Lý do chắc nhất là **map không có biên của một vật chưa tồn tại**, và **cue trong ảnh không phải cách ViP-LLaVA được dạy dùng mũi tên**. Luận điểm “attention không bao giờ là những gì model nghĩ, nên mọi cách dùng attention để định vị đều bất khả thi” nói rộng hơn chứng cứ. Các bài mà thí nghiệm `EXP.md` đang dùng cho thấy một vài head định vị được **vật có biên**, đồng thời cũng cho thấy map trung bình và map chọn theo từng mẫu không đáng tin.

Phạm vi: ViP-LLaVA-13B (`mucai/vip-llava-13b`), CLIP ViT-L/14 ở 336×336, 576 image token trên lưới 24×24, và ba map trong `llava/eval/attention_maps.py`.

## Kết luận theo từng luận điểm

| Luận điểm | Phần đứng | Phần nói quá |
| --- | --- | --- |
| 1. Attention map không phải những gì model nghĩ | Trọng số attention không phải giải thích nhân quả. Map trung bình trên nhiều head/layer trộn sink, ngôn ngữ và vùng không liên quan câu hỏi. | “Chưa có phương pháp nào định vị được” không đúng với LLaVA-1.5-13B trên RefCOCO, khi head được chọn cố định trên dữ liệu có mask. Việc đó không suy ra được cho **chỗ trống không có vật**. |
| 2. Attention → bbox không có phối cảnh và không kiểm soát được | Bbox lấy từ map không mang độ sâu, hướng mặt phẳng, hay kích thước vật sẽ chèn. Ngưỡng hóa một heatmap 24×24 không có biên đúng. | Phối cảnh không phải việc của bbox trong các pipeline chèn vật. Phần vỡ riêng của attention là **extent**, không phải bản thân phép chiếu phối cảnh. |
| 3. Cue trong ảnh nằm ngoài phân phối ViP-LLaVA | Mũi tên huấn luyện ngắn, đầu mũi tên nằm trên vật đã annotate, và câu hỏi là “vật được mũi tên trỏ tới là gì”. Không có tác vụ “đưa A tới chỗ trống B”. | Từng mảnh (màu đỏ, mũi tên, đường bao quanh vật) đã xuất hiện lúc train. Lệch phân phối bác bỏ dùng zero-shot, không bác bỏ việc huấn luyện lại nếu có nhãn bbox. |

## Luận điểm 1. Attention map không cho biết model đang nghĩ gì

Luận điểm này đúng về **giải thích**, và đúng một phần về **định vị chỗ trống**. Nó không đúng nếu được đọc thành “mọi attention map của LLaVA đều vô dụng cho định vị”.

### Các họ phương pháp, nói ngắn

**Trọng số attention thô.** Một head ở một layer là ma trận softmax: mỗi query token chia một đơn vị trọng số cho các key. Jain và Wallace (NAACL 2019, *Attention is not Explanation*) chỉ ra có thể tìm một phân phối attention rất khác mà dự đoán gần như không đổi, nên trọng số không phải lời giải thích duy nhất. Wiegreffe và Pinter (EMNLP 2019, *Attention is not not Explanation*) bổ sung rằng attention vẫn có thể là một lời giải thích *hợp lý* trong một số phép thử, chỉ là không độc quyền. Serrano và Smith (ACL 2019, *Is Attention Interpretable?*) xóa các token được attend mạnh và thấy đầu ra đổi ít hơn kỳ vọng nếu trọng số cao đồng nghĩa với “token này quyết định câu trả lời”.

**Attention nhân với độ lớn vector.** Kobayashi và cộng sự (EMNLP 2020, *Attention is Not Only a Weight*) đo đóng góp bằng chuẩn của vector sau khi nhân trọng số. Token có trọng số lớn nhưng vector nhỏ gần như không đi vào residual stream. Đây là lý do một đỉnh heatmap có thể không ảnh hưởng gì tới chữ model viết ra.

**Gộp qua nhiều layer.** Abnar và Zuidema (ACL 2020, *Quantifying Attention Flow in Transformers*) chỉ ra attention một layer không phải luồng thông tin của cả mạng. Attention rollout/flow giả định cách trộn residual đơn giản, và giả định đó không khớp transformer thật. Trung bình “nửa layer giữa, mọi head” trong thí nghiệm này là một phép gộp heuristic, không phải một đại lượng đã được chứng minh là vùng model dùng.

**Relevancy có gradient.** Chefer, Gur và Wolf (CVPR 2021, *Transformer Interpretability Beyond Attention Visualization*; ICCV 2021 cho mô hình hai nhánh) nhân attention với gradient để lấy relevancy theo quyết định đầu ra. LVLM-Interpret dùng họ này. Relevancy gắn với một token đầu ra cụ thể hơn attention thô, nhưng vẫn là tương quan trên một forward/backward, không phải can thiệp nhân quả.

**Can thiệp.** Che token, RISE (Petsiuk và cộng sự, BMVC 2018), hoặc activation patching (Meng và cộng sự, NeurIPS 2022, ROME) hỏi “nếu vùng này đổi thì đầu ra có đổi không”. Đó là câu hỏi gần với “model có dùng vùng này không”. Cái giá là tốn compute, và ảnh bị che thường nằm ngoài phân phối model đã thấy (Adebayo và cộng sự, NeurIPS 2018, cho thấy một số saliency map không đổi khi trọng số model bị phá).

Trên LVLM còn hai nhiễu riêng. Darcet và cộng sự (ICLR 2024, *Vision Transformers Need Registers*) cho thấy ViT có các token chuẩn lớn, hút attention để gom thông tin toàn cục. Kang và cộng sự (ICLR 2025, bên dưới) tìm thấy hiện tượng tương tự ở image token của LLaVA. Chen và cộng sự (ECCV 2024, *An Image is Worth 1/2 Tokens After Layer 2*) cắt bớt image token sau layer 2 mà chất lượng giảm ít, nghĩa là attention lên ảnh ở các layer sau không đồng nghĩa với “model còn đang đọc ảnh ở đó”.

### Ba map trong thí nghiệm này

Cả ba đều lấy attention từ LLM lên 576 image token, bỏ visual sink, upsample bicubic từ lưới 24×24 về ảnh gốc. Query và cách gộp thì khác nhau. Mỗi cách có một bài báo, và bài đó không kết luận “đây là những gì model nghĩ”.

**Visual sink** (Kang, Kim, Kim, Hwang, ICLR 2025, *See What You Are Told: Visual Attention Sink in Large Multimodal Models*, arXiv:2503.03321). Một số image token nhận attention cao từ text ở vị trí cố định, không theo câu hỏi. Hidden state của chúng có kích hoạt khổng lồ trên cùng vài chiều với token BOS. Bài phát hiện sink khi \(\varphi(h)=\max_d |h_d|/\mathrm{RMS}(h) \ge \tau\), với \(\tau=20\) chọn trên LLaVA-1.5-7B. Với LLaVA-1.5-13B, hai chiều sink của LLM nền là \(\{2100, 4743\}\). Code thí nghiệm làm tương đương: lấy 2 chiều mà BOS có \(|h|/\mathrm{RMS}\) lớn nhất, rồi ngưỡng \(\tau=20\) trung bình trên dải layer. Knockout trên POPE (LLaVA-1.5-7B) cho thấy chặn attention từ sink sang text gần như không đổi điểm, trong khi chặn cùng số token ảnh ngẫu nhiên làm điểm tụt. Đóng góp \(\lVert \alpha \cdot x \cdot W_{OV}\rVert\) của sink cũng thấp. Trên Pascal-VOC và MS-COCO, khoảng 90–94% sink nằm ở nền. Bài viết thẳng: raw attention map không hoàn hảo để ground vật, và che sink chỉ để bớt nhiễu. Hệ quả cho ảnh kính–sách: một đỉnh nằm trên vùng giấy trống có thể là sink. Sink đúng kiểu “nền, ít thông tin”.

**Answer-token attention**, lấy cảm hứng từ LVLM-Interpret (Stan và cộng sự, CVPR 2024 Workshop, arXiv:2404.03118). Công cụ này cho xem attention thô **từng head, từng layer, từng token được chọn**, cộng relevancy map (Chefer) và giải thích nhân quả CLEANN trên layer cuối. Case study trên LLaVA-1.5-7B dùng MMVP: có mẫu relevancy vào chữ cao hơn vào ảnh, và model trả lời mâu thuẫn khi câu hỏi đổi còn ảnh giữ nguyên (xe rác, “cửa mở” và “cửa đóng” đều được nhận “yes”). Bài không đề xuất trung bình mọi token câu trả lời, mọi head, nửa layer giữa, rồi lấy đỉnh làm vị trí. Trung bình đó là thiết kế của `attention_maps.py`. Bất cập: các token sinh ra (“the”, “book”, “page”) attend những chỗ khác nhau; gộp chúng làm đỉnh không còn là “chỗ đích”.

**Relative attention** (Zhang, Khayatkhoei, Chhikara, Ilievski, ICLR 2025, *MLLMs Know Where to Look*, arXiv:2502.17422). Công thức trong bài:

\[
A_{\mathrm{rel}}(x, q) = \frac{A_{si}(x, q)}{A_{si}(x, q')}
\]

với \(q'\) đúng câu `"Write a general description of the image."`. \(A_{si}\) là attention của **token bắt đầu câu trả lời** lên ảnh, trung bình theo head. Với LLaVA, connector là MLP nên không có attention thứ hai ở connector. Họ chọn **một** layer trên tập held-out (LLaVA-1.5-7B: layer 14), rồi dùng map để **crop và zoom**, không để vẽ bbox vật. Cửa sổ crop có cạnh từ 1× đến 2× độ phân giải đầu vào; ảnh crop được resize và **nối thêm** vào token ảnh gốc. Trên TextVQA, LLaVA-1.5-7B tăng từ 47.80 lên 55.17 (rel-att) và 56.06 (grad-att). Attention ratio bên trong bbox ground truth lớn hơn 1 ở hầu hết layer, **kể cả khi câu trả lời sai**. Đó là kết quả mạnh nhất chống lại cách đọc “attention = kết luận của model”: model có thể nhìn đúng chỗ và vẫn viết sai. Bài cũng nêu giới hạn: câu hỏi quan hệ và đếm không được ViCrop cứu, vì map chỉ focus một vùng. Code thí nghiệm lệch bài ở hai chỗ có hậu quả: chia có thêm `0.01 × mean(A_generic)` (ổn định số, không có trong bài), và trung bình dải layer 10–29 của model 40 layer thay vì một layer đã chọn. Ablation của bài cho thấy trung bình mọi layer gần như không hại TextVQA (+0.28 điểm so với chọn layer), nên lệch này nhỏ hơn lệch về mục đích dùng map. ViP-LLaVA-13B cũng không phải bản 7B 32 layer mà bài đo.

**Localization heads** (Kang, Kim, Kim, Hwang, CVPR 2025, *Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding*, arXiv:2503.06287). Query là token text cuối của input. Họ loại 2 layer đầu, giữ head có tổng attention lên ảnh cao (ngưỡng ở điểm cong nhất, ví dụ \(\tau=0.24\) với LLaVA-1.5-7B), rồi chấm spatial entropy: nhị phân hóa tại trung bình, entropy của kích thước các vùng liên thông 8-hàng xóm. Head được chọn **cố định** theo tần suất lọt top-10 entropy thấp trên 1.000 mẫu RefCOCO train, không chọn lại mỗi ảnh. Với mọi model họ lấy \(k=3\). LLaVA-1.5-13B dùng L15 H39, L16 H30, L7 H2 (Figure 18). Map được làm mượt Gaussian, cộng lại, nhị phân tại trung bình, lấy convex hull lớn nhất, rồi bbox bao hull. Mask phân đoạn là bbox đó đưa vào SAM.

Số trên LLaVA-1.5-13B, Acc@0.5 (Table 1): RefCOCO val/testA/testB = 87.2/90.0/83.3, ngang Shikra-13B và Ferret-13B là các model được fine-tune để ground. RES (có SAM) RefCOCO val = 76.1 cIoU. Khi bắt chính model viết tọa độ, cùng họ LLaVA-1.5-13B chỉ đạt **5.28** trên RefCOCOg, trong khi head đạt 84.3 (Table 6). Tác giả viết localization head “might provide only indirect support when text generation unfolds in its usual course”: head định vị được, còn mạch sinh chữ thì không đọc head đó ra thành bbox.

Chỗ này đập vào code thí nghiệm. `localization_heads_attention` chọn 3 head **riêng cho từng run**: top 20% theo tổng attention lên ảnh, layer ≥ 2, rồi 3 head entropy thấp nhất. Bài gọi cách này là greedy. Table 5, RefCOCO val, LLaVA-1.5-13B, cả hai tiêu chí: greedy 67.4 REC / 63.8 RES, bộ head cố định 87.2 / 76.1. Văn bản mục 6.3: tiêu chí chỉ bảo đảm có cụm, không bảo đảm cụm bám chữ; greedy “may select heads that are localized but not text-referred”. Bộ head cố định của LLaVA-1.5-13B cũng không được phép mang sang ViP-LLaVA-13B mà không đo lại, vì stage 2–3 của ViP đã cập nhật trọng số.

Failure case của chính bài (mục 6.4, Figure 9): head L15 H39 của LLaVA-1.5-13B loang lên cả quả chuối thứ ba và thứ tư khi câu hỏi chỉ một quả. Acc@0.5 là ngưỡng lỏng. Mọi benchmark (RefCOCO, RefCOCO+, RefCOCOg, ReasonSeg) đều là **vật có mask**. Entropy được biện minh bằng “object patches tend to stay near each other”. Không có thí nghiệm nào về vùng trống.

### Phản biện đúng mức cho luận điểm 1

Câu “attention visualize ra chính là những gì model nghĩ” chưa được chứng minh, và các bài trên còn cho chứng cứ ngược: sink có attention cao nhưng đóng góp thấp; relative attention chỉ đúng vùng ngay cả khi câu trả lời sai; head định vị chỉ “hỗ trợ gián tiếp” cho việc sinh chữ; LVLM-Interpret có ca model bám chữ của câu hỏi hơn bám ảnh.

Câu “vì vậy không thể dùng attention để biết chỗ đích” mạnh hơn chứng cứ, nếu chỗ đích là một **vật**. Trên LLaVA-1.5-13B, ba head cố định đủ để bbox vượt IoU 0.5 khoảng 87% trên RefCOCO val, không cần train thêm. Điều kiện kèm theo rất hẹp: query là cụm chỉ một vật có biên, head được thống kê trên RefCOCO, bbox lấy từ hull sau ngưỡng trung bình, và mask đẹp là nhờ SAM bám biên vật.

Chỗ trống cuối mũi tên trong ảnh kính–sách không thỏa điều kiện đó. Không có vật để các patch “nằm gần nhau”, không có biên cho SAM bám, và nền giấy là đúng kiểu vùng mà sink chiếm. Đỉnh heatmap ở đó không phân biệt được “model hiểu đây là đích” với “token nền hút attention” hoặc “model đang nhìn mũi tên / trang sách”, tức những vật thật sự có trong ảnh. Luận điểm 1 thắng ở tác vụ này vì **đích không phải vật**, không phải vì attention chưa bao giờ định vị được gì.

## Luận điểm 2. Từ attention map sang bbox không có phối cảnh và khó kiểm soát

Luận điểm này gộp ba việc khác nhau. Chỉ một việc là lỗi của attention.

**Điểm rơi.** Relative attention và localization head được đo như tín hiệu “nhìn về vùng nào”: attention ratio so với bbox chữ, hoặc Acc@0.5 của một hộp thô. Một đỉnh hoặc một tâm vùng là đại lượng các bài này thực sự có. Với mũi tên, đại lượng hợp lệ nhất để hỏi là: đỉnh có rơi gần đầu mũi tên hơn so với vòng tròn, thân mũi tên, và các vật khác hay không. `peak_xy` trong thí nghiệm đang ghi đúng đại lượng đó. Nó không phải bbox.

**Extent.** Bbox của Kang và cộng sự là hình chữ nhật bao convex hull của vùng lớn hơn trung bình, sau Gaussian. Đổi \(\sigma\) làm Acc@0.5 trên RefCOCO val của LLaVA-1.5-13B dao động từ 84.3 đến 87.2. Lưới 24×24 trên ảnh 336px là ô khoảng 14px; bicubic không tạo thêm thông tin không gian. Với vật có mask, SAM gánh phần biên, và toàn bộ số RES (76.1 cIoU) đi qua SAM. Chỗ trống trên trang sách không có biên cho SAM. Kích thước blob attention là kích thước vùng được attend (trang giấy, đầu mũi tên, vệt sink), không phải kích thước kính sẽ chiếm ở độ sâu đó. Ngưỡng nào cũng cho một hộp, và không có tiêu chí để biết hộp nào đúng.

**Phối cảnh.** Hộp 2D không mã hóa độ sâu, hướng mặt phẳng, sự co ngắn, thứ tự che, hay bóng tiếp xúc. Kính nằm trên trang sách nghiêng cần đúng những thứ đó. Đây là giới hạn của **mọi** bbox, kể cả bbox người vẽ. Khảo sát image composition của Niu và cộng sự (arXiv:2106.14490) tách placement (vị trí, tỷ lệ, hình dạng) khỏi blending, harmonization và bóng. TopNet (CVPR 2023) dự đoán heatmap vị trí × tỷ lệ. OPA (Liu và cộng sự, arXiv:2107.01889) chấm composite sai vì sai kích thước, mất mặt đỡ, sai che khuất, và phối cảnh không nhất quán. Paint-by-Example, ObjectStitch (CVPR 2023) và AnyDoor (CVPR 2024) sinh **bên trong** mask: tỷ lệ vật đi theo hộp người đưa, bóng bị cắt ở biên mask (ObjectStitch nói rõ giới hạn này). ObjectDrop (ECCV 2024) mô hình được ảnh của vật lên cảnh, và nêu các ca hướng và ánh sáng của vật không khớp cảnh.

Vì vậy “bbox từ attention không có phối cảnh” đúng như mô tả hình học, nhưng không phải lý do riêng khiến attention thất bại. Các pipeline chèn vật vốn nhận một hộp thô rồi để model sinh lo phần nhìn. SEELE (Wang và cộng sự, TMLR 2024, arXiv:2401.16861) lấy tỷ lệ từ depth đơn ảnh (MiDaS) giữa vật và chỗ đặt. Ảnh này còn lợi thế: vật nguồn và đích nằm cùng một ảnh, cùng máy ảnh, nên tỷ lệ pixel xấp xỉ tỷ lệ độ sâu nếu có depth (Depth Anything, CVPR 2024; Depth Anything V2, NeurIPS 2024). Hướng mặt trang giấy là pháp tuyến của mặt đỡ, không nằm trong attention. ObjectMover (CVPR 2025) và GeoDiffuser (WACV 2025) xử lý đúng bài “dịch vật và chỉnh perspective”; input của chúng là vật và phép biến đổi, không phải heatmap.

Phần “hoàn toàn khó kiểm soát” đứng ở extent, không đứng ở phối cảnh. Một pipeline còn kiểm soát được nếu attention chỉ bị đòi một điểm (đầu mũi tên có nằm trong vùng đỉnh hay không), tỷ lệ lấy từ mask vật nguồn nhân tỷ lệ depth, hướng lấy từ mặt phẳng đỡ, và ảnh cuối để một model dịch/chèn vật chịu. Pipeline “ngưỡng hóa heatmap thành mask rồi inpaint” không có chốt nào trong ba chốt đó. *Thinking Outside the BBox* (ECCV 2024) nói thẳng rằng xin một mask đúng vị trí và đúng tỷ lệ là phần khó, và model train theo mask thì bị nhốt trong mask.

Ramrakhya và cộng sự (CVPR 2024, *Seeing the Unseen*) hỏi MLLM chỗ đặt một vật **chưa có trong ảnh**. GPT-4V và LLaVA trả lời hợp lý bằng lời, và đặt bbox lệch trong không gian ảnh. Đó cùng một vết với việc bắt ViP-LLaVA mô tả “vị trí mới so với vật xung quanh” trong prompt 3 của `EXP.md`: câu chữ có thể đúng trong khi hộp pixel sai.

## Luận điểm 3. Visual cue nằm ngoài phân phối huấn luyện

Đây là luận điểm chắc nhất cho **zero-shot**. Nó không chứng minh fine-tune là bất khả thi.

### Mũi tên lúc train trỏ vào vật, và ngắn

`llava/visual_prompt_generator.py`, `draw_arrow`: tâm bbox cộng nhiễu \(\pm 25\%\) mỗi chiều, rồi vẽ mũi tên **có đầu ở gần tâm đó**. Góc ngẫu nhiên \(0\)–\(2\pi\). Độ dài Uniform từ \(0.8\) lần cạnh ngắn của bbox đến `max_arrow_length`. Caller đặt `max_arrow_length = 50 * max(W,H) / 336`, tức tối đa khoảng 15% cạnh dài ảnh. Đuôi không gắn vào prompt nào khác. Một nửa số mũi tên có điểm gãy nhỏ để giả nét tay. Paper (Cai và cộng sự, CVPR 2024, arXiv:2312.00784, mục 3.2 và 5.1) nói đầu mũi tên, cả point, triangle và scribble nằm trong mask; chiều và độ dài mũi tên được random nhưng đầu vẫn trong mask.

Cụm từ đi kèm trong `words_shape` là “pointed to by the arrow”: mũi tên là cách **chỉ một instance đã có hộp**. Không có mẫu nào mũi tên nghĩa là “chỗ sẽ đặt vật” hay “đưa vật từ A tới B”.

Trong ảnh đính kèm, mũi tên dài hơn nhiều so với trần 15% cạnh, đầu mũi tên nằm trên vùng giấy không phải một instance được annotate, và đuôi xuất phát từ vòng quanh kính. Ba điểm này cùng lệch với generator.

### Vòng tự do không phải một trong tám kiểu prompt

Tám kiểu: rectangle, ellipse, triangle, point, scribble, mask contour, mask, arrow. Ellipse là đường bao **trục song song**, phóng 1–1.5 lần bbox hoặc bounds của mask. Mask contour bám đa giác vật thể. Không có vòng kín tự do, lớn hơn vật, rồi nối thành một ký hiệu với mũi tên.

Mũi tên chỉ được bật ở RefCOCOg, VCR và Flickr30k (`visual_prompt_config`). Dữ liệu quan hệ hai vùng của Visual Genome **tắt mũi tên** (dòng arrow bị comment, chỉ còn rectangle và ellipse). Đúng tập mà lẽ ra phải dạy “hai dấu hiệu trong một ảnh”, model chỉ thấy hộp và ellipse.

Hai dấu cùng màu đỏ không phải phần lệch chính. Màu được gán theo từng shape: hai shape khác nhau có thể cùng tên màu. Phần lệch là **hai dấu hợp thành một chỉ dẫn**. Lúc train, mỗi prompt là một instance độc lập. Câu hỏi quan hệ có dạng “subject within the red rectangle and the object within the blue ellipse”, rồi trả về một triplet. Stage 3 (khoảng 13K mẫu GPT-4V) vẫn thay placeholder bbox bằng các cụm chỉ vùng đó, không dạy chuyển động.

Phụ lục bài có một eval mũi tên nối hai vật: đầu và đuôi là tâm hai bbox, câu hỏi chỉ là vật nào nằm ở **đầu** mũi tên. ViP-LLaVA-13B đạt khoảng 90% trên 3.520 cặp. Cả hai đầu đều là vật khác loại, và nhãn không phải tọa độ đích. Kết quả này ủng hộ “model đọc được hướng mũi tên giữa hai vật”, và không ủng hộ “model biết một điểm trống để đặt vật”.

### ViP-Bench không đo tác vụ này

ViP-Bench có 303 cặp ảnh–câu hỏi, chấm 0–10 bởi GPT-4. Hai chế độ: bbox tổng hợp và nét người vẽ (mũi tên, vòng). Bảng điểm 13B của bài: tổng bbox 48.3, tổng nét người vẽ 48.2. Hạng relationship chỉ 28 mục; nét người vẽ của 13B là 48.6. Lệch vài điểm trên 28 mục không kết luận được gì về vòng-nối-mũi-tên. Ablation theo kiểu nét (phụ lục, model 7B, prompt tổng hợp) cho arrow thấp hơn hoặc sát đáy so với ellipse và rectangle, trên Visual7W, PointQA và ViP-Bench. Bài không có attention map. Câu ở mục 3.1 rằng CLIP “focus attention” lên mũi tên và nét nguệch ngoạc được chống bằng điểm benchmark, không bằng heatmap.

Dữ liệu LLaVA-1.5 trộn vào stage 2 có nhiệm vụ **viết** bbox chuẩn hóa \([x_1,y_1,x_2,y_2]\) (Visual Genome trong mix 665K). ViP-LLaVA vì thế có thể trả lời một prompt xin tọa độ. Table 6 của bài localization head đo đúng hành vi này trên LLaVA-1.5-13B: 5.28 Acc@0.5 khi model tự viết hộp, 84.3 khi đọc head. Viết bbox bằng chữ không phải lối thoát cho cùng backbone.

### “Ngoài phân phối” không bằng “không huấn luyện được”

Kế hoạch ban đầu là huấn luyện ViP-LLaVA trên tác vụ này. Lệch phân phối của checkpoint hiện tại bác bỏ kỳ vọng zero-shot, và dự đoán prompt 3 trong `EXP.md` sẽ yếu. Nó không bác bỏ fine-tune nếu có ảnh, câu hỏi, và **nhãn hình học**. Generator hiện tại cũng không sinh được dữ liệu đó: `draw_arrow` đặt đầu mũi tên lên bbox nguồn, không có tham số “đuôi ở vật A, đầu ở điểm trống B”.

Fine-tune bằng câu trả lời chữ (“kính nằm trên trang bên phải, phía trên đồng hồ”) không biến attention thành bbox đã hiệu chuẩn. Bài localization head nói head chỉ hỗ trợ gián tiếp việc sinh chữ. Muốn bbox, nhãn phải là bbox hoặc mask, và đầu ra phải là hộp được giám sát, hoặc một head grounding được đo lại trên chính kiểu đích này. Attention readout của checkpoint chat không trở thành bộ định vị chỉ vì câu trả lời chữ đúng hơn.

`EXP.md` vẫn là thí nghiệm đúng cho một câu hẹp hơn: model có **nói** đúng vật trong vòng, đúng chỗ đầu mũi tên, và đúng quan hệ sau khi “dịch” hay không. Ba heatmap là chẩn đoán phụ. Chúng không phải module lấy mask.

## Việc thí nghiệm đang làm so với bài nó trích

| Việc trong `attention_maps.py` | Bài gốc làm | Hệ quả |
| --- | --- | --- |
| Answer-token: trung bình mọi token sinh ra, mọi head, layer giữa | LVLM-Interpret xem từng head, từng token; khuyến nghị relevancy và CLEANN vì attention thô thiếu | Đỉnh map không gắn với một quyết định |
| Relative: trung bình layer 10–29, cộng \(\epsilon\) ở mẫu số | Một layer (với 7B là layer 14); chia đúng; map để crop 1–2× ảnh rồi nối token | Gần đúng về công thức, khác mục đích. Bài không xuất bbox đặt vật |
| Localization: 3 head entropy thấp nhất **từng run** | 3 head cố định theo tần suất trên RefCOCO; greedy kém hơn rõ (67.4 so với 87.2 REC) | Đúng biến thể mà bài báo là không bảo đảm bám chữ |
| Bỏ sink \(\tau=20\) | Đúng hướng bài sink; chiều sink 13B trong bài là \(\{2100, 4743\}\) | Nên ghi lại `sink_dims` từng run (JSON đã ghi) và đối chiếu hai chiều này |
| Upsample bicubic, lấy `argmax` làm đỉnh | Bài head: Gaussian, ngưỡng trung bình, hull, SAM | Đỉnh là metric hợp lý; hull/SAM không có nghĩa khi đích không có vật |

Chưa có bài nào trong danh sách trên chạy trên ViP-LLaVA. Mọi số RefCOCO là của LLaVA-1.5. Stage 2–3 của ViP đổi trọng số, nên bộ head L15 H39 / L16 H30 / L7 H2 chỉ là giả thuyết cần đo lại, không phải kết quả chuyển giao.

## Kết luận cho hướng đi

Ba luận điểm, sau khi siết lại cho khớp chứng cứ:

1. Attention không phải bản ghi suy luận. Các bài thí nghiệm đang trích còn mạnh hơn thế ở một điểm và yếu hơn ở một điểm. Mạnh hơn: nhìn đúng chỗ vẫn trả lời sai, và attention cao có thể không có đóng góp. Yếu hơn: với vật có biên, vài head của LLaVA-1.5-13B là bộ định vị training-free thật (87.2 Acc@0.5), với điều kiện chọn head trên RefCOCO và chấm IoU lỏng. Điều kiện đó không có ở một điểm trống trên trang sách.

2. Chuyển map thành bbox không kiểm soát được **kích thước**. Phối cảnh là việc của depth và của model chèn/dịch vật, dù bbox đến từ attention hay từ tay người. Inpaint trong mask sẽ phóng đại mọi hộp sai tỷ lệ.

3. Cue vòng nối mũi tên dài, đầu mũi tên trên chỗ trống, nghĩa là “đưa vật tới đây”, nằm ngoài cách ViP-LLaVA được dạy đọc visual prompt. Mũi tên train ngắn, đầu nằm trên vật, và câu hỏi là nhận diện vật đó. Đây là lý do zero-shot không đáng kỳ vọng. Đây không phải lý do không huấn luyện được, miễn là nhãn mới là hình học chứ không phải chỉ câu chữ, và generator prompt được viết lại.

Hướng còn đo được, nếu mục tiêu vẫn là đặt vật theo cue trong ảnh: dùng thí nghiệm hiện tại để xem câu trả lời chữ có bám cue không; chấm heatmap bằng điểm rơi so với đầu mũi tên, không bằng bbox; nếu cần hộp để chèn vật thì giám sát hộp trực tiếp, lấy tỷ lệ từ depth của vật nguồn và của điểm đích, và để phối cảnh cho một model dịch vật. Đọc attention của checkpoint chat rồi ngưỡng hóa thành mask là khâu không có bài nào trong chuỗi trích dẫn đang bảo vệ.
