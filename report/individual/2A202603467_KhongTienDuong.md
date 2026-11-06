# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603467
- **Họ và tên:** Khong Tien Duong
- **SEED** (in ở ô đầu notebook): **5102** (từ `STUDENT_ID = 2A202603467` → `✅ STUDENT_ID=2A202603467 → SEED=5102`)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ đã chạy Run all lần cuối, còn output; bộ kiểm tra 14/15 đạt — mục ❌ duy nhất: `4.1 + 4.4 call_openai / llm_classify_batch` |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ `SEED=5102 · thử thách #3`, đã lưu |
| ảnh tính tay có các bước | ☑ nộp dưới tên [`hand_calc1.png`](hand_calc1.png) và [`hand_calc2.png`](hand_calc2.png) — đủ 5 phép tính theo bộ kiểm tra: baseline_accuracy = 0.8659, σ(0) = 0.5, F1(TP=80, FP=20, FN=40) = 0.727, cosine([1,2,0],[2,1,2]) = 0.596, llm_cost_per_day = 10.5 USD/ngày |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (chạy của tôi, SEED=5102; F1 (test) đo trên 775 tin test sau khi bỏ trùng; SUB 40 = 20 spam + 20 ham lấy từ test; inference đo từ văn bản thô → nhãn, gọi từng tin một):

| Mô hình | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.901 | 0.919 | 0.090 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.944 | 0.974 | 0.222 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.889 | 1.000 | 125.504 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.962 | 0.976 | 40.302 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.946 | 0.976 | 39.437 | 0.0 |
| 4a–4d. LLM (Gemini 2.5 Flash / Flash-Lite) | — | — | — | — |

Hàng LLM không có số: các ô Phần 4 (4.A so model lớn/nhỏ, 4.B few-shot) và ô chạy 50 case khó đều dừng với `NameError: retry_with_backoff is not defined` (TODO 4.1 — cũng chính là mục kiểm tra ❌ duy nhất). Còn lại các mô hình 1, 2, 3b, 3c đều đạt SLA p95 < 50 ms/tin; cả 5 phương án trên bảng chi phí tiền là 0 USD/1k tin.

## 3. Dự đoán và thực tế

1. **`dl_relu` (03:24:00)** — *Đoán:* bỏ ReLU thì hai lớp Linear liên tiếp gộp thành một hàm tuyến tính nên F1 sẽ gần LogReg và **giảm** so với MLP có ReLU (0.944). *Thực tế:* **F1 = 0.944, không đổi** — phần tôi dự đoán sai. Linear∘Linear đúng là chỉ còn một hàm tuyến tính (biểu đạt lực không tăng), nhưng nó được Adam tối ưu với khởi tạo và lịch trình epoch khác (early stopping ở epoch có val loss thấp nhất) nên tìm được điểm riêng — trên đầu vào BoW thưa, bài toán gần tuyến tính nên một hàm tuyến tính tối ưu tốt vẫn chạm F1 0.944, thậm chí hơn LogReg tôi train 300 epoch (0.901). *Bài học:* đơn giản hoá kiến trúc không đồng nghĩa "F1 rớt về bằng hàng 1"; kết quả phụ thuộc cả quá trình tối ưu chứ không chỉ biểu đạt lực kiến trúc.
2. **`emb_tricky` (03:33:40)** — *Đoán:* Embedding + LogReg sẽ xử lý từ đồng nghĩa/ngữ cảnh tốt hơn BoW, và câu ham có "free/win" sẽ làm BoW nhầm. *Thực tế:* trên 6 tin thử, **cả hai đúng 5/6 và cùng sai đúng 1 tin ham** — "Last chance! Your account rewards expire tonight, confirm de…". *Vì sao lệch:* tôi tưởng embedding "thông minh hơn" sẽ tách bạch rõ ràng hơn; nhưng tin này vừa dùng đúng *văn mẫu khẩn cấp/thắng thưởng* của spam (ngữ nghĩa nghe như spam → vector nằm cạnh vùng spam), vừa không cần từ đồng nghĩa mới nào đặc biệt. Embedding không cứu được khi **nội dung câu** thực sự giống spam.
3. **`emb_search` (03:28:34)** — *Đoán:* query "you have been selected to receive a complimentary holiday, reply now to claim" sẽ nằm gần các SMS spam dù không dùng từ khoá quen. *Thực tế:* đúng hướng — 6 kết quả gần nhất đều là spam ("…4* Tenerife Holiday or #5000 CASH…", "you have won £1000 cash or a Spanish holiday! CALL NOW…"), **nhưng cosine chỉ ~0.53–0.56**, thấp nhiều so với tôi tưởng. *Vì sao lệch:* tôi nhầm "cùng chủ đề" với "vector gần đè nhau"; ngay giữa chính các tin spam, biểu diễn MiniLM vẫn tách rời đáng kể nên cosine 0.55 đã đủ xếp đầu bảng. Không thể lấy một ngưỡng cosine tuyệt đối cao (~0.8+) làm "điểm spam".

## 4. Phân tích 50 case khó

**Lưu ý minh bạch:** ô "Chạy mọi model trên 50 case khó" và ô "Đọc lỗi" không chạy được (`NameError: retry_with_backoff is not defined` kế thừa từ TODO 4.1; ô đọc lỗi còn phụ thuộc `names_m` chưa được gán). Vì vậy **không có bảng F1-hard thực chạy**; phân tích dưới đây dựa trên: (1) dự đoán đã ghi `hard_eval` lúc 03:42:48, (2) các lỗi thực tế BoW đã mắc trên test thường, (3) nội dung 50 case trong notebook.

- **Model dự kiến tụt nhiều nhất: `1. BoW + LogReg` (F1 0.901 trên test thường)**, nặng nhất ở nhóm *Né từ khoá (obfuscation)* và *Nói về spam (talks_about_spam)*. Tín hiệu đã có sẵn trên test thường: cả 10 tin model sai đều là spam đoán thành ham, ví dụ "Latest News! Police station toilet stolen, cops have nothing to go on!" (kiểu *talks_about_spam*) và "You have an important customer service announcement from PREMIER." (kiểu *transactional/ham_looks_spam*) — đúng bản chất của 2/7 nhóm khó.
- **Case (1) — cơ chế mặt chữ (obfuscation #30):** `"F R E E  e n t r y … t x t  WIN to 80086"` — tokenizer `[a-z0-9£$€]+` tách mỗi chữ cái thành token riêng ("f", "r", "e"…), không token nào khớp "free"/"txt"/"win" trong vocab → các trọng số spam mạnh nhất không được kích hoạt, tin bị đẩy về ham. Tương tự #29 "C0ngr4tulat10ns… g1ft c4rd" tạo token lai như "c0ngr4tulat10ns" — một từ chưa từng có trong vocab, dù từ gốc "congratulations" là tín hiệu spam mạnh.
- **Case (2) — cơ chế ngữ cảnh (manipulation #21):** `"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account?…"` — không có từ khoá spam nào; BoW chỉ thấy "mum", "phone", "number"… (một loạt từ kéo về ham theo bảng trọng số in ở Phần 1: "i, me, my…"); embedding cũng thấy câu *nghe như* tin nhắn thân mật → vector rơi vào vùng ham. Chỉ LLM "đọc như người" mới bám được chi tiết "send £200 to my friend's account" — nhưng Phần 4 chưa chạy được nên không xác minh được trên số liệu của tôi.
- **Điểm chung:** test thường "quá lạc quan" vì cùng phân phối train (và spam mẫu chỉ khác số điện thoại/mã thưởng vẫn còn trong dữ liệu), còn 50 case khó cố tình out-of-distribution. Mô hình nhìn văn bản kiểu nào thì gãy kiểu đó: BoW gãy ở biến đổi mặt chữ, embedding gãy ở câu ngữ nghĩa lẫn lộn, LLM (nếu chạy) dự kiến gãy ở nhóm *prompt_injection* #35–39 ("Ignore all previous instructions…").

## 5. Thử thách cá nhân

**Đề #3 (theo SEED 5102): Deep Learning — đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?**

*Cách làm:* giữ nguyên `build_mlp` / `train_mlp` của Phần 2, chỉ đổi số nơ-ron lớp ẩn; đặt `torch.manual_seed(SEED)` trước mỗi lần train; epochs=30, lr=1e-3, batch_size=64; so với cấu hình gốc 64 nơ-ron.

| Hidden | Tham số | F1 (test) | Thời gian train |
| --- | --- | --- | --- |
| 4 | 8,009 | 0.8889 | 5.11 s |
| 64 (gốc — mốc đối chiếu) | 128,129 | 0.944 | 7.2 s |
| 512 | 1,025,025 | 0.9326 | 28.99 s |

*Giải thích:* cả hai đầu đều tệ hơn mức 64 nơ-ron. Mạng 4 nơ-ron thiếu sức chứa (F1 tụt ~5.5 điểm — "quá nhỏ để tách các tổ hợp từ phức tạp"); mạng 512 nơ-ron tăng gấp 8 lần số tham số nhưng F1 lại **giảm** — dấu hiệu overfit trên dữ liệu nhỏ với đầu vào BoW thưa (khớp biểu đồ Phần 2: epoch tốt nhất chỉ là epoch 4), cộng thêm chi phí train ~4 lần. Vậy nên chọn mạng nhỏ khi: (1) tài nguyên/chi phí train bị giới hạn (CPU, cần retrain thường xuyên), (2) dữ liệu ít nhãn và đầu vào thưa — hàng 1 (BoW + LogReg) cho thấy 2.001 tham số tuyến tính đã đạt F1 0.901, nên capacity lớn rất dễ thành overfit, (3) cần nhanh một baseline để đối chiếu — chấp nhận mất vài điểm F1 lấy 6 lần ít tham số hơn và train nhanh hơn. Ngược lại, chỉ nâng lên 512 khi thấy train/val loss của mạng 64 vẫn giảm chậm và chưa chạm đáy, tức còn thiếu capacity thật sự.

## 6. Phương án cho sản phẩm thật

Yêu cầu: **p95 < 50 ms/tin**, **10 triệu SMS/ngày** (~116 tin/s trung bình, cao hơn vào giờ cao điểm) — với quy mô đó, cả **độ trễ** lẫn **chi phí tiền** đều sống còn, không riêng F1.

Số liệu quyết định (rút từ bảng Phần 5 của tôi, SEED 5102):

| Phương án | F1 (test) | p95 ms/tin | Đạt SLA? | Chi phí |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.901 | 0.090 | ✔ | 0 USD/1k tin |
| 2. BoW + MLP | 0.944 | 0.222 | ✔ (dư địa ~225 lần) | 0 USD/1k tin |
| 3a. MiniLM + kNN | 0.889 | 125.504 | ✘ | 0 |
| 3b. MiniLM + LogReg | 0.962 | 40.302 | ✔ (chỉ dư địa 1.24 lần) | 0 |
| 4. LLM (Gemini) | chưa đo được | p95 chắc chắn vượt mốc | kém khả thi | tính tay: 10.5 USD/ngày cho 1M tin |

**Chọn: `2. BoW + MLP` làm lớp lọc chính.** Dẫn chứng: F1 0.944 là cao nhất trong nhóm đạt SLA; p95 0.222 ms — nhỏ hơn mốc 50 ms hơn 200 lần; inference cả lô 43.267 tin/s nên 10 triệu tin/ngày chỉ cần một vài instance CPU; train chỉ 7.2 s, chi phí 0 USD/1k tin; 128 nghìn tham số nên retrain rất rẻ mỗi khi có nhãn mới. Có thể kết hợp tinh chỉnh ngưỡng quyết định (khác 0.5) để nâng recall mà không phá precision 0.977.

**Lớp thứ hai: 3b. MiniLM + LogReg cho các tin lưỡng lự.** F1 0.962 (cao nhất) nhưng p95 40.3 ms chỉ còn dư địa 1.24 lần SLA, dễ gãy khi tải cao điểm, và mỗi lần retrain phải encode lại toàn bộ train (113.8 s cho 3.619 tin) — nên chỉ chạy offline/chọn lọc chứ không đặt trên đường chính. **LLM loại ngay từ đầu:** tính tay của tôi cho 1.000.000 SMS/ngày (100 token input + 1 token output, giá $0.10/$0.50 mỗi 1M token) là **10.5 USD/ngày**, chưa kể độ trễ mạng mỗi lần gọi chắc chắn vượt p95 50 ms — chỉ giữ LLM làm lớp rà soát trực tuyến hoặc audit ngoài luồng.

Đánh đổi tôi chấp nhận: (1) vẫn bỏ sót ~9% spam thật (recall 0.914), cộng điểm yếu đã phân tích ở Phần 4 (né từ khoá, nói về spam) — có thể bù bằng bước tiền xử lý chuẩn hoá leetspeak trước model; (2) số p50/p95 đo trong Colab — trước khi commit SLA phải đo lại p95 trên hạ tầng thật; (3) spam chạy thật có biến chủng ngoài phân phối train, nên cần monitor recall trên production và retrain định kỳ, không tin con số test 0.944 là vĩnh viễn.

## 7. AI log và điều đã học

Lần 1 — *hỏi:* các công thức toán của Phần 1 (hàm sigmoid, gradient descent, BCE) và làm sao tự code gradient descent mà không cần sklearn. *Nhận được:* công thức `σ(z) = 1/(1+e^(−z))`, `grad_w = Xᵀ·error/n`, công thức BCE — nhưng bản nháp đầu bị tràn `exp` trả về `nan` và dấu gradient có lúc bị ngược. *Sửa:* thêm `np.clip(z, -500, 500)` và kiểm tra dấu bước cập nhật `w ← w − lr·grad_w` bằng cách train trên bộ toy 4 mẫu của ô kiểm tra 1.3, thấy loss giảm mới chấp nhận.

Lần 2 — *hỏi:* cách chỉnh Python khi gặp lỗi cú pháp / `NameError` khi chạy. *Nhận được:* cách đọc traceback từ dưới lên, và quy tắc "hàm bị gọi nhưng chưa định nghĩa → thiếu định nghĩa trước chỗ gọi". *Sửa:* tự phát hiện lỗi Phần 4 của tôi là thiếu hàm `retry_with_backoff` ở TODO 4.1, nên các ô 4.A/4.B và chạy 50 case khó dừng với `NameError: retry_with_backoff is not defined` — đã ghi nhận minh bạch trong mục 2 và mục 4 của báo cáo này thay vì che lỗi.

Lần 3 — *hỏi:* vì sao accuracy 98% vẫn vô nghĩa khi dữ liệu lệch lớp; precision/recall khác nhau thế nào. *Nhận được:* giải thích precision = TP/(TP+FP), recall = TP/(TP+FN), F1 là trung bình điều hoà — nhưng lời giải thích ban đầu khô khan, rối. *Sửa:* tôi kiểm chứng lại trên số liệu test của chính mình (accuracy 0.978 nhưng recall chỉ 0.828 — hơn 1/6 spam bị bỏ sót) rồi đối chiếu với 10 tin sai in ở Phần 1 để thấy chính xác con số này đã che sót spam kiểu nào.

Lần 4 — *hỏi:* PyTorch autograd hoạt động thế nào, vì sao cần `zero_grad()`, và vì sao phân loại LLM phải đặt `temperature=0`, `max_tokens` nhỏ (Phần 4 — vùng được dùng AI). *Nhận được:* giải thích đủ về autograd và nguy cơ output dài → tốn token/tiền. *Sửa:* viết lại giải thích bằng ngôn ngữ riêng ở "Giải thích 2", và tự bắt lỗi 3 — `parse_label_v2` dùng `"spam" in response` sẽ gán cả "not spam" thành spam.

**Điều hiểu ra hôm nay (2–3 câu):** Trước giờ tôi tin "điểm cao là model tốt" — giờ tôi hiểu một con số đơn lẻ có thể che giấu lỗi nghiêm trọng: accuracy 0.978 nhưng recall 0.828 nghĩa là lọt hơn 1/6 spam mỗi ngày, nên phải nhìn F1 và đọc lỗi bằng mắt thay vì chỉ tin accuracy. Thứ hai, gradient descent không huyền bí: chỉ là đo "mỗi tham số góp bao nhiêu vào lỗi" rồi chỉnh ngược chiều — cùng logic đó chạy từ LogReg viết tay (Phần 1) đến MLP PyTorch (Phần 2), khác nhau chỉ là autograd giúp tôi tính gradient. Thứ ba, embedding/LLM không phải phép màu: BoW nghèo 2.001 tham số đã đạt F1 0.901 gần như miễn phí, còn embedding/LLM chỉ lấy thêm vài điểm F1 bằng cái giá của độ trễ lớn — chọn model là chọn điểm trên đường Pareto theo yêu cầu thật (SLA, chi phí), không phải chọn "công nghệ hiện đại nhất".
