# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyễn Nhân Sâm · 2A202602672 · K4 Track 3A

## Kết quả đo thật

Cùng corpus và 20 câu gốc; generator/judge gpt-5.6-luna, local bge-m3, RAGAS 0.1.22. Baseline dùng lại sau kiểm chứng fingerprint; production v2 sinh mới toàn bộ answers. Mỗi pipeline đủ 20 LLM answers/80 scores hữu hạn, không còn lỗi chưa phục hồi. Production dùng Qdrant server. Tác vụ timeout, nếu có, được chấm lại riêng bằng cùng cấu hình; các scores thành công được giữ nguyên, lượt partial được lưu trong history.

| Metric | Baseline | Production v1 | Production v2 | v2 − v1 |
|---|---:|---:|---:|---:|
| faithfulness | 0.866667 | 0.755952 | 0.938214 | +0.182262 |
| answer_relevancy | 0.800456 | 0.832710 | 0.856665 | +0.023955 |
| context_precision | 0.866667 | 0.975000 | 0.975000 | +0.000000 |
| context_recall | 0.800000 | 0.825000 | 0.875000 | +0.050000 |

V1→v2 đồng thời sửa cách biểu diễn source trong evaluator và thay đổi retrieval/generation. Không gán toàn bộ chênh lệch Faithfulness cho retrieval; đây là một lượt judge, chưa có ablation toàn bộ bộ câu hoặc khoảng tin cậy. Baseline trước đó đã dùng context nhất quán giữa generator/judge.

## Nguyên nhân đã xác minh và sửa

1. **Boundary source:** v1 generator nhận filename nhưng judge chỉ nhận parent raw. Trace case mật khẩu 12 ký tự chấp nhận số 12 và từ chối citation filename. Giữ nguyên answer, gắn metadata nguồn thật làm diagnostic Faithfulness 0,5→1,0. V2 dùng `_evidence_contexts()` cho cả hai bên; checkpoint giữ nguyên raw parent.
2. **Coverage nhiều ý:** v1 tìm được salary child nhưng rerank xếp thứ 12; parent-only probe vẫn thứ 5. Planner tách từng ý và chỉ giữ điều kiện liên quan; pipeline dành một parent mỗi facet, dedup và giữ tối đa 3 parent. Planner chỉ nhận query, không nhận golden. Đây là heuristic, không bảo đảm coverage mọi câu.

## Bottom-5 theo trung bình bốn metrics

Năm câu dưới đây lấy trực tiếp từ `failures` của report v2, không chọn theo kết luận mong muốn. Ground truth/corpus giữ nguyên; diagnosis phân biệt chứng cứ đọc tay và giả thuyết về judge.

### #1 — Câu 17

**Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

**Expected (golden gốc):** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).

**Got (answer thật):**

> Theo **tam_ung.md**, phí quá hạn là:
> 
> **15.000.000 × 2%/tháng = 300.000 VNĐ/tháng**
> 
> Khoản này bị khấu trừ vào lương tháng kế tiếp. Chính sách không quy định cách tính phí theo riêng **5 ngày quá hạn**, nên không xác định được mức phí chính xác theo ngày.

**Mean:** 0.625000; **Worst metric:** answer_relevancy.

faithfulness: 1.000000; answer_relevancy: 0.000000; context_precision: 1.000000; context_recall: 0.500000.

**Sources:** tam_ung.md, chi_phi_expense.md, nghi_phep_nam_v2024.md.

**Evidence:** tam_ung.md quy định thanh toán trong 15 ngày; sau đó tính phí 2%/tháng trên số tiền chưa hoàn ứng và khấu trừ vào lương. Nguồn không quy định prorate theo 5 ngày.

**Diagnosis:** Answer nêu đúng mức phí tháng và nói rõ không xác định được phí theo ngày. Điểm Answer Relevancy thấp có thể do không chốt một con số theo golden; đây là khác biệt giữa dữ kiện nguồn và giả định pro-rata trong câu trả lời mẫu.

**Error Tree:** Answer thiếu 50.000? → Context có 2%/tháng? Có → Context có công thức theo ngày? Không → Không gán lỗi retrieval khi policy thiếu công thức.

**Suggested fix:** Giữ câu trả lời có điều kiện; nếu sản phẩm cần prorate, xác nhận owner chính sách và ghi rõ giả định tháng 30 ngày trước khi tính.

### #2 — Câu 8

**Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?

**Expected (golden gốc):** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.

**Got (answer thật):**

> Có. Tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ. Phương thức được chấp nhận: ứng dụng Authenticator, SMS OTP hoặc YubiKey.  
> **Nguồn:** `mat_khau_v2.md`

**Mean:** 0.797288; **Worst metric:** context_recall.

faithfulness: 1.000000; answer_relevancy: 0.689152; context_precision: 1.000000; context_recall: 0.500000.

**Sources:** mat_khau_v2.md, vpn_truy_cap.md, mua_sam.md.

**Evidence:** mat_khau_v2.md ghi bắt buộc MFA cho email, VPN và hệ thống nội bộ. Golden còn nêu v1.0 không yêu cầu MFA; v1 bị filter vì query hỏi chính sách hiện hành.

**Diagnosis:** Context đủ quy định hiện hành nhưng thiếu mệnh đề lịch sử mà golden yêu cầu. Recall 0,5 phù hợp khác biệt phạm vi current-policy/lịch sử; chưa có trace Recall của lượt v2 để quy kết chính xác. Relevancy thấp cũng cần kiểm tra trace, không kết luận answer sai.

**Error Tree:** Recall thấp → Context có MFA hiện hành? Có → Golden yêu cầu cả bản cũ? Có → Context không có v1 → Phân biệt coverage lịch sử và correctness hiện hành.

**Suggested fix:** Giữ filter hiện hành; nếu người dùng hỏi so sánh, lấy cả hai bản và gắn nhãn hết hiệu lực. Đánh giá câu lịch sử riêng, không thêm policy cũ vào câu hiện hành chỉ để tăng điểm.

### #3 — Câu 13

**Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?

**Expected (golden gốc):** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.

**Got (answer thật):**

> - **Người phê duyệt:** Giám đốc phòng ban (Director), vì đơn hàng **30.000.000 VNĐ** thuộc mức từ 5–50 triệu VNĐ.  
> - **Từ phòng CNTT:** Cần **xác nhận cấu hình kỹ thuật trước khi đề xuất mua sắm**.  
> - Vì đơn hàng trên **10.000.000 VNĐ**, cần đính kèm **ít nhất 3 báo giá**.
> 
> *Nguồn: `mua_sam.md`*

**Mean:** 0.831649; **Worst metric:** faithfulness.

faithfulness: 0.500000; answer_relevancy: 0.826596; context_precision: 1.000000; context_recall: 1.000000.

**Sources:** mua_sam.md, phan_loai_du_lieu.md, dao_tao_noi_bo.md.

**Evidence:** mua_sam.md quy định 5–50 triệu cần Director, trên 10 triệu cần ít nhất 3 báo giá, và thiết bị CNTT cần xác nhận cấu hình từ phòng CNTT. Answer nêu đủ ba điều.

**Diagnosis:** Faithfulness thấp dù nội dung có trong source; có thể do phép áp ngưỡng 30 triệu và citation bị tách statement. Cần trace NLI để phân biệt judge variance với claim không được hỗ trợ.

**Error Tree:** Faithfulness thấp → Ba quy định có trong context? Có → Số 30 triệu là premise câu hỏi → Kiểm tra statement decomposition và source label.

**Suggested fix:** Giữ phép áp ngưỡng nhưng đánh dấu số tiền là premise người hỏi; lưu trace judge và dùng source framing chung đã sửa.

### #4 — Câu 5

**Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

**Expected (golden gốc):** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.

**Got (answer thật):**

> Từ **3 năm thâm niên trở lên**, nhân viên được cộng thêm **1 ngày phép cho mỗi 3 năm làm việc liên tục**.  
> Nguồn: **nghi_phep_nam_v2024.md**

**Mean:** 0.842670; **Worst metric:** context_recall.

faithfulness: 1.000000; answer_relevancy: 0.870680; context_precision: 1.000000; context_recall: 0.500000.

**Sources:** nghi_phep_nam_v2024.md, nghi_phep_khong_luong.md, nghi_phep_dac_biet.md.

**Evidence:** nghi_phep_nam_v2024.md ghi từ 3 năm được cộng 1 ngày cho mỗi 3 năm liên tục. Golden còn yêu cầu quy tắc 5 năm của v2023, không nằm trong context hiện hành.

**Diagnosis:** Answer đúng quy tắc hiện hành; thiếu historical coverage so với golden. Recall 0,5 phù hợp việc không lấy v2023 sau version filter; không có trace Recall để xác định đóng góp từng statement.

**Error Tree:** Recall thấp → Context có quy tắc 3 năm? Có → Golden thêm 5 năm bản cũ? Có → Query không hỏi lịch sử → Kiểm tra phạm vi đánh giá trước khi thay version filter.

**Suggested fix:** Đánh giá version comparison riêng và giữ nhãn nguồn cũ khi query yêu cầu. Không bỏ filter hoặc thay golden để tối ưu điểm hiện hành.

### #5 — Câu 18

**Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?

**Expected (golden gốc):** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.

**Got (answer thật):**

> Lương thử việc Junior mức cao nhất:
> 
> **20.000.000 × 85% = 17.000.000 VNĐ/tháng**
> 
> Theo **[Nguồn: bang_luong_2024.md]** và **[Nguồn: thu_viec.md]**.

**Mean:** 0.865097; **Worst metric:** context_precision.

faithfulness: 1.000000; answer_relevancy: 0.960389; context_precision: 0.500000; context_recall: 1.000000.

**Sources:** thu_viec.md, bang_luong_2024.md, hoan_chi_dao_tao.md.

**Evidence:** thu_viec.md quy định lương thử việc bằng 85% lương cấp bậc; bang_luong_2024.md cung cấp Senior/Junior và mức cao nhất của Junior. Answer tính 20.000.000 × 85% = 17.000.000 VNĐ/tháng.

**Diagnosis:** Answer tính đúng 17 triệu với hai nguồn cần thiết; parent hoàn chi là nhiễu không liên quan. Tuy nhiên context phụ ở cuối không tự giải thích RAGAS average precision 0,5: judge có thể đánh giá thiếu tính hữu ích của nguồn tỷ lệ 85% hoặc phép kết hợp. Chưa có verdict Precision v2 để xác định chính xác.

**Error Tree:** Precision thấp → Có cả 85% và mức Junior? Có → Có parent phụ không liên quan? Có → Kiểm tra verdict hữu ích/thứ tự từng parent trước khi quy kết riêng top-k.

**Suggested fix:** Lưu trace context-precision cho từng source; thử thứ tự/cutoff context và bộ câu nhiều nguồn mới. Giảm parent nhiễu có ích nhưng không bảo đảm score này tăng.

## Kiểm chứng câu Senior 9 năm

**Nguồn v2:** nghi_phep_nam_v2024.md, bang_luong_2024.md, nghi_phep_khong_luong.md.

> - **Phép năm:** 18 ngày  
>   - Công thức: 15 ngày cơ bản + (9 năm ÷ 3) × 1 ngày = **18 ngày**.  
>   - Nguồn: `nghi_phep_nam_v2024.md`
> 
> - **Lương gross:** **20.000.000–35.000.000 VNĐ/tháng** đối với cấp Senior (P3–P4).  
>   - Nguồn: `bang_luong_2024.md`

faithfulness: 1.000000; answer_relevancy: 0.833202; context_precision: 1.000000; context_recall: 1.000000.

V1 không có bảng lương trong context. So sánh answer và sources v2 ở trên là kiểm chứng end-to-end của case này; không suy ra mọi câu nhiều ý đều được giải quyết.

## Giới hạn và bước tiếp theo

- NLI Faithfulness của RAGAS 0.1.22 không nhận question khi kiểm chứng statement. Trace hoàn chi đã từ chối số tiền/thời gian do người hỏi cung cấp; không đưa question/golden vào context để tăng điểm.
- Tạm ứng: nguồn chỉ có 2%/tháng, golden thêm prorate 5 ngày bằng tháng 30 ngày. Cần owner chính sách xác nhận công thức; không sửa corpus/golden lab.
- Password golden yêu cầu lịch sử bản cũ dù query hỏi hiện hành. Giữ version filter và nguồn hiện hành; đánh giá completeness lịch sử riêng.
- Đo trên bộ câu mới và ablation source formatting/planner/prompt; planner gọi thêm API khi chưa cache, rerank parent nhiều facet tăng CPU latency.

Raw evidence được kiểm tra bởi `scripts/verify_reports.py`; source labels chỉ là metadata gốc, không phải dữ kiện do LLM sinh. Chi tiết latency tại `reports/latency_breakdown.md`. Bản v1 và analysis cũ giữ tại `reports/history/v1/`. Chưa commit/push hoặc nộp LMS.
