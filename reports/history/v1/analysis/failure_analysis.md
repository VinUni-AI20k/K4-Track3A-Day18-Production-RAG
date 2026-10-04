# Failure Analysis — Lab 18: Production RAG

**Học viên:** NguyenNhanSam · 2A202602672 · K4 Track 3A

## Kết quả thật

Cùng 20 câu và corpus gốc; generator/judge gpt-5.6-luna, embedding local bge-m3, RAGAS 0.1.22. Cả baseline và production complete: 20/20 LLM answers, 80/80 scores hữu hạn, không generation/evaluation errors. Production dùng Qdrant server. Baseline được tái dùng; hash/provenance bổ sung hồi tố có ghi chú trong JSON, không gọi đây là baseline mới.

| Metric | Baseline | Production | Delta |
|---|---:|---:|---:|
| faithfulness | 0.866667 | 0.755952 | -0.110714 |
| answer_relevancy | 0.800456 | 0.832710 | +0.032254 |
| context_precision | 0.866667 | 0.975000 | +0.108333 |
| context_recall | 0.800000 | 0.825000 | +0.025000 |

Cả 4 metrics đạt ≥0,75, vượt điều kiện ≥3 metrics đạt 0,70 của rubric. Precision/Relevancy/Recall tăng nhưng Faithfulness giảm 0,110714. Không đạt bonus Faithfulness ≥0,85. Đây là bằng chứng một lượt đánh giá với judge đã chọn, chưa phải điểm giảng viên hoặc đánh giá trên dữ liệu ngoài lab.

## Bottom-5 theo trung bình bốn metrics

Xếp hạng trực tiếp từ scores của production report, không chọn thủ công case thuận lợi. Một case điểm thấp có thể trả lời đúng theo source; phần dưới phân biệt lỗi thực và giả thuyết về judge. Ground truth được giữ nguyên, không đưa vào generation.

### #1 — Câu 17

**Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

**Expected (golden gốc):** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).

**Got (answer thật):**

> Theo **tam_ung.md**:
> 
> - Hạn thanh toán: 15 ngày; nhân viên thanh toán sau 20 ngày nên quá hạn **5 ngày**.
> - Phí phạt: **2%/tháng trên số tiền chưa hoàn ứng**.
> - Nếu tính đủ 1 tháng: **15.000.000 × 2% = 300.000 VNĐ**.
> 
> Tài liệu không quy định cách tính phí theo số ngày quá hạn, nên **chưa xác định được mức phạt chính xác cho 5 ngày quá hạn**.

**Mean:** 0.375000; **Worst metric:** answer_relevancy.

faithfulness: 0.500000; answer_relevancy: 0.000000; context_precision: 1.000000; context_recall: 0.000000.

**Sources/context lấy được:** tam_ung.md, chi_phi_expense.md, nghi_phep_nam_v2024.md.

**Evidence:** tam_ung.md ghi hạn 15 ngày và phí 2%/tháng trên số tiền chưa hoàn ứng. Không có quy định quy đổi tháng thành 30 ngày hoặc pro-rata cho 5 ngày. 20−15=5 và 15.000.000×2%=300.000 là phép tính từ dữ kiện câu hỏi và mức phí trong tài liệu.

**Root cause / kết luận sau đọc evidence:** Có khác biệt giữa mức chi tiết ground truth và nguồn gốc: golden bổ sung pro-rata khoảng 50.000 VNĐ, trong khi nguồn không quy định cách tính theo ngày. Answer nêu đúng phí tháng và giới hạn thiếu dữ kiện. Relevancy/Recall bằng 0 là số đo judge, không tự chứng minh toàn bộ answer sai. Chưa có statement/verdict trace để xác nhận vì sao Faithfulness chỉ 0,5.

**Error Tree:** Answer không khớp đầy đủ golden → Context có hạn và mức phí? Có → Context có công thức pro-rata ngày? Không → Kiểm tra tính đầy đủ của policy/golden trước khi gán lỗi retrieval hoặc model.

**Suggested fix:** Giữ corpus/golden gốc trong lab. Ở project thực, hỏi owner chính sách cách prorate; chỉ tính 50.000 nếu nêu rõ giả định tháng 30 ngày và được phép dùng giả định. Lưu trace judge để phân biệt noncommittal/thiếu dữ kiện với hallucination.

### #2 — Câu 14

**Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?

**Expected (golden gốc):** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.

**Got (answer thật):**

> Nhân viên phải hoàn trả **25.000.000 VNĐ**, tương đương **100% chi phí đào tạo được tài trợ**, vì nghỉ việc sau 8 tháng, chưa đủ cam kết làm việc ít nhất 1 năm.
> 
> **Nguồn:** `hoan_chi_dao_tao.md` (Điều “Cam kết hoàn chi”).

**Mean:** 0.732404; **Worst metric:** faithfulness.

faithfulness: 0.166667; answer_relevancy: 0.762951; context_precision: 1.000000; context_recall: 1.000000.

**Sources/context lấy được:** hoan_chi_dao_tao.md, dao_tao_noi_bo.md, chi_phi_expense.md.

**Evidence:** hoan_chi_dao_tao.md, mục Cam kết hoàn chi: cam kết làm việc ít nhất 1 năm sau khóa học; nghỉ sớm phải hoàn trả 100% chi phí được tài trợ. Con số 25 triệu và 8 tháng nằm trong câu hỏi. Answer trả 25 triệu, khớp golden.

**Root cause / kết luận sau đọc evidence:** Đọc tay không thấy sai khoản hoàn trả. Faithfulness 1/6 là kết quả judge thấp bất thường so với answer và policy. Một giả thuyết là các dữ kiện từ câu hỏi/phép tính và citation bị coi là thiếu hỗ trợ trong context. Context report là nguyên văn parent, trong khi prompt generation bổ sung tên file nguồn; chưa có trace NLI để chứng minh giả thuyết này.

**Error Tree:** Faithfulness thấp → Answer có sai kết quả? Đọc tay: không → Context có 100%/1 năm? Có → Các dữ kiện 25 triệu/8 tháng từ đâu? Câu hỏi → Kiểm tra statement extraction, phép suy luận và cách biểu diễn source trong judge.

**Suggested fix:** Lưu statement/verdict của judge, đánh dấu rõ dữ kiện người hỏi và phép tính 100%×25 triệu, cung cấp cùng source labels có provenance cho generation/evaluation. Đánh giá lại riêng case sau thay đổi; chưa công bố fix đã cải thiện điểm.

### #3 — Câu 7

**Question:** Bao lâu phải đổi mật khẩu một lần?

**Expected (golden gốc):** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.

**Got (answer thật):**

> Phải đổi mật khẩu **mỗi 120 ngày**.  
> Nguồn: **mat_khau_v2.md**.

**Mean:** 0.735386; **Worst metric:** faithfulness.

faithfulness: 0.500000; answer_relevancy: 0.941542; context_precision: 1.000000; context_recall: 0.500000.

**Sources/context lấy được:** mat_khau_v2.md, vpn_truy_cap.md, bao_mat_su_co.md.

**Evidence:** mat_khau_v2.md, mục Chu kỳ thay đổi: mật khẩu phải được thay đổi mỗi 120 ngày. Parent hiện hành không chứa chu kỳ cũ 90 ngày. Answer 120 ngày đúng chính sách hiện hành.

**Root cause / kết luận sau đọc evidence:** Recall 0,5 có căn cứ ở việc golden còn yêu cầu chi tiết 90 ngày của bản cũ nhưng câu hỏi không chỉ định lịch sử, nên pipeline loại nguồn cũ. Faithfulness 0,5 vẫn cần trace judge để giải thích: số 120 được source hỗ trợ; tên file citation không hiện diện dưới dạng source label trong context raw.

**Error Tree:** Điểm thấp → 120 ngày có được context hỗ trợ? Có → Golden có thêm 90 ngày cũ? Có → Retrieval đang lọc superseded theo intent hiện hành → Tách completeness lịch sử và faithfulness của câu trả lời hiện hành.

**Suggested fix:** Nếu sản phẩm muốn so sánh phiên bản, dùng intent so sánh và đưa nguồn cũ có nhãn hết hiệu lực, không cho nó ghi đè quy tắc hiện hành. Với Faithfulness, lưu trace và đồng bộ source labels trước khi kết luận là hallucination.

### #4 — Câu 6

**Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?

**Expected (golden gốc):** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.

**Got (answer thật):**

> Mật khẩu phải có tối thiểu **12 ký tự**.  
> Nguồn: **mat_khau_v2.md**.

**Mean:** 0.736702; **Worst metric:** faithfulness.

faithfulness: 0.500000; answer_relevancy: 0.946807; context_precision: 1.000000; context_recall: 0.500000.

**Sources/context lấy được:** mat_khau_v2.md, phan_loai_du_lieu.md, bao_mat_su_co.md.

**Evidence:** mat_khau_v2.md ghi tối thiểu 12 ký tự. mat_khau_v1.md ghi 8 ký tự và trạng thái ĐÃ THAY THẾ. Answer 12 ký tự đúng bản hiện hành; context trả lời không chứa bản v1.

**Root cause / kết luận sau đọc evidence:** Recall 0,5 phù hợp với phần golden yêu cầu thêm thông tin 8 ký tự bản cũ; pipeline có chủ ý chỉ dùng source hiện hành ở query này. Faithfulness 0,5 chưa giải thích được chỉ bằng nội dung answer: số 12 đúng source. Source citation/statement extraction là giả thuyết đo lường cần kiểm chứng, không phải root cause đã được chứng minh.

**Error Tree:** Answer sai? Đọc policy: 12 đúng → Context có đủ mọi chi tiết golden? Thiếu 8 cũ → Kiểm tra mục tiêu current-policy so với lịch sử; riêng Faithfulness cần kiểm tra judge trace.

**Suggested fix:** Giữ đáp án current-policy và policy filter; thêm chế độ so sánh khi người dùng hỏi rõ. Kiểm tra judge với source metadata gốc và statement/verdict. Không sửa golden hoặc đẩy số 8 vào answer hiện hành chỉ để tăng điểm.

### #5 — Câu 12

**Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

**Expected (golden gốc):** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.

**Got (answer thật):**

> - **Phép năm:** **18 ngày**  
>   **Cách tính:** 15 ngày cơ bản + 3 ngày (9 năm thâm niên ÷ 3 năm × 1 ngày) = **18 ngày**.  
>   *(Nguồn: `nghi_phep_nam_v2024.md`)*
> 
> - **Lương:** Chính sách chỉ nêu đây là **18 ngày phép có lương**, không quy định mức tiền hoặc khoảng lương cụ thể.  
>   *(Nguồn: `nghi_phep_nam_v2024.md`)*

**Mean:** 0.776722; **Worst metric:** context_recall.

faithfulness: 0.714286; answer_relevancy: 0.892600; context_precision: 1.000000; context_recall: 0.500000.

**Sources/context lấy được:** nghi_phep_nam_v2024.md, nghi_phep_khong_luong.md, nghi_om.md.

**Evidence:** Diagnostic đọc lại index thật: sau filter có 16 candidates, bảng lương có trong RRF (score 0.026389) nhưng rerank đứng thứ 12, score 0.003825. Top-3 parent là nghỉ phép năm v2024, nghỉ không lương, nghỉ ốm; không có bảng lương. Context diagnostic khớp hoàn toàn lượt production. Child bảng lương có Senior và 20–35 triệu nhưng bị tách khỏi header/đơn vị bảng.

**Root cause / kết luận sau đọc evidence:** Thiếu coverage nhiều ý ở bước rerank/chọn parent: salary candidate đã được retrieval tìm thấy nhưng bị loại trước generation. LLM đúng phần 18 ngày và không bịa khoảng lương, nhưng không trả lời được cả hai ý. Đây là failure end-to-end thực, không phải thiếu corpus hoặc lỗi số học.

**Error Tree:** Answer thiếu lương → Context thiếu bảng lương → Corpus có 20–35 triệu? Có → RRF có child bảng lương? Có → Rerank/chọn top-3 đánh rớt salary → Cần ranking theo từng ý/giữ header bảng.

**Suggested fix:** Thử decomposition cho câu nhiều ý (phép/thâm niên và lương/cấp bậc), hợp nhất/dedup parent theo coverage; hoặc rerank parent/section có header bảng. Đo lại trên bộ câu mới và kiểm tra câu Junior để tránh hồi quy. Query decomposition và parent reranking chưa được triển khai trong bài này.

## Case study để demo: Senior 9 năm

1. Ngày phép: 15 + 9÷3 = 18, answer đúng.
2. Lương: corpus có Senior P3–P4 20–35 triệu VNĐ/tháng nhưng top-3 contexts không có bảng lương.
3. Diagnostic có salary ở candidates, nên lỗi nằm sau retrieval: rerank và lựa chọn coverage.
4. Cần cải thiện ranking nhiều ý/header bảng; không sửa answer bằng cách chép ground truth.

## Giới hạn và ưu tiên nếu có thêm một giờ

- 20 phút: lưu statement/verdict của judge và kiểm tra hai case password, source labels và dữ kiện từ query. Hiện report chỉ lưu metrics/evidence, chưa có trace NLI nên không khẳng định judge sai.
- 20 phút: prototype coverage theo từng ý và rerank section/parent với bảng còn header.
- 20 phút: chạy ablation trên bộ câu mở rộng, so quality/latency và kiểm tra regression.

60/60 contexts trong 20 câu đã được đối chiếu đúng parent gốc, không có enrichment làm evidence. Điều đó xác nhận provenance, không chứng minh mọi answer đều được judge chấm faithful. Mọi proposed fix ở trên chưa có kết quả benchmark mới. Điểm production trong báo cáo này giữ nguyên lượt đo đã chạy.
