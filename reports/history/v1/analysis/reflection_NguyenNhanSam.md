# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Nhân Sâm  
**MSSV:** 2A202602672  
**Khóa:** K4 - Track 3A  
**Ngày thực hiện:** 04/10/2026  
**Cách thực hiện:** Hoàn thiện có Codex hỗ trợ, kiểm tra theo từng checkpoint; số liệu lấy từ artifacts đã chạy. Cần tự đọc code và thực hành demo trước khi nộp.

## Mapping bài giảng vào code

| Concept | Module / hàm | Quan sát có bằng chứng |
|---|---|---|
| Semantic / hierarchical / structure-aware chunking | M1: `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware` | Với 26 tài liệu có text: basic 57, semantic 208 ở threshold 0.85, hierarchical 101 children/26 parents, structure 107. Kích thước parent 2048/child 256 tính bằng ký tự. Semantic nhiều chunks hơn không tự chứng minh accuracy tốt hơn. |
| BM25, Dense và RRF | M2: `segment_vietnamese`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion` | Lowercase và segmentation dùng chung cho query/corpus; dấu `_` được thay bằng khoảng trắng. Bge-m3 sinh vector 1024 chiều, lưu Qdrant COSINE. RRF cộng `1/(60+rank+1)` để tránh cộng các scores khác thang. Smoke chạy Docker Qdrant thật. |
| Cross-encoder reranking | M3: `CrossEncoderReranker.rerank`, `benchmark_reranker` | Bge-reranker-v2-m3 xét query/document cùng lúc. Smoke CPU 20 candidates → 3 kết quả, 5 warm runs trung bình 4377,42 ms. v2024 đứng đầu nhưng v2023 vẫn thứ hai: ranking tốt chưa đủ để kiểm soát hiệu lực. |
| RAGAS 4 metrics và Error Tree | M4: `evaluate_ragas`, `failure_analysis`, `save_report` | Baseline thật đủ 20 answers/80 scores: faithfulness 0,8667; relevancy 0,8005; precision 0,8667; recall 0,8000. Mỗi row giữ nguyên evidence. Lỗi judge được tách khỏi lỗi chất lượng; điểm thiếu là null, không dùng fallback làm điểm bài. Production và so sánh được ghi trong failure analysis. |
| Contextual embeddings / HyQA / metadata | M5: `_enrich_single_call`, `enrich_chunks` | Combined request tạo summary, questions, context, metadata; cache chỉ lưu response hợp lệ. Smoke API thật một fixture đạt đủ trường trong 16,82 giây. Metadata sinh thêm không được ghi đè source/parent/chunk/version. Nội dung làm giàu chỉ giúp tìm kiếm; generator đọc parent gốc. |

Tại query time, pipeline rerank raw child kèm tiêu đề lấy từ tài liệu, rồi deduplicate parent để tránh ba children cùng nguồn chiếm hết top-3. Không có query rewrite trong bài này. Ground truth chỉ dùng khi chấm; không đưa vào generator hoặc enrichment.

## Khó khăn và quá trình debug thực tế

### 1. Proxy trả nội dung website thay vì API

Response ở URL host gốc có `HTTP 200` nhưng `Content-Type: text/html`; chỉ nhìn status code sẽ kết luận sai. Kiểm tra URL `/v1` cho response JSON, chuẩn hóa host root trong config. Key được tiến trình app kế thừa khác key của project, nên config đọc đúng `.env` của repo với `override=True`, client truyền key/URL explicit. Không in giá trị key.

### 2. Model mặc định scaffold không có trên proxy

Exact error đã ghi ở `docs/SETUP_PROXY.md`: `model_not_found: No available channel for model gpt-4o-mini under group default (distributor)`. Kiểm tra danh sách models; người dùng chọn `gpt-5.6-luna` và local bge-m3. Cả baseline/production generation và judge dùng cùng lựa chọn. Không tự đổi toàn bộ dependency sang RAGAS mới để giải quyết lỗi model.

### 3. Downloader reranker bị đứng

Hugging Face downloader không có tiến triển với file model lớn. Tải HTTP có resume, đối chiếu SHA256 với linked ETag của Hugging Face, sau đó đặt vào model cache và load CrossEncoder thật. File khoảng 2,27 GB. Download và model load tách khỏi warm inference; smoke model load 5607,60 ms, không cộng vào con số warm 4377,42 ms.

### 4. Enrichment ghi đè nguồn chứng cứ

Contract test ban đầu thất bại với exact assertion `AssertionError: assert 'invented' == 'leave.md'`. Scaffold gộp generated metadata sau original metadata, nên source và parent ID có thể bị ghi đè. Sửa bằng whitelist các trường topic/entities/category/language và để metadata gốc thắng. Tests còn kiểm tra generated summary/questions được index nhưng không trở thành evidence trả lời.

### 5. Checker báo sẵn sàng khi tests/reflection chưa đạt

Test tái hiện lỗi đếm pytest collection: `assert (3, 3) == (3, 4)`, và thiếu analysis/reflection vẫn in sẵn sàng. Sửa checker để tính collection errors, yêu cầu 100% tests pass, analysis/reflection tồn tại, TODO bằng 0 và production report hợp lệ. Checker trả exit khác 0 khi có lỗi. Ngoài checker còn đối chiếu từng row, số metrics và source để không chỉ dựa vào sự tồn tại của JSON.

## Điều cần học thêm

Cần hiểu kỹ sự khác nhau giữa retrieval và generation failure. Faithfulness cao chỉ nói answer được context hỗ trợ; context có thể là chính sách cũ. Precision/recall của LLM judge có biến thiên, không phải xác nhận tuyệt đối answer đúng. Cần kiểm tra thủ công version, câu phủ định, thâm niên, đơn vị tiền và người phê duyệt trên các câu khó. Parent lớn giúp recall nhưng có thể kéo thêm noise; cần đo cùng latency thay vì chỉ tăng top-k.

## Action plan áp dụng vào project cá nhân

**Project:** Trợ lý hỏi đáp chính sách nội bộ ProductionRAG, mở rộng từ repository này. Đây là kế hoạch phát triển tiếp sau lab, chưa phải chức năng đã triển khai/deploy.

**Hiện trạng có thể kiểm tra trong repo:** text loader cho Markdown/PDF có text layer, 5 modules và CLI pipeline, bộ 20 câu gốc; scan PDFs bị bỏ qua. Enrichment/cache và model CPU khiến build/query chậm; backend Qdrant fallback cần xem nhãn khi demo. Chưa có giao diện web, OCR, authentication hoặc đo tải nhiều người dùng.

| Thời gian | Công việc cụ thể | Cách xác nhận |
|---|---|---|
| Tuần 1, ngày 1–2 | Audit corpus, source ID, nhóm chính sách, ngày hiệu lực, owner phê duyệt; xác định hai PDF scan cần OCR | Inventory có hash/source/version và danh sách tài liệu đọc được |
| Tuần 1, ngày 3–4 | Thử structure-aware cho bảng/lists, hierarchical cho văn bản dài; giữ parent-child và source | Test không cắt bảng/điều kiện; so retrieval trên bộ câu tách khỏi golden lab |
| Tuần 1, ngày 5 | Giữ hybrid RRF làm mốc, bổ sung bộ 40 câu do người dùng viết: lịch sử, phủ định, nhiều nguồn, ngoài corpus | Lưu contexts/answers; reviewed ground truth; không chỉnh golden theo output model |
| Tuần 2, ngày 1–2 | So sánh bật/tắt enrichment và reranker trên cùng dữ liệu; thử GPU hoặc model nhẹ nếu CPU quá chậm | Ablation có quality, request count, warm p50/p95 và chi phí thực |
| Tuần 2, ngày 3–4 | Thử OCR có human review, quản trị ngày hết hiệu lực và chính sách tương lai; đánh giá lại bottom cases | Test source grounding; OCR không được tự bổ sung số liệu thiếu |
| Tuần 2, ngày 5 | Thêm giao diện hỏi đáp có source citation; thử deploy local với nhóm nhỏ | Smoke UI/API, đo 10 query/người và ghi nhận câu không trả lời được |

Mục tiêu pilot đề xuất: ít nhất 3 metrics ≥0,70 trên bộ mới, 100% câu trả lời có nguồn và không chọn tài liệu hết hiệu lực cho câu hiện hành. Mục tiêu latency sẽ chốt sau ablation/GPU; chưa có bằng chứng để cam kết CPU dưới 2 giây. Không xem điểm golden lab là chất lượng trên dữ liệu doanh nghiệp mới.

## Kết quả và giới hạn sau lượt production

Production thật đủ 20 answers/80 scores: Faithfulness **0,755952**, Answer Relevancy **0,832710**, Context Precision **0,975000**, Context Recall **0,825000**. Ba metrics tăng, Faithfulness giảm **0,110714** so baseline. Cả bốn vượt 0,75 trong một lượt judge, nhưng không chứng minh hệ thống đạt mức này trên dữ liệu khác hoặc mọi lần chạy.

Case Senior thiếu lương được chẩn đoán trên index thật: salary child có trong RRF, nhưng rerank thứ 12 nên top-3 parent không có bảng lương. Child có Senior 20–35 triệu nhưng mất header/đơn vị bảng khi chia theo 256 ký tự. Điều này làm rõ tradeoff của hierarchical child nhỏ; việc đã implement structure-aware không đồng nghĩa pipeline default đang dùng strategy đó. Hướng thử tiếp là giữ header bảng/rerank section và coverage cho từng ý hỏi.

Case hoàn chi đào tạo trả đúng 25 triệu nhưng Faithfulness chỉ 0,166667. Chưa có trace statement/verdict để kết luận chính xác vì sao judge chấm thấp. Dữ kiện 25 triệu/8 tháng xuất phát từ câu hỏi; source citation trong prompt generation có tên file, còn context report là parent raw. Cần kiểm tra phép suy luận và biểu diễn provenance trong evaluator, không tự gọi tất cả điểm thấp là hallucination hoặc tự khẳng định judge sai.

Tài liệu tạm ứng chỉ có 2%/tháng, còn golden thêm cách prorate 5 ngày. Bài giữ nguyên corpus và golden, ghi rõ giới hạn nguồn. Đây là ví dụ cần human review ground truth/policy trước khi tối ưu model theo điểm.

Build cuối dùng cache 101/101 enrichment, index 55744,55 ms, load reranker 4281,82 ms. Online query mean 9973,07 ms, p95 13484,48 ms; RAGAS 523,37 giây, tách khỏi latency trả lời. Bằng chứng đầy đủ tại failure analysis, latency breakdown và JSON reports.
