# Lab 18 — Production RAG

Nguyễn Nhân Sâm · 2A202602672 · K4 Track 3A

Pipeline hỏi đáp chính sách tiếng Việt: chia tài liệu thành parent/child → enrichment để tìm kiếm → tách ý nếu câu hỏi nhiều ý → BM25 + bge-m3/Qdrant → RRF candidates → CrossEncoder và coverage từng ý → tối đa 3 parent gốc kèm nhãn nguồn → LLM trả lời → RAGAS 4 metrics.

## Chạy trên Windows

Từ root repository, dùng môi trường `.venv` đã setup:

```powershell
docker compose up -d
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe main.py
```

`main.py` dùng lại baseline thật nếu bộ dữ liệu, cấu hình và scores còn phù hợp; sau đó chạy production thật. Nếu thiếu baseline hợp lệ, chương trình chạy baseline trước. RAGAS có gọi API và có thể mất hơn 20 phút trên cấu hình hiện tại.

```powershell
# Chỉ đọc/kiểm chứng các reports đã lưu, không gọi API
& .\.venv\Scripts\python.exe main.py --reuse
& .\.venv\Scripts\python.exe src/pipeline.py --reuse

# Chạy production thật trực tiếp
& .\.venv\Scripts\python.exe src/pipeline.py

# Chủ động chạy lại baseline thật
& .\.venv\Scripts\python.exe main.py --refresh-baseline

# Kiểm tra bài
& .\.venv\Scripts\python.exe -m pytest tests/ -q
& .\.venv\Scripts\python.exe check_lab.py
```

`--reuse` chỉ chấp nhận report complete, đủ câu/metrics, tất cả answers sinh bởi LLM và đúng fingerprint; sẽ exit khác 0 nếu dữ liệu/code/cấu hình production đã đổi. Đây là kiểm chứng lượt đo đã lưu, không phải lượt benchmark mới. Chạy production thường sẽ tái dùng answers trong `reports/production_answers.json` nếu fingerprint khớp để tiếp tục sau gián đoạn, rồi chấm RAGAS mới. Đổi source/corpus/config làm fingerprint thay đổi và sinh lại answers.

Nếu lượt judge bị timeout và report `partial` còn ô metric thiếu, chạy `python scripts/retry_missing_scores.py` bằng `.venv`. Script dùng cùng question/answer/context/judge và chỉ gọi lại metric thiếu, giữ scores thành công; lưu report partial gốc trong history và ghi recovery metadata. Không dùng thao tác này cho report stale hoặc lỗi generation.

## Setup từ đầu

Cần Python 3.11+, Docker Desktop đang chạy và API proxy hỗ trợ model đã chọn.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d
```

Điền key và URL vào `.env` cục bộ theo [SETUP_PROXY.md](docs/SETUP_PROXY.md). Cấu hình của bài: generator/judge `gpt-5.6-luna`, embedding tìm kiếm và đánh giá local `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3`. Không đưa `.env` vào Git. Config ưu tiên `.env` project và thêm `/v1` nếu URL chỉ gồm host gốc.

Tải models trước khi chạy; riêng reranker khoảng 2,27 GB:

```powershell
& .\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); SentenceTransformer('BAAI/bge-m3')"
& .\.venv\Scripts\python.exe -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

Khi tất cả models đã cache, có thể đặt `$env:HF_HUB_OFFLINE='1'` để tránh kiểm tra Hugging Face qua mạng. API generation/evaluation vẫn hoạt động. Giữ bộ RAGAS 0.1.x/LangChain 0.2.x tương thích trong requirements; không nâng riêng từng thư viện.

## Cách hiểu và demo

- **M1**: `src/m1_chunking.py` có semantic, hierarchical và structure-aware. Basic 57 chunks; semantic 208; hierarchical 101 children/26 parents; structure 107. Kích thước là ký tự. Hierarchical là strategy production.
- **M2**: `src/m2_search.py` dùng cùng segmentation cho query/corpus. BM25 khớp từ khóa, Dense khớp ngữ nghĩa, RRF cộng đóng góp thứ hạng, không cộng hai thang scores khác nhau.
- **M3**: `src/m3_rerank.py` dùng CrossEncoder thật. Câu đơn chấm lại children; câu nhiều ý chấm parent gốc cho từng facet, dành một parent mỗi ý rồi fill theo câu gốc. Dedup parent hạn chế một tài liệu chiếm cả top-3.
- **M4**: `src/m4_eval.py` giữ question/answer/context/ground truth và 4 scores từng câu. Judge bị lỗi thì scores thiếu là `null`, không công bố fallback là điểm thật.
- **M5**: `src/m5_enrichment.py` dùng một request/chunk chưa cache cho summary, HyQA, context, metadata; tối đa 4 request song song. Chỉ cache kết quả hợp lệ. Fallback có nhãn riêng.

Chứng cứ trả lời luôn là parent gốc hoặc child gốc, không dùng summary/câu hỏi do LLM sinh làm sự thật. Generator và evaluator dùng chung `_evidence_contexts()` để nhận cùng raw evidence kèm filename gốc; checkpoint vẫn giữ raw riêng. Planner chỉ nhận query, cache theo prompt/model/endpoint/query; nếu lỗi sẽ tìm bằng câu gốc. Coverage là heuristic, chưa bảo đảm đủ nguồn mọi câu nhiều ý. Version metadata lấy từ tiêu đề/ngày hiệu lực trong corpus; phiên bản cũ bị loại ở câu hỏi hiện hành, vẫn được phép tìm cho câu hỏi về năm/phiên bản cũ. Quy tắc này phù hợp corpus lab và chưa thay cho một hệ thống quản trị hiệu lực tài liệu đầy đủ.

Demo nên chọn nghỉ phép hiện hành, quyền nghỉ phép khi thử việc và câu nhiều nguồn về Senior. Giải thích bằng `reports/production_answers.json`: nguồn nào được lấy, parent có đủ điều kiện áp dụng không, câu trả lời dùng số liệu nào. Ground truth chỉ vào evaluator, không vào retrieval, enrichment hay generator.

## Bằng chứng và bài nộp

| File | Ý nghĩa |
|---|---|
| `reports/naive_baseline_report.json` | Baseline thật, cùng 20 câu và model đánh giá |
| `reports/ragas_report.json` | Production thật, từng câu/4 metrics/provenance/timing |
| `reports/production_answers.json` | Answers/context/nguồn/timing, hỗ trợ resume |
| `reports/m1_chunking_report.json` | Số chunks và thời gian đo M1 |
| `reports/m2_dense_smoke.json` | Dense/Qdrant server smoke |
| `reports/m3_reranking_report.json` | CrossEncoder CPU, 5 warm runs |
| `reports/m4_ragas_smoke.json`, `reports/m5_enrichment_smoke.json` | Smoke fixture, không dùng làm benchmark 20 câu |
| `reports/latency_breakdown.md` | Build, online query và judge được đo riêng |
| `analysis/failure_analysis.md` | Bottom-5, evidence, Error Tree và fix |
| `analysis/reflections/reflection_NguyenNhanSam.md` | Mapping, lỗi thật và timeline áp dụng |
| `docs/IMPROVEMENT.md` | Điều tra judge/citation và coverage, giới hạn so sánh v1→v2 |
| `reports/history/v1/` | Reports, analysis và source snapshot trước cải thiện |

Corpus gốc có 25 Markdown và 3 PDF. Loader đọc được 26 tài liệu; 2 PDF scan không có text layer cần OCR và đang bị bỏ qua. Corpus/test set gốc được giữ nguyên. Dense có fallback Qdrant in-memory khi Docker không kết nối; backend được ghi rõ trong report, nên kiểm tra `qdrant_server` khi demo.

Unit tests M4/M5 cô lập API; contract tests kiểm tra metadata, cache, thiếu scores, parent expansion và report reuse. Scores unit tests không thay thế RAGAS thật. Flashrank là phần optional chưa triển khai; gọi vào sẽ báo `NotImplementedError`, pipeline dùng CrossEncoder.

Đề/rubric: [ASSIGNMENT.md](ASSIGNMENT.md), [RUBRIC.md](RUBRIC.md). Kiểm tra đủ report, analysis, reflection, tests và không còn TODO trước khi commit/push. Thao tác GitHub và nộp VLearn do học viên thực hiện; hoàn thiện local chưa đồng nghĩa đã nộp bài.

## Kết quả production đã kiểm chứng

Lượt production v2 ngày 04/10/2026: 20/20 answers do LLM sinh, 80/80 RAGAS scores hữu hạn, eval complete, Qdrant server. Ba metric bị proxy timeout trong lượt đầu đã được chấm bổ sung riêng bằng cùng inputs/cấu hình; các scores thành công được giữ nguyên và partial reports lưu trong `reports/history/v2_partial/`. So sánh cùng bộ câu/cấu hình:

| Metric | Baseline | Production v1 | Production v2 |
|---|---:|---:|---:|
| Faithfulness | 0.866667 | 0.755952 | 0.938214 |
| Answer Relevancy | 0.800456 | 0.832710 | 0.856665 |
| Context Precision | 0.866667 | 0.975000 | 0.975000 |
| Context Recall | 0.800000 | 0.825000 | 0.875000 |

Cả 4 metrics v2 ≥0,75 và Faithfulness v2 đạt ngưỡng bonus ≥0,85. So với v1, Faithfulness tăng 0,182262 và Context Recall tăng 0,050000; so với baseline, Faithfulness tăng 0,071548. V1→v2 đồng thời sửa evaluator source framing và retrieval/generation, nên không gán toàn bộ gain cho một thay đổi riêng. Bottom-5 v2 gồm tạm ứng, MFA, laptop 30 triệu, thâm niên phép và lương thử việc Junior; chi tiết evidence/error tree trong failure analysis.

Đối chiếu 60 raw contexts của 20 câu đều đúng parent gốc. Mean online query v2 21749,81 ms, p95 58464,91 ms; trong đó rerank parent/facet làm tăng chi phí câu nhiều ý. RAGAS gồm lượt đầu và recovery là 1496,10 giây; không thuộc latency trả lời. Enrichment build cuối sử dụng 101 cache entries, không công bố latency cache như chi phí 101 request mới.

```powershell
& .\.venv\Scripts\python.exe scripts/verify_reports.py
```

Các scripts `report_latency.py` và `finalize_analysis.py` tạo báo cáo từ artifacts đã kiểm chứng; `diagnose_multihop.py` đọc index để chẩn đoán, không gọi generation API. `finalize_analysis.py` yêu cầu `analysis/reviewed_failures.json` khớp fingerprint và answers của lượt đo, không dùng lại diagnosis cũ cho benchmark mới; script không sửa scores JSON.
