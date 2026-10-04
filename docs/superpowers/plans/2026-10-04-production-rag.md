# Production RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Kế hoạch này ưu tiên thực hiện trong chat hiện tại bằng executing-plans, báo cáo sau từng checkpoint. Chưa có yêu cầu triển khai code, commit, push hoặc nộp LMS trong lượt lập kế hoạch.

**Goal:** Hoàn thiện 5 module của Lab 18, chạy baseline và Production RAG trên cùng 20 câu hỏi, lưu bằng chứng RAGAS thật và viết phân tích/reflection có căn cứ.

**Architecture:** Khi lập chỉ mục: load tài liệu → chunking → enrichment → BM25 và dense index. Khi hỏi: hybrid retrieval → reranking → lấy lại parent context → LLM trả lời dựa trên tài liệu. RAGAS đánh giá answer/context so với ground truth, sau đó chọn bottom-5 để phân tích.

**Tech Stack:** Python 3.11+, sentence-transformers, underthesea, rank-bm25, Qdrant, OpenAI-compatible clients, RAGAS 0.1.x, pytest, Docker Compose.

**Spec:** `ASSIGNMENT.md`, `RUBRIC.md`; tham chiếu thêm `README.md`, `tests/test_m1.py` đến `tests/test_m5.py`.

## Global Constraints

- Bài cá nhân, implement toàn bộ 5 modules; tổng 100 điểm + 10 bonus.
- Python 3.11+; `.python-version` hiện ghi `3.11`.
- Giữ `ragas>=0.1.10,<0.2`, `langchain-community>=0.2,<0.3`, `langchain-openai>=0.1,<0.2` khi setup; không tự chuyển sang API RAGAS mới.
- Dense model `BAAI/bge-m3`, dimension `1024`; semantic model theo scaffold là `all-MiniLM-L6-v2`.
- Hierarchical parent `2048`, child `256`: scaffold tính bằng ký tự, không phải token.
- BM25, Dense và Hybrid top-k `20`; rerank top-k `3`.
- CrossEncoder dùng `BAAI/bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder`.
- Báo cáo bắt buộc: `reports/ragas_report.json`; phân tích: `analysis/failure_analysis.md`; reflection: `analysis/reflections/reflection_NguyenNhanSam.md`.
- Không sửa `test_set.json` hoặc corpus để làm tăng điểm. Không dùng ground truth làm đầu vào sinh answer/enrichment.
- Không công bố điểm fallback/mocked là RAGAS thật; trạng thái eval thất bại phải thể hiện rõ.
- API keys chỉ ở môi trường hoặc `.env`; không ghi vào code, reports, logs hoặc Git.
- Cấu hình được user chọn ở checkpoint M4: generation/judge `gpt-5.6-luna` qua proxy, evaluation embedding local `BAAI/bge-m3`. `config.py` ưu tiên `.env` project và thêm `/v1` nếu URL chỉ là host gốc. Các bước M5/tích hợp phải dùng cùng cấu hình này, không giữ hard-coded `gpt-4o-mini` từ scaffold.
- Deadline đề ghi 23h59 ngày diễn ra lab (GMT+7); chưa xác minh ngày học trên LMS.

## Hiện trạng được kiểm tra ngày 04/10/2026

- Repo clone tại `C:\Users\SAM\IdeaProjects\VinUniAIThucChien\lap 18\K4-Track3A-DAY18-NguyenNhanSam-2A202602672-ProductionRAG`.
- Branch `main`, base commit `3a56b90` (`fix(deps): add pytest and ruff to requirements.txt`).
- Corpus: 25 Markdown + 3 PDF. Loader có sẵn, bỏ qua PDF scan không có text; OCR không phải module bắt buộc trong rubric.
- Test set có 20 câu hỏi; 37 hàm test có sẵn; 18 marker `# TODO:` trong modules bắt buộc. Flashrank có marker optional riêng.
- `reports/` mới có `.gitkeep`; analysis/reflection hiện là mẫu.
- Shell hiện không tìm thấy `py`/`python` trên PATH. Python bundled được xác minh là 3.12.14; phần lớn dependencies lab chưa có.
- Có Docker CLI nhưng kiểm tra hiện không kết nối được Docker daemon; shell sandbox cũng không đọc được Docker config. Chưa thể kết luận Docker Desktop hoạt động trong tài khoản người dùng.
- Chưa cài môi trường lab, tải models, chạy unit tests hoặc gọi API.

## Những kiến thức cần nắm để hiểu và demo

| Nội dung | Cần hiểu | Ví dụ trong bài |
|---|---|---|
| RAG cơ bản | LLM chỉ trả lời từ context tìm được; retrieval sai dẫn đến answer sai | Tìm đúng chính sách nghỉ phép trước khi trả lời |
| Chunking | Chia nhỏ giúp tìm chính xác; phải giữ đủ ngữ cảnh, nguồn, section | Child tìm đúng ý, parent giữ điều kiện áp dụng |
| Semantic / hierarchical / structure | Theo ngữ nghĩa / quan hệ lớn-nhỏ / cấu trúc tiêu đề | Không cắt rời bảng lương hoặc câu phủ định |
| BM25 | Khớp từ khóa, hữu ích cho mã, số và tên riêng | "PVI", "MFA", "120 ngày" |
| Dense retrieval | Embedding biểu diễn ý nghĩa; vector similarity tìm câu diễn đạt khác | "nghỉ đám cưới" và "nghỉ khi kết hôn" |
| RRF | Hợp nhất thứ hạng thay vì cộng điểm BM25 và cosine có thang khác nhau | `sum(1 / (60 + rank + 1))` với rank từ 0 |
| Reranking | Chấm lại từng cặp query-document để chọn top-3 từ candidates | Tài liệu nghỉ phép phải đứng trên VPN |
| Enrichment | Thêm mô tả/summary/câu hỏi giả định/metadata trước index | Thêm tên chính sách và phiên bản để chunk rõ nghĩa |
| Phiên bản và phủ định | Similarity cao chưa chứng minh văn bản còn hiệu lực hoặc điều kiện đúng | 15 ngày v2024 thay 12 ngày v2023; thử việc KHÔNG được phép năm |
| Multi-hop và số học | Có câu cần nhiều nguồn, sau đó mới tính | 9 năm thâm niên: 15 + 9/3 = 18 ngày; kết hợp bảng lương |
| Evaluation | Lưu câu hỏi, answer, retrieved contexts, ground truth và từng metric | Điểm trung bình không đủ để debug câu trả lời sai |

### Bốn metric RAGAS

- **Faithfulness:** Những khẳng định trong answer có được context hỗ trợ không?
- **Answer Relevancy:** Answer có trả lời đúng điều câu hỏi muốn biết không?
- **Context Precision:** Context được lấy có liên quan và được xếp hạng tốt không?
- **Context Recall:** Context có đủ thông tin cần cho ground truth không?

Faithfulness cao vẫn có thể trả lời chính sách cũ nếu context toàn văn bản cũ. Cần kiểm tra thêm nguồn, hiệu lực và đáp án từng câu.

## File structure và trách nhiệm

| File | Thay đổi dự kiến |
|---|---|
| `.gitignore` | Bỏ qua `.venv/`, cache models/enrichment, log tạm; giữ reports cần nộp |
| `config.py` | Giữ defaults; cho phép cấu hình client/model khi endpoint thực tế cần |
| `src/m1_chunking.py` | 3 chunkers, giữ source/section, parent IDs không trùng giữa tài liệu |
| `src/m2_search.py` | Tokenization, BM25, dense Qdrant, RRF |
| `src/m3_rerank.py` | Lazy load CrossEncoder, predict, sort, top-k |
| `src/m4_eval.py` | RAGAS thật, bottom-N, lưu dữ liệu từng câu, trạng thái evaluation |
| `src/m5_enrichment.py` | Combined mode, hàm riêng và fallback, bảo toàn provenance |
| `src/pipeline.py` | Nối module, parent lookup, tách text tìm kiếm với text chứng cứ, đo latency |
| `naive_baseline.py`, `main.py` | Chỉ sửa khi cần cùng cấu hình generator/evaluator, lưu provenance, so sánh công bằng |
| `tests/test_contracts.py` | Thêm tests cho các hợp đồng mà tests gốc chưa kiểm tra |
| `tests/test_pipeline_contracts.py` | Kiểm tra parent expansion, chống dùng enrichment làm chứng cứ |
| `analysis/failure_analysis.md` | Điểm thật, bottom-5, Error Tree, root cause và fix |
| `analysis/reflections/reflection_NguyenNhanSam.md` | Mapping 5 module, lỗi thực tế, project application |
| `reports/latency_breakdown.md` | Latency từng bước; tách cold start và warm runs |

## Task 1: Setup môi trường và kiểm tra dịch vụ

**Checkpoint 04/10/2026:** dùng được `.venv` Python 3.11.7; `pip check` không có lỗi; Docker 29.6.1 và Qdrant server hoạt động. Đã load thực tế all-MiniLM-L6-v2 và bge-m3. API chưa đạt: cấu hình `.env` mới trả HTTP 200 nhưng `text/html`, không phải JSON OpenAI API; chưa sửa `.env`. Key của tiến trình khác key project nên các client API sau này phải chọn nguồn cấu hình nhất quán. Reranker sẽ kiểm tra ở M3. Không tạo venv riêng.

**Files:** Modify `.gitignore`; tạo `.venv/` và `.env` cục bộ.
**Interfaces:** Python interpreter chạy được, Qdrant endpoint `localhost:6333`, client generation/evaluation được xác minh khả năng tương thích.

- [ ] Xác minh Python 3.11 nếu có ngoài PATH; nếu không có, thử Python bundled 3.12 trong venv và kiểm tra tương thích dependency. Ưu tiên Python 3.11 cho bộ dependencies cũ.
- [ ] Dùng interpreter đã xác minh để tạo venv. Ví dụ fallback đã có trên máy:

```powershell
$labPython = 'C:\Users\SAM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $labPython -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

- [ ] Bổ sung `.venv/` và cache phát sinh vào `.gitignore` trước khi tạo dữ liệu lớn.
- [ ] Tạo `.env` nếu chưa tồn tại, không ghi đè file đã cấu hình:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
docker version
docker compose up -d
Invoke-RestMethod http://localhost:6333/collections
```

- [ ] Kiểm tra keys chỉ dưới dạng set/missing. Xác minh endpoint hỗ trợ chat và embedding dùng trong RAGAS; tên key có sẵn chưa chứng minh model/endpoint dùng được.
- [ ] Pre-download các model theo README; ghi thời gian download riêng khỏi benchmark inference.
- [ ] Chạy tests gốc để ghi baseline lỗi. Tests M4/M5 có thể gọi API sau khi implement: khi đó tests thông thường cần mock, đánh giá thật chạy riêng.

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Checkpoint:** interpreter, dependencies và backend được ghi rõ. In-memory Qdrant sẵn trong scaffold chỉ dùng debug khi cần; chạy Docker thật để chứng minh yêu cầu của đề.

## Task 2: M1 — implement và kiểm tra chunking

**Checkpoint hoàn tất 04/10/2026:** 13 tests gốc M1 + 10 test cases bổ sung pass (23/23). Báo cáo `reports/m1_chunking_report.json`: 26 documents có text, basic 57, semantic 208, hierarchical 101 children/26 parents, structure 107. Hai PDF scan bị bỏ qua đúng loader gốc. Semantic dùng encoder thật; giữ source, parent identity và fenced code/table.

**Files:** Modify `src/m1_chunking.py`; test `tests/test_m1.py`; create `tests/test_contracts.py`.
**Consumes:** `load_documents() -> list[dict]`, từng document có `text`, `metadata.source`.
**Produces:** `chunk_semantic(text, threshold, metadata) -> list[Chunk]`, `chunk_hierarchical(text, parent_size, child_size, metadata) -> tuple[list[Chunk], list[Chunk]]`, `chunk_structure_aware(text, metadata) -> list[Chunk]`.

- [ ] Chạy tests M1 để thấy chức năng còn thiếu.
- [ ] Implement semantic: tách câu, encode, so cosine của câu kề nhau, ngắt khi thấp hơn threshold; cache model thay vì load lại mỗi document.
- [ ] Implement hierarchical: gom đoạn thành parent, chia child; tách thêm paragraph dài; parent ID phải duy nhất xuyên tài liệu, ví dụ prefix SHA256 của source/text + index.
- [ ] Implement structure-aware: giữ header path, bảng, danh sách và fenced code; không nhận `#` trong code block làm heading.
- [ ] Thêm test cho liên kết và giới hạn kích thước (test sau viết trước implementation tương ứng):

```python
from src.m1_chunking import chunk_hierarchical

def test_hierarchy_limits_and_document_identity():
    text = "Đây là nội dung dài. " * 100
    pa, ca = chunk_hierarchical(text, 200, 80, {"source": "a.md"})
    pb, cb = chunk_hierarchical(text, 200, 80, {"source": "b.md"})
    assert pa and ca and pb and cb
    assert all(len(p.text) <= 200 for p in pa + pb)
    assert all(len(c.text) <= 80 for c in ca + cb)
    assert {p.metadata["parent_id"] for p in pa}.isdisjoint(
        {p.metadata["parent_id"] for p in pb}
    )
```

- [ ] Chạy `python -m pytest tests/test_m1.py tests/test_contracts.py -v`; ghi số chunk của mỗi strategy để dùng trong reflection.

**Checkpoint:** 3 strategies đúng logic; có source, section, parent-child hợp lệ. Không thay semantic bằng split cố định chỉ để tests pass.

## Task 3: M2 — BM25 + Dense + RRF

**Checkpoint hoàn tất 04/10/2026:** toàn bộ M1+M2+contracts 31/31 pass. Smoke `BAAI/bge-m3` dimension 1024 chạy với Docker Qdrant server, tìm đúng nguồn nghỉ phép; kết quả `reports/m2_dense_smoke.json`. Collection smoke riêng đã được dọn. RRF giữ các nguồn khác nhau có text giống nhau. Chưa sang M3; theo chỉ dẫn mới của user phải báo cáo và chờ trước mỗi bước tiếp theo. Branch `feat/production-rag`, chưa commit/push.

**Files:** Modify `src/m2_search.py`; test `tests/test_m2.py`, `tests/test_contracts.py`.
**Consumes:** `chunks: list[dict]`, mỗi phần tử có `text`, `metadata`.
**Produces:** `BM25Search.index/search`, `DenseSearch.index/search`, `reciprocal_rank_fusion(...) -> list[SearchResult]`; `HybridSearch` giữ nguyên giao diện scaffold.

- [ ] Chạy tests M2 trước khi implement.
- [ ] Dùng cùng normalization cho query và corpus: lowercase, word_tokenize, replace underscores như đề yêu cầu. Không mô tả bước này là giữ nguyên từ ghép sau khi đã replace.
- [ ] BM25 index và search: rỗng trả `[]`, top-k hữu hạn, score descending.
- [ ] Dense: encode bge-m3, kiểm tra dimension 1024, Qdrant COSINE, upsert payload source/text/IDs, truy vấn bằng `query_points()`.
- [ ] RRF: hợp nhất identity của chunk, không cộng thẳng BM25 score với cosine score. Nếu chỉ dùng text làm key theo scaffold, ghi nhận nguy cơ gộp hai nguồn giống text; ưu tiên chunk identity khi có metadata.
- [ ] Thêm test RRF có điểm số chính xác:

```python
import pytest
from src.m2_search import SearchResult, reciprocal_rank_fusion

def test_rrf_adds_rank_contributions():
    a = [SearchResult("x", 99.0, {}, "bm25"),
         SearchResult("y", 1.0, {}, "bm25")]
    b = [SearchResult("y", 0.8, {}, "dense")]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=2)
    assert fused[0].text == "y"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].method == "hybrid"
```

- [ ] Chạy `python -m pytest tests/test_m2.py tests/test_contracts.py -v`.
- [ ] Chạy smoke Dense thật với corpus nhỏ, xác minh payload/source và collection baseline/production tách biệt. Tests M2 gốc không kiểm tra Dense.

**Checkpoint:** BM25, Dense và Hybrid đều có kết quả thực; ghi backend đã dùng.

## Task 4: M3 — reranking và latency

**Checkpoint hoàn tất 04/10/2026:** implement lazy-load `sentence_transformers.CrossEncoder`, rerank giảm dần và top-k, giữ original score/metadata, kiểm tra một score hữu hạn mỗi document; benchmark từ chối `n_runs <= 0`. Trước implementation, tests M3/contract có 9 failed, 5 passed; sau implementation toàn bộ M1–M3 và contracts **45/45 pass**, gồm 5 tests M3 gốc chạy model thật. Model `BAAI/bge-reranker-v2-m3` 2,27 GB đã tải từ Hugging Face và xác minh SHA256; downloader mặc định không tiến triển, tải HTTP trực tiếp hoàn tất. Smoke trên CPU lấy 20 candidates → 3 kết quả: v2024 đứng đầu, v2023 đứng thứ hai (rerank chưa loại chính sách cũ). Báo cáo `reports/m3_reranking_report.json`: model load 5607,60 ms, first inference 4361,08 ms; 5 warm runs trung bình 4377,42 ms (min 4338,40 / max 4436,63). Đây là retrieval smoke, chưa phải accuracy/RAGAS. Flashrank optional chưa triển khai. Chưa commit/push; dừng trước M4 để user xem checkpoint.

**Files:** Modify `src/m3_rerank.py`; test `tests/test_m3.py`.
**Consumes:** query và docs có `text`, `score`, `metadata`.
**Produces:** `CrossEncoderReranker.rerank(query, documents, top_k) -> list[RerankResult]`.

- [ ] Chạy tests M3 trước khi implement.
- [ ] Lazy-load CrossEncoder một lần mỗi instance; tạo `(query, text)` pairs và gọi `predict()` theo batch.
- [ ] Sort rerank score giảm dần, giữ original score và metadata; hỗ trợ input rỗng và top-k hợp lệ.
- [ ] Chạy `python -m pytest tests/test_m3.py -v`; kiểm tra nghỉ phép đứng trước VPN.
- [ ] Load/warm-up trước khi đo benchmark; lưu cold-load riêng khỏi thời gian query. Không dùng fallback chỉ giữ thứ tự cũ rồi gọi đó là CrossEncoder.
- [ ] Flashrank optional: chỉ triển khai khi phần bắt buộc đã hoàn tất; nếu bỏ, ghi rõ unsupported và xử lý marker optional minh bạch.

**Checkpoint:** top-20 → top-3 có rerank scores thật và benchmark có đơn vị ms.

## Task 5: M4 — RAGAS và baseline thật

**Checkpoint hoàn tất 04/10/2026:** implement RAGAS 0.1.22 với clients explicit, giữ answer/context/ground truth từng câu, aggregate và metric counts, trạng thái complete/partial/failed/not_run. Điểm thiếu lưu `null` ở cả row và aggregate khi không có phép đo; fallback numeric chỉ giữ cho giao diện scaffold. Bottom-N có diagnosis hypothesis, suggested fix và Error Tree; không gán lỗi kỹ thuật judge thành lỗi chất lượng answer. Unit tests mock riêng external judge, không sửa assertions tests gốc.

API đã xác minh: `.env` project được ưu tiên, host URL được chuẩn hóa `/v1`; `gpt-4o-mini` không có trên proxy, user chọn `gpt-5.6-luna` và local bge-m3. Chat smoke thành công; một câu RAGAS smoke đạt đủ 4 metrics, ghi riêng tại `reports/m4_ragas_smoke.json` với nhãn hand-written fixture. Hướng dẫn tại `docs/SETUP_PROXY.md`.

Baseline thật tại `reports/naive_baseline_report.json`: **20/20 answers do LLM sinh, 80/80 scores hữu hạn, eval_status=complete, không generation/evaluation errors**, Docker Qdrant backend. Faithfulness **0.8666666667**, Answer Relevancy **0.8004561258**, Context Precision **0.8666666666**, Context Recall **0.8000000000**. Judge/embedding config được lưu trong report; không coi đây là điểm production. Runtime stdout: tổng 1445,3 giây; phần RAGAS khoảng 19 phút 11 giây với 2 workers. Top-5 thấp nhất gồm phân loại lương, chu kỳ đổi mật khẩu, câu Senior nhiều nguồn, MFA và phí tạm ứng.

Kiểm tra cuối: **61/61 tests M1–M4 và contracts pass**; kiểm tra JSON đủ 20 rows, average khớp scores từng câu, counts mỗi metric 20, đủ 5 failures và key project không có trong code/docs/reports đã kiểm tra. Lỗi Unicode của script audit stdin được xử lý bằng `PYTHONIOENCODING=utf-8`; source module đã dùng stdout UTF-8. Chưa triển khai M5, chưa tạo report production, chưa commit/push. Dừng để user xem checkpoint.

**Files:** Modify `src/m4_eval.py`; test `tests/test_m4.py`, `tests/test_contracts.py`; review `naive_baseline.py`.
**Consumes:** `questions`, `answers`, `contexts`, `ground_truths` cùng độ dài.
**Produces:** `evaluate_ragas(...) -> dict` với 4 metrics và `per_question: list[EvalResult]`; `failure_analysis(eval_results, bottom_n=5) -> list[dict]`; `save_report(...)` tạo JSON có `aggregate`, `num_questions`, `failures`, thêm `per_question` và trạng thái eval.

- [ ] Chạy tests M4 trước khi implement; mock external judge trong unit tests, không sửa nội dung assertion gốc.
- [ ] Dataset dùng schema RAGAS 0.1.x: question, answer, contexts, ground_truth; configure judge và embedding clients phù hợp endpoint.
- [ ] Lưu từng `EvalResult`; kiểm tra NaN/lỗi từng metric, không tự đổi NaN thành điểm cao. Nếu API lỗi, fallback có trạng thái/error rõ và không coi là đánh giá đạt.
- [ ] Sửa serializer lưu per-question bằng `dataclasses.asdict`; aggregate chỉ chứa 4 metric số, metadata chạy đặt riêng ở cấp report.
- [ ] Thêm test report có bằng chứng từng câu:

```python
import json
from src.m4_eval import EvalResult, save_report

def test_report_keeps_answer_context_evidence(tmp_path):
    row = EvalResult("q", "a", ["c"], "gt", .8, .7, .6, .9)
    metrics = {"faithfulness": .8, "answer_relevancy": .7,
               "context_precision": .6, "context_recall": .9}
    path = tmp_path / "report.json"
    save_report({**metrics, "per_question": [row]}, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["per_question"][0]["answer"] == "a"
    assert report["per_question"][0]["contexts"] == ["c"]
```

- [ ] Bottom-N xếp theo trung bình 4 metric tăng dần; worst metric chỉ là gợi ý diagnosis, xác nhận root cause bằng answer/context/source.
- [ ] Chạy `python -m pytest tests/test_m4.py tests/test_contracts.py -v`.
- [ ] Smoke RAGAS thật trên 1-2 câu để kiểm tra API/schema trước khi chạy 20 câu.
- [ ] Chạy `python naive_baseline.py` sau khi M2 và M4 hoạt động; lưu baseline thật. Chạy baseline trước implementation chỉ tạo placeholder, không dùng làm kết quả so sánh.

**Checkpoint:** có đường eval thật và evidence từng câu; baseline không còn chỉ là zeros mặc định.

## Task 6: M5 — enrichment combined và fallback

**Files:** Modify `src/m5_enrichment.py`; test `tests/test_m5.py`, `tests/test_contracts.py`.
**Consumes:** raw chunk text + metadata.source, không có ground truth.
**Produces:** `_enrich_single_call(text, source) -> dict` với `summary`, `questions`, `context`, `metadata`; `enrich_chunks(...) -> list[EnrichedChunk]`.

- [ ] Chạy tests M5 trước khi implement. Tests gốc chủ yếu gọi mode riêng, chưa chứng minh combined mode.
- [ ] Combined: một request/chunk, yêu cầu JSON hợp lệ, kiểm tra types từng trường; output không có facts ngoài raw text/nguồn.
- [ ] Hoàn thiện các hàm riêng cho giao diện tests; fallback deterministic không gọi LLM khi thiếu key hoặc request thất bại.
- [ ] Giữ nguyên `original_text`; auto metadata không được ghi đè source, chunk ID, parent ID hoặc version đã lấy từ tài liệu.
- [ ] Tách `original_text` và `enriched_text` trong payload để phần mô tả do LLM tạo chỉ phục vụ tìm kiếm, không tự trở thành chứng cứ cho answer.
- [ ] Mock client để đếm 1 call/chunk ở combined; cache theo SHA256(text + source + model + prompt version) để rerun không gọi lại toàn bộ.
- [ ] Chạy `python -m pytest tests/test_m5.py tests/test_contracts.py -v`; smoke combined thật trên 1 chunk và kiểm tra JSON.

**Checkpoint:** combined hoạt động thật, fallback có nhãn, source và nguyên văn vẫn còn.

## Task 7: Tích hợp và chạy đánh giá đầy đủ

**Files:** Modify `src/pipeline.py`, review `main.py`, `naive_baseline.py`, `config.py`; create `tests/test_pipeline_contracts.py`; output `reports/`.
**Consumes:** giao diện M1–M5 hiện có.
**Produces:** giữ `build_pipeline() -> tuple[HybridSearch, CrossEncoderReranker]` và `run_query(query, search, reranker) -> tuple[str, list[str]]` để `main.py` tiếp tục chạy.

- [ ] Gắn `search.parent_lookup: dict[str, str]` khi build pipeline, đồng thời lưu original child text vào payload. Parent IDs phải duy nhất xuyên corpus.
- [ ] Sau rerank, resolve parent_id và bỏ parent trùng lặp; giới hạn tổng context theo budget. Nếu không có parent, dùng original child text.
- [ ] Thêm test với fake search/reranker, không cần model hoặc API:

```python
from types import SimpleNamespace
from src import pipeline

def test_run_query_expands_original_parent(monkeypatch):
    import config
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    child = SimpleNamespace(text="Mô tả đã enrich", score=1.0,
        metadata={"parent_id": "policy:p0", "original_text": "Nghỉ 15 ngày."})
    parent = "Nhân viên chính thức được nghỉ 15 ngày. Thử việc không được hưởng."
    search = SimpleNamespace(search=lambda query: [child],
                             parent_lookup={"policy:p0": parent})
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: [child])
    _, contexts = pipeline.run_query("Thử việc có phép năm không?", search, reranker)
    assert contexts == [parent]
```

- [ ] Prompt answer giữ phủ định/đơn vị, ưu tiên văn bản hiệu lực khi có mâu thuẫn, chỉ tính từ số có trong nguồn; thiếu chứng cứ phải nói rõ.
- [ ] Parse metadata phiên bản/hiệu lực từ corpus nếu đánh giá phát hiện nhầm bản. Không xóa tài liệu cũ hoặc hard-code đáp án từng câu để đạt điểm.
- [ ] Chạy `python -m pytest tests/ -v` và smoke vài query đại diện: version, negation, multi-hop, numeric.
- [ ] Chạy `python main.py`: cùng 20 câu, cùng generator/judge/config cho baseline và production; ghi model names, backend, modes, timestamp, source/context mỗi câu.
- [ ] Chạy thêm `python src/pipeline.py` theo tiêu chí rubric; dùng cache enrichment và log riêng để tránh lặp chi phí không cần thiết.
- [ ] Kiểm tra report có 20 rows, 4 metrics hữu hạn, đúng trạng thái live; tạo bảng baseline/production/delta. Mục tiêu ≥3 metrics đạt .70, không đảm bảo trước khi đo.
- [ ] Ghi latency chunking/enrichment/indexing/search/rerank/generation/eval. Tách thời gian build index khỏi online query.

**Checkpoint:** 37 tests gốc cùng tests bổ sung pass; cả hai entry points chạy; report có nguồn và evidence thực. Pass tests riêng không thay thế checkpoint này.

## Task 8: Failure analysis, reflection và kiểm tra bài nộp

**Files:** Modify `analysis/failure_analysis.md`; create `analysis/reflections/reflection_NguyenNhanSam.md`, `reports/latency_breakdown.md`.
**Consumes:** reports thật, per-question evidence, logs lỗi và latency đã đo.
**Produces:** bộ deliverables theo rubric, vẫn chưa commit/push/nộp LMS.

- [ ] Chọn đúng bottom-5 theo scores; mỗi case ghi question, expected, got, worst metric, context/source, Error Tree, root cause, fix.
- [ ] Error Tree: answer sai → context đủ và đúng không? → nếu thiếu, kiểm tra corpus/chunk/index/retrieval → nếu context đúng, kiểm tra version/negation/reasoning/prompt. Chỉ nhắc query rewrite nếu đã thực sự implement.
- [ ] Reflection mapping đủ 5 modules, dùng số chunk/latency/metrics thật; ghi exact error đã gặp và quá trình debug. Không viết lỗi giả định như trải nghiệm thực.
- [ ] Project application: xác nhận project muốn áp dụng; chỉ mô tả hiện trạng có bằng chứng. Gợi ý tuần 1 audit corpus/chunking/hybrid; tuần 2 rerank/enrichment/evaluation và so latency/chất lượng.
- [ ] Chạy kiểm tra cuối:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/ -v
& .\.venv\Scripts\python.exe check_lab.py
rg -n '# TODO' src
git status --short
git diff --stat
```

- [ ] Kiểm tra JSON ngoài `check_lab.py`: đủ 20 câu, không zeros fallback, không NaN, đúng reports; reflection không còn mẫu; bottom-5 có Error Tree.
- [ ] Rà `.env`, `.venv`, cache/logs và bí mật trước khi đưa ra lệnh Git. User thực hiện/ủy quyền commit, push và nộp LMS ở bước riêng.

**Checkpoint:** báo cáo files changed, tests, scores thật, lỗi còn lại và Git state; không dùng câu "sẵn sàng" của `check_lab.py` làm bằng chứng duy nhất.

## Thứ tự triển khai và thời gian

Thứ tự code: Setup → M1 → M2 → M3 → M4 + baseline thật → M5 → tích hợp → analysis/reflection.
Thứ tự runtime: M1 → M5 → M2 → M3 → LLM → M4.

| Checkpoint | Ước tính khi môi trường đã sẵn sàng | Bằng chứng |
|---|---:|---|
| Setup | 10–20 phút, chưa tính download/install/khắc phục Docker | Interpreter, pip check, Qdrant/API smoke |
| M1 | 20–30 phút | 3 strategies, metadata, parent integrity |
| M2 | 20–30 phút | BM25/Dense/RRF, Dense smoke thật |
| M3 | 15–20 phút | CrossEncoder relevance + latency |
| M4 + baseline | 20–30 phút, cộng thời gian API | Eval từng câu, baseline thật |
| M5 | 20–30 phút, cộng thời gian API | Combined 1 call/chunk + fallback |
| Tích hợp/eval | 20–40 phút, cộng thời gian API | Pipeline + 20 câu + delta |
| Analysis/reflection | 30 phút | Bottom-5, mapping, project timeline |

Đề phân bổ 2h implement + 30 phút reflection; máy chưa setup có thể cần thêm thời gian, đặc biệt tải bge-m3/reranker và CPU inference.

## Tự rà kế hoạch

- [x] M1–M5 đủ 60 điểm implementation.
- [x] End-to-end, RAGAS và failure analysis đủ phần 25 điểm.
- [x] Mapping, debugging và project plan đủ phần 15 điểm reflection.
- [x] Combined mode và latency là bonus khả thi; bonus điểm RAGAS chỉ kết luận sau khi đo.
- [x] Phát hiện và đưa vào kế hoạch: parent context chưa nối, per-question report bị bỏ, tests Dense/combined còn thiếu, baseline placeholder và checker thiếu gate.
- [x] Phân biệt việc đã kiểm tra, việc dự kiến, scores thật/fallback và thao tác Git/LMS.

## Checkpoint M5 và tích hợp — 04/10/2026

- M5 hoàn tất: combined JSON request, cache thành công theo model/endpoint/prompt/source/text, fallback có nhãn, 4 workers, giữ thứ tự input và metadata gốc. Tests M5 gốc + contracts 15/15 pass. Live smoke fixture `reports/m5_enrichment_smoke.json` có đầy đủ summary/questions/context/metadata (16,82 giây); không phải benchmark 20 câu.
- Đã chạy enrichment corpus 101 children; lần build đầu dừng sau enrichment để sửa review, giữ cache thành công. Latency toàn bộ build bị dừng không được coi là số đo chính thức.
- Pipeline: parent lookup gốc, rerank raw children và tiêu đề document, dedup tối đa 3 parent, generation explicit proxy/model, lưu sources/generation status/query timings và answer checkpoints.
- Review độc lập phát hiện hai P2: explicit version v1.0 chưa được nhận diện lịch sử; baseline reuse thiếu implementation/config fingerprint. Đã sửa với red/green contract tests; reviewer recheck không còn vấn đề quan trọng trong hai fixes.
- Baseline thật được giữ lại. Dataset hash được bổ sung sau khi kiểm tra corpus/test không đổi so Git HEAD. Baseline fingerprint được attestation hồi tố với ghi chú rõ: generation/retrieval/evaluator không đổi, các chỉnh sửa baseline sau lượt chạy chỉ thêm provenance. Không gọi đó là lượt baseline mới.
- Main dùng lại baseline hợp lệ và chấm production thật. Source production được giữ nguyên trong lúc benchmark để fingerprint phản ánh code đã thực thi.
- Checker không còn báo sẵn sàng khi thiếu analysis/reflection, TODO hoặc tests chưa đạt; report phải complete và hợp lệ. README có Windows commands và giải thích reuse/resume.
- Trước review fixes: full suite 84/84 pass. Sau fixes: targeted main/pipeline/baseline contracts 8/8 pass; full verification cuối sẽ thực hiện sau benchmark.
- Chưa commit/push/nộp VLearn.

## Hoàn tất local — 04/10/2026

- [x] Setup đã dùng môi trường Python 3.11.7 của user và Qdrant server.
- [x] M1: semantic/hierarchical/structure-aware; measurements và contracts.
- [x] M2: BM25/Dense/RRF; Qdrant server smoke.
- [x] M3: CrossEncoder thật và benchmark; optional Flashrank ghi rõ chưa hỗ trợ.
- [x] M4: evaluator/status/evidence, baseline thật đủ 20/80.
- [x] M5: combined/cache/fallback/provenance, smoke thật và corpus enrichment.
- [x] Tích hợp: parent gốc, filter phiên bản, generation explicit, answers checkpoint, latency, production RAGAS thật.
- [x] Reports/reflection/failure analysis và kiểm tra cuối.

Production v1 complete: Faithfulness 0.7559523809523809, Answer Relevancy 0.8327103501049559, Context Precision 0.974999999905, Context Recall 0.825; 20 LLM answers, 80 finite scores. Đây là artifact lịch sử trước cải thiện.

`python main.py` đã chạy production thật và exit 0, dùng lại baseline được xác minh. Hai entry points chạy với `--reuse` đều exit 0 và nói rõ đọc lượt đo đã lưu, không gọi API mới. `scripts/verify_reports.py` xác minh 60/60 contexts từ parent gốc, 20/80 cho mỗi pipeline, 5 failures, Qdrant server và corpus/test không sửa. Diagnostic câu Senior xác nhận salary candidate có trong retrieval, rerank thứ 12 nên thiếu nguồn trong top-3.

Kiểm tra cuối trước cải thiện `python check_lab.py`: exit 0, **85/85 tests pass**, zero source TODOs, analysis/reflection/report đủ. Secret scan 80 text files: key project không xuất hiện ngoài `.env`. Branch `feat/production-rag`; code/docs/reports/tests còn uncommitted, chưa push hoặc nộp LMS.

### Kết quả production v2 sau cải thiện

Report complete: Faithfulness **0.938214**, Answer Relevancy **0.856665**, Context Precision **0.975000**, Context Recall **0.875000**; 20/20 LLM answers, 80/80 finite scores, zero errors after recovery. So với v1, delta lần lượt **+0.182262**, **+0.023955**, **0**, **+0.050000**. So với baseline, Faithfulness **+0.071548**. Cả 4 metrics đạt ≥0,75 và Faithfulness đạt bonus ≥0,85. V1→v2 gồm evaluator source framing và retrieval/generation changes; không gán toàn bộ delta cho retrieval.

Recovery xử lý ba timeout metric bằng cùng question/answer/context/model/embedding/configuration, giữ 77 scores đã thành công và lưu các partial reports cùng SHA256 trong `reports/history/v2_partial/`. Case Senior v2 lấy được `bang_luong_2024.md`, trả lời 18 ngày phép và lương gross 20–35 triệu/tháng. Kiểm tra checkpoint xác nhận 60/60 raw contexts đúng parent gốc. Mean online query **21749,81 ms**, p95 **58464,91 ms**; RAGAS tổng lượt đầu + recovery **1496,10 giây**. Full verification sau cải thiện sẽ chạy lại với tests mới.

## Đợt cải thiện theo yêu cầu user — 04/10/2026

User yêu cầu kiểm tra lý do Faithfulness giảm và cải thiện. Điều tra trước khi sửa:

- Trace RAGAS thật ở `reports/faithfulness_diagnostic.json`: case mật khẩu 12 ký tự có 2 statements. Số 12 verdict=1; filename `mat_khau_v2.md` verdict=0 vì context không chứa source label. Giữ nguyên answer và thêm metadata nguồn thật làm diagnostic score 0,5→1,0. Đây là lỗi đồng bộ input generator/judge, không sửa metric/judge hoặc ground truth.
- Case 120 ngày tái hiện đúng pattern citation; case hoàn chi cho thấy judge không thấy các số liệu tình huống 25 triệu/8 tháng vì NLI chỉ đọc context. Diagnostic score có biến thiên 0,1667→0,3333, không được coi là benchmark mới.
- Probe parent rerank riêng cho câu Senior vẫn để bảng lương thứ 5; không đủ giải quyết coverage. Artifact `reports/parent_rerank_diagnostic.json`.

Design trong phạm vi pipeline hiện có: cùng một formatter source-label + raw parent cho generation và evaluation; checkpoint giữ raw contexts riêng. Với câu nhiều ý, planner chỉ nhận query và sinh 2–3 câu hỏi tìm tài liệu, không nhận answer/golden; cache có model/endpoint/prompt/query. Tìm từng facet, union candidates, reserve 1 parent/facet rồi fill theo full query. Single-query giữ child reranking. Không thêm dữ kiện giả vào context hoặc nới rubric.

4 tests cải thiện ban đầu fail vì tính năng chưa có. Sau triển khai và fix identity fallback, new+pipeline contracts 8/8 pass. Lượt `main.py` production v2 đã chạy trên 20 câu gốc; baseline được dùng lại hợp lệ. Lượt đầu partial do ba timeout, sau recovery đã complete. Artifacts/code/report v1 được giữ tại `reports/history/v1/` trước khi sửa. Không commit/push.

### Kiểm chứng trong lượt v2

- Planner đầu tiên bị dừng trước RAGAS: salary facet mang cả thâm niên, khiến ranking vẫn nghiêng về nghỉ phép. Giữ checkpoint ở `reports/history/v2_attempt/`, không công bố điểm cho lượt này. Prompt `facets-v2` chỉ giữ điều kiện cần cho chính ý đó; probe `reports/facet_rerank_diagnostic.json` chọn đúng leave và salary.
- Review read-only không tìm thấy lỗi quan trọng; bổ sung offline boundary test bắt input generator/judge và raw checkpoint. Full suite trước lượt đo chính thức: **90/90 pass**.
- Lượt chính thức sinh mới 20/20 answers. Case Senior lấy được bảng lương và trả 18 ngày + gross 20–35 triệu/tháng; Junior vẫn tính 20 triệu × 85% = 17 triệu/tháng. Kiểm tra checkpoint: fingerprint khớp và **60/60 raw contexts** đúng parent gốc, không generation errors.
- Một job judge báo `APITimeoutError(Request timed out.)` trong lượt RAGAS chính thức. Không đổi src/config/model/metric đang chạy. Chuẩn bị `scripts/retry_missing_scores.py` để chấm đúng metric thiếu bằng cùng inputs/cấu hình, archive partial trước cập nhật, giữ toàn bộ điểm đã đo và ghi recovery provenance. 4 tests red/green đã xác minh helper không sửa evidence/score thành công, không công bố NaN/null là complete.
- `finalize_analysis.py` được sửa để lấy bottom-5 thật và yêu cầu manual review khớp fingerprint/question/answer; không sửa benchmark JSON. Phân tích v1 giữ riêng trong history. So sánh v1→v2 gồm sửa protocol source framing và retrieval/generation; không gán toàn bộ gain cho retrieval.

### Verification cuối bản cải thiện

- Full pytest **94/94 pass**; `check_lab.py` exit 0, **94/94 pass**, zero TODO markers và report/analysis/reflection đủ.
- `main.py --reuse` và `src/pipeline.py --reuse` exit 0; không API benchmark mới. `verify_reports.py` exit 0, 20/80 mỗi pipeline, 60 raw parents, Qdrant server, corpus/test set nguyên gốc.
- `pip check`: No broken requirements found. Secret scan 107 text files: key project ngoài `.env` bằng 0. SHA256 của ba archived partial reports đều khớp; kiểm chứng 77 scores đã đo và tất cả question/answer/context/golden giữ nguyên trong recovery.
- Bottom-5 v2 được manual review theo đúng fingerprint/answer và render lại. Recall MFA/thâm niên phân biệt lịch sử golden với intent hiện hành; Precision Junior chưa có trace nên không khẳng định parent nhiễu là nguyên nhân duy nhất.
- Branch `feat/production-rag`, thay đổi còn local uncommitted; chưa push hoặc nộp LMS. Hoàn thiện local không đồng nghĩa đã nộp bài.
