# Latency breakdown — Lab 18

Python 3.11.7; models local chạy CPU; API generator/judge gpt-5.6-luna; Qdrant server. Đơn vị ms; số đo từ run_metadata của production report.

## Build đã chạy

| Bước | Thời gian (ms) | Phạm vi |
|---|---:|---|
| load_chunk_ms | 250.64 | Đọc tài liệu và hierarchical chunking |
| enrichment_ms | 24.34 | Build sau review: dùng cache enrichment; không phải 101 request mới |
| index_ms | 64795.01 | Load embedding, BM25, encode corpus và Qdrant index |
| reranker_load_ms | 4462.34 | Load CrossEncoder thật trước query |

Enrichment trạng thái: {"cache": 101}. Lần build đầu đã làm enrichment thật nhưng dừng để sửa review; thời gian toàn lượt đó không được lưu. Không suy ra cold-enrichment latency từ số đo cache.

## Online query (20 câu, mỗi câu chạy một lần)

| Bước | Mean (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| retrieval_ms | 1510.06 | 184.97 | 9918.32 | 149.42 | 16975.47 |
| rerank_ms | 9774.12 | 4925.63 | 38579.67 | 3848.59 | 46375.71 |
| parent_expand_ms | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| generation_ms | 10465.62 | 8329.03 | 21089.44 | 1933.11 | 43142.60 |
| total_online_ms | 21749.81 | 14751.97 | 58464.91 | 6348.23 | 67305.18 |

retrieval gồm query planning (nếu câu nhiều ý), BM25 + query embedding + Qdrant + RRF/filter. Single query rerank raw child; nhiều ý rerank parent cho từng facet rồi fill theo câu gốc. Planner có cache và có thể gọi thêm 1 API request/câu nhiều ý chưa cache. generation gồm thời gian API/network. Đây là 20 câu khác nhau trên máy đang dùng, không phải thử tải đồng thời hay SLA.

## Evaluation và benchmark riêng

- RAGAS 80 metric tasks (20 × 4), 2 workers: 1496.10 giây; bao gồm khởi tạo judge/embedding và thời gian recovery (nếu có), không thuộc online answer latency.
- Baseline stdout phiên chạy trước: 1445,3 giây tổng, RAGAS khoảng 19 phút 11 giây. Baseline JSON không có breakdown đầy đủ; không suy diễn latency từng bước.
- CrossEncoder smoke riêng (`m3_reranking_report.json`): model load 5607,60 ms; first inference 4361,08 ms; 5 warm runs mean 4377,42 ms, min 4338,40 / max 4436,63 ms. Input benchmark này khác các query production.
- M5 smoke fixture (`m5_enrichment_smoke.json`): 1 API request 16816,82 ms. Không dùng một fixture để ước lượng chính xác build corpus.

Download models và tests không nằm trong online query table. Reranker optional Flashrank chưa triển khai; không có số đo dưới 5 ms cho bài này. Có thể thử GPU/model nhẹ và ablation enrichment sau lab, rồi đo lại trên cùng máy/bộ câu trước khi kết luận cải thiện.

Recovery: 6 metric tasks được chấm bổ sung vì lượt đầu thiếu điểm. Scores thành công giữ nguyên; thời gian bổ sung cộng vào evaluation_ms. Provenance và lượt partial gốc lưu trong run_metadata.evaluation_recovery.
