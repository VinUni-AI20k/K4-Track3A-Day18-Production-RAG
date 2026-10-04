# Latency breakdown — Lab 18

Python 3.11.7; models local chạy CPU; API generator/judge gpt-5.6-luna; Qdrant server. Đơn vị ms; số đo từ run_metadata của production report.

## Build đã chạy

| Bước | Thời gian (ms) | Phạm vi |
|---|---:|---|
| load_chunk_ms | 226.86 | Đọc tài liệu và hierarchical chunking |
| enrichment_ms | 168.09 | Build sau review: dùng cache enrichment; không phải 101 request mới |
| index_ms | 55744.55 | Load embedding, BM25, encode corpus và Qdrant index |
| reranker_load_ms | 4281.82 | Load CrossEncoder thật trước query |

Enrichment trạng thái: {"cache": 101}. Lần build đầu đã làm enrichment thật nhưng dừng để sửa review; thời gian toàn lượt đó không được lưu. Không suy ra cold-enrichment latency từ số đo cache.

## Online query (20 câu, mỗi câu chạy một lần)

| Bước | Mean (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| retrieval_ms | 154.47 | 149.04 | 195.29 | 129.00 | 207.95 |
| rerank_ms | 4014.75 | 4046.24 | 5110.80 | 3186.43 | 5490.60 |
| parent_expand_ms | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| generation_ms | 5803.84 | 5337.33 | 9020.57 | 4088.83 | 10482.42 |
| total_online_ms | 9973.07 | 9313.27 | 13484.48 | 7574.25 | 14845.68 |

retrieval gồm BM25 + query embedding + Qdrant + RRF/filter. rerank chấm các candidates raw child trước khi chọn tối đa 3 parent khác nhau. generation gồm thời gian API/network. Đây là 20 câu khác nhau trên máy đang dùng, không phải thử tải đồng thời hay SLA.

## Evaluation và benchmark riêng

- RAGAS 80 metric tasks (20 × 4), 2 workers: 523.37 giây; bao gồm khởi tạo judge/embedding, không thuộc online answer latency.
- Baseline stdout phiên chạy trước: 1445,3 giây tổng, RAGAS khoảng 19 phút 11 giây. Baseline JSON không có breakdown đầy đủ; không suy diễn latency từng bước.
- CrossEncoder smoke riêng (`m3_reranking_report.json`): model load 5607,60 ms; first inference 4361,08 ms; 5 warm runs mean 4377,42 ms, min 4338,40 / max 4436,63 ms. Input benchmark này khác các query production.
- M5 smoke fixture (`m5_enrichment_smoke.json`): 1 API request 16816,82 ms. Không dùng một fixture để ước lượng chính xác build corpus.

Download models và tests không nằm trong online query table. Reranker optional Flashrank chưa triển khai; không có số đo dưới 5 ms cho bài này. Có thể thử GPU/model nhẹ và ablation enrichment sau lab, rồi đo lại trên cùng máy/bộ câu trước khi kết luận cải thiện.
