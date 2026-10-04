"""Generate measured latency table only from a complete saved production report."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import reusable_production_report


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered)-1) * q
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower] + (ordered[upper]-ordered[lower]) * (position-lower)


def main():
    report = reusable_production_report()
    assert report is not None, 'Need a complete matching report'
    meta = report['run_metadata']
    build = meta['build']
    lines = ['# Latency breakdown — Lab 18', '',
             'Python 3.11.7; models local chạy CPU; API generator/judge gpt-5.6-luna; Qdrant server. '
             'Đơn vị ms; số đo từ run_metadata của production report.', '',
             '## Build đã chạy', '', '| Bước | Thời gian (ms) | Phạm vi |', '|---|---:|---|']
    descriptions = {'load_chunk_ms': 'Đọc tài liệu và hierarchical chunking',
                    'enrichment_ms': 'Build sau review: dùng cache enrichment; không phải 101 request mới',
                    'index_ms': 'Load embedding, BM25, encode corpus và Qdrant index',
                    'reranker_load_ms': 'Load CrossEncoder thật trước query'}
    for name, value in build['timings_ms'].items():
        lines.append(f'| {name} | {value:.2f} | {descriptions[name]} |')
    lines += ['', 'Enrichment trạng thái: '+json.dumps(build['enrichment_counts'], ensure_ascii=False)+'. '
              'Lần build đầu đã làm enrichment thật nhưng dừng để sửa review; thời gian toàn lượt đó không được lưu. '
              'Không suy ra cold-enrichment latency từ số đo cache.', '',
              '## Online query (20 câu, mỗi câu chạy một lần)', '',
              '| Bước | Mean (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |', '|---|---:|---:|---:|---:|---:|']
    timings = [row['timings_ms'] for row in meta['query_metadata']]
    values_by_stage = {stage: [row[stage] for row in timings] for stage in timings[0]}
    values_by_stage['total_online_ms'] = [sum(row.values()) for row in timings]
    for name, values in values_by_stage.items():
        lines.append(f'| {name} | {statistics.mean(values):.2f} | {percentile(values,.5):.2f} | '
                     f'{percentile(values,.95):.2f} | {min(values):.2f} | {max(values):.2f} |')
    lines += ['', 'retrieval gồm query planning (nếu câu nhiều ý), BM25 + query embedding + Qdrant + RRF/filter. '
              'Single query rerank raw child; nhiều ý rerank parent cho từng facet rồi fill theo câu gốc. '
              'Planner có cache và có thể gọi thêm 1 API request/câu nhiều ý chưa cache. generation gồm thời gian API/network. '
              'Đây là 20 câu khác nhau trên máy đang dùng, không phải thử tải đồng thời hay SLA.', '',
              '## Evaluation và benchmark riêng', '',
              f'- RAGAS 80 metric tasks (20 × 4), 2 workers: {meta["evaluation_ms"]/1000:.2f} giây; '
              'bao gồm khởi tạo judge/embedding và thời gian recovery (nếu có), không thuộc online answer latency.',
              '- Baseline stdout phiên chạy trước: 1445,3 giây tổng, RAGAS khoảng 19 phút 11 giây. '
              'Baseline JSON không có breakdown đầy đủ; không suy diễn latency từng bước.',
              '- CrossEncoder smoke riêng (`m3_reranking_report.json`): model load 5607,60 ms; '
              'first inference 4361,08 ms; 5 warm runs mean 4377,42 ms, min 4338,40 / max 4436,63 ms. '
              'Input benchmark này khác các query production.',
              '- M5 smoke fixture (`m5_enrichment_smoke.json`): 1 API request 16816,82 ms. '
              'Không dùng một fixture để ước lượng chính xác build corpus.', '',
              'Download models và tests không nằm trong online query table. Reranker optional Flashrank chưa '
              'triển khai; không có số đo dưới 5 ms cho bài này. Có thể thử GPU/model nhẹ và ablation enrichment '
              'sau lab, rồi đo lại trên cùng máy/bộ câu trước khi kết luận cải thiện.']
    if meta.get('evaluation_recovery'):
        attempts = [a for recovery in meta['evaluation_recovery'] for a in recovery['attempts']]
        lines += ['', f'Recovery: {len(attempts)} metric tasks được chấm bổ sung vì lượt đầu thiếu điểm. '
                  'Scores thành công giữ nguyên; thời gian bổ sung cộng vào evaluation_ms. '
                  'Provenance và lượt partial gốc lưu trong run_metadata.evaluation_recovery.']
    (ROOT/'reports/latency_breakdown.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Latency report generated from verified saved measurements.')


if __name__ == '__main__':
    main()
