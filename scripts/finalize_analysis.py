"""Render measured bottom-5 using manual reviews tied to the current run.

Does not modify benchmark JSON, answers, corpus or ground truth.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import reusable_baseline_report, reusable_production_report
from src.m4_eval import METRICS


def main():
    baseline, prod = reusable_baseline_report(), reusable_production_report()
    assert baseline and prod, 'Need complete matching reports'
    old = json.loads((ROOT / 'reports/history/v1/ragas_report.json').read_text(encoding='utf-8'))
    reviewed = json.loads((ROOT / 'analysis/reviewed_failures.json').read_text(encoding='utf-8'))
    assert reviewed['input_fingerprint'] == prod['run_metadata']['input_fingerprint'], 'Reviews are stale'
    lines = ['# Failure Analysis — Lab 18: Production RAG', '',
             '**Học viên:** Nguyễn Nhân Sâm · 2A202602672 · K4 Track 3A', '',
             '## Kết quả đo thật', '',
             'Cùng corpus và 20 câu gốc; generator/judge gpt-5.6-luna, local bge-m3, RAGAS 0.1.22. '
             'Baseline dùng lại sau kiểm chứng fingerprint; production v2 sinh mới toàn bộ answers. '
             'Mỗi pipeline đủ 20 LLM answers/80 scores hữu hạn, không còn lỗi chưa phục hồi. '
             'Production dùng Qdrant server. Tác vụ timeout, nếu có, được chấm lại riêng bằng cùng cấu hình; '
             'các scores thành công được giữ nguyên, lượt partial được lưu trong history.', '',
             '| Metric | Baseline | Production v1 | Production v2 | v2 − v1 |',
             '|---|---:|---:|---:|---:|']
    for metric in METRICS:
        b, v1, v2 = baseline['aggregate'][metric], old['aggregate'][metric], prod['aggregate'][metric]
        lines.append(f'| {metric} | {b:.6f} | {v1:.6f} | {v2:.6f} | {v2-v1:+.6f} |')
    lines += ['', 'V1→v2 đồng thời sửa cách biểu diễn source trong evaluator và thay đổi retrieval/generation. '
              'Không gán toàn bộ chênh lệch Faithfulness cho retrieval; đây là một lượt judge, chưa có ablation '
              'toàn bộ bộ câu hoặc khoảng tin cậy. Baseline trước đó đã dùng context nhất quán giữa generator/judge.', '',
              '## Nguyên nhân đã xác minh và sửa', '',
              '1. **Boundary source:** v1 generator nhận filename nhưng judge chỉ nhận parent raw. '
              'Trace case mật khẩu 12 ký tự chấp nhận số 12 và từ chối citation filename. '
              'Giữ nguyên answer, gắn metadata nguồn thật làm diagnostic Faithfulness 0,5→1,0. '
              'V2 dùng `_evidence_contexts()` cho cả hai bên; checkpoint giữ nguyên raw parent.',
              '2. **Coverage nhiều ý:** v1 tìm được salary child nhưng rerank xếp thứ 12; '
              'parent-only probe vẫn thứ 5. Planner tách từng ý và chỉ giữ điều kiện liên quan; '
              'pipeline dành một parent mỗi facet, dedup và giữ tối đa 3 parent. '
              'Planner chỉ nhận query, không nhận golden. Đây là heuristic, không bảo đảm coverage mọi câu.', '',
              '## Bottom-5 theo trung bình bốn metrics', '',
              'Năm câu dưới đây lấy trực tiếp từ `failures` của report v2, không chọn theo kết luận mong muốn. '
              'Ground truth/corpus giữ nguyên; diagnosis phân biệt chứng cứ đọc tay và giả thuyết về judge.']
    for rank, failure in enumerate(prod['failures'], 1):
        index = next(i+1 for i, row in enumerate(prod['per_question']) if row['question'] == failure['question'])
        review = reviewed['reviews'][str(index)]
        assert review['question'] == failure['question'] and review['answer'] == failure['answer'], 'Review mismatch'
        sources = prod['run_metadata']['retrieval_sources'][index-1]
        lines += ['', f'### #{rank} — Câu {index}', '', f'**Question:** {failure["question"]}', '',
                  f'**Expected (golden gốc):** {failure["ground_truth"]}', '',
                  '**Got (answer thật):**', '', '> '+failure['answer'].replace('\n', '\n> '), '',
                  f'**Mean:** {failure["score"]:.6f}; **Worst metric:** {failure["worst_metric"]}.', '',
                  '; '.join(f'{m}: {failure["metrics"][m]:.6f}' for m in METRICS)+'.', '',
                  '**Sources:** '+', '.join(s['source'] for s in sources)+'.', '',
                  '**Evidence:** '+review['evidence'], '', '**Diagnosis:** '+review['diagnosis'], '',
                  '**Error Tree:** '+review['error_tree'], '', '**Suggested fix:** '+review['fix']]
    case = prod['per_question'][11]
    sources = prod['run_metadata']['retrieval_sources'][11]
    lines += ['', '## Kiểm chứng câu Senior 9 năm', '',
              '**Nguồn v2:** '+', '.join(s['source'] for s in sources)+'.', '',
              '> '+case['answer'].replace('\n', '\n> '), '',
              '; '.join(f'{m}: {case[m]:.6f}' for m in METRICS)+'.', '',
              'V1 không có bảng lương trong context. So sánh answer và sources v2 ở trên là kiểm chứng '
              'end-to-end của case này; không suy ra mọi câu nhiều ý đều được giải quyết.', '',
              '## Giới hạn và bước tiếp theo', '',
              '- NLI Faithfulness của RAGAS 0.1.22 không nhận question khi kiểm chứng statement. '
              'Trace hoàn chi đã từ chối số tiền/thời gian do người hỏi cung cấp; không đưa question/golden '
              'vào context để tăng điểm.',
              '- Tạm ứng: nguồn chỉ có 2%/tháng, golden thêm prorate 5 ngày bằng tháng 30 ngày. '
              'Cần owner chính sách xác nhận công thức; không sửa corpus/golden lab.',
              '- Password golden yêu cầu lịch sử bản cũ dù query hỏi hiện hành. '
              'Giữ version filter và nguồn hiện hành; đánh giá completeness lịch sử riêng.',
              '- Đo trên bộ câu mới và ablation source formatting/planner/prompt; '
              'planner gọi thêm API khi chưa cache, rerank parent nhiều facet tăng CPU latency.', '',
              'Raw evidence được kiểm tra bởi `scripts/verify_reports.py`; source labels chỉ là metadata '
              'gốc, không phải dữ kiện do LLM sinh. Chi tiết latency tại `reports/latency_breakdown.md`. '
              'Bản v1 và analysis cũ giữ tại `reports/history/v1/`. Chưa commit/push hoặc nộp LMS.']
    (ROOT / 'analysis/failure_analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Bottom-5 analysis rendered; benchmark JSON unchanged.')


if __name__ == '__main__':
    main()
