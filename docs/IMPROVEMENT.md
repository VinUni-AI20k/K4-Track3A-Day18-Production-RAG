# Điều tra và cải thiện Production RAG

## Vì sao Faithfulness v1 giảm?

Trace judge thật (`reports/faithfulness_diagnostic.json`) tái hiện câu mật khẩu: statement “tối thiểu 12 ký tự” được verdict 1, nhưng statement “nguồn là mat_khau_v2.md” được verdict 0. Judge giải thích context không cung cấp tên file. Generator thực tế đã nhận nhãn nguồn đó, còn evaluator chỉ nhận parent raw. Đây là lỗi đồng bộ dữ liệu tại boundary generation/evaluation.

Thử giữ nguyên answer và thêm đúng metadata source vào context: Faithfulness case 6 từ 0,5 lên 1,0. Case 7 cũng tái hiện lỗi citation. Sửa bằng `_evidence_contexts()` dùng chung cho cả generator/judge; raw contexts vẫn được lưu riêng và đối chiếu nguyên văn với parent gốc.

Ở case hoàn chi đào tạo, trace còn từ chối số 25 triệu/8 tháng vì chúng nằm trong câu hỏi, không trong context. Judge NLI của RAGAS 0.1.22 không nhận question ở bước kiểm chứng statement. Đây là giới hạn cần trình bày trung thực; không thêm câu hỏi/ground truth vào context để làm tăng điểm.

## Vì sao câu Senior thiếu lương?

Bảng lương đã có trong candidates v1, nhưng rerank child toàn câu xếp bảng lương thứ 12. Rerank parent đầy đủ một mình vẫn xếp nó thứ 5; không đủ để vào top-3. Câu hỏi hai ý bị phần thâm niên/nghỉ phép chi phối.

Thử planner đầu tiên vẫn thất bại vì truy vấn lương mang cả “9 năm thâm niên”. Planner được sửa để mỗi truy vấn chỉ giữ điều kiện cần cho chính ý đó. Probe thật (`reports/facet_rerank_diagnostic.json`) chọn:

- Ý nghỉ phép: `nghi_phep_nam_v2024.md`, score 0,990147.
- Ý lương: `bang_luong_2024.md`, score 0,298065, đứng đầu nhóm candidates thử.

Pipeline union các candidates theo query/facets và dành một parent cho mỗi facet trước khi fill theo câu hỏi gốc. Facet planning chỉ nhận question, không nhận ground truth. Parent selection là heuristic, chưa bảo đảm đủ dữ kiện cho mọi câu nhiều ý. Câu đơn vẫn dùng child reranking.

## Cách kiểm chứng

Giữ corpus/test set, generator/judge model và 4 metrics gốc. Chạy mới toàn bộ 20 câu sau khi source ổn định. Báo cáo v1 và source snapshot ở `reports/history/v1/`; lượt thử planner bị dừng trước RAGAS ở `reports/history/v2_attempt/`, không công bố scores cho lượt đó.

Điểm v1→v2 gồm cả sửa protocol context lẫn thay đổi retrieval/generation. Không gán toàn bộ mức tăng cho chất lượng retrieval; diagnostic cùng answer/citation chỉ xác định nguyên nhân trên case đã thử, chưa định lượng tỷ trọng cả bộ.

## Kết quả v2 đã đo

| Metric | V1 | V2 |
|---|---:|---:|
| Faithfulness | 0,755952 | 0,938214 |
| Answer Relevancy | 0,832710 | 0,856665 |
| Context Precision | 0,975000 | 0,975000 |
| Context Recall | 0,825000 | 0,875000 |

Đủ 20 LLM answers và 80 scores hữu hạn. Ba ô thiếu do timeout được chấm bổ sung bằng cùng inputs/model/metric; 77 scores thành công giữ nguyên. Sáu tác vụ retry qua ba lượt (gồm các lần retry vẫn lỗi) được ghi trong `run_metadata.evaluation_recovery`, với bản partial gốc và hash archive. CLI chính ban đầu exit 1 vì partial; recovery và cả hai entry points `--reuse` sau đó exit 0. Không mô tả lượt chính ban đầu là hoàn tất không lỗi.

Case Senior lấy đúng bảng lương và chính sách phép: answer 18 ngày + 20–35 triệu/tháng; Faithfulness/Recall đều 1,0. Case password 6 và 7 Faithfulness đều từ 0,5 lên 1,0. Mean online query tăng từ 9,97 lên 21,75 giây, p95 từ 13,48 lên 58,46 giây; parent rerank nhiều facet và API latency góp phần làm chậm. Đây là số đo hai lượt trên máy hiện tại, chưa có controlled latency ablation.

Bottom-5 mới được review riêng trong `analysis/reviewed_failures.json` và render ở `analysis/failure_analysis.md`. Tạm ứng vẫn thiếu công thức prorate trong nguồn; golden MFA/thâm niên/mật khẩu còn yêu cầu lịch sử dù query hỏi hiện hành. Các hạn chế này được giữ rõ, không sửa corpus/golden để tăng điểm.
