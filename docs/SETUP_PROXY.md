# Cấu hình proxy đã kiểm tra cho Lab 18

User chọn generation/judge `gpt-5.6-luna` từ danh sách model proxy trả về, embedding local `BAAI/bge-m3`. Đây là cấu hình đánh giá của bài làm, không phải điểm benchmark công bố cho model embedding/judge khác.

Đặt cấu hình riêng trong `.env` ở root repo:

```dotenv
OPENAI_API_KEY=<key của bạn>
OPENAI_BASE_URL=<URL API proxy>/v1
GENERATION_MODEL=gpt-5.6-luna
EVALUATION_MODEL=gpt-5.6-luna
EVALUATION_EMBEDDING_BACKEND=local
EVALUATION_EMBEDDING_MODEL=BAAI/bge-m3
```

`config.py` đọc `.env` từ đúng thư mục repo và ưu tiên giá trị trong file này hơn environment tiến trình. Nếu URL chỉ có host gốc, config thêm `/v1`; URL có đường dẫn tùy chỉnh được giữ nguyên. Không tự sửa file `.env` hoặc đưa key vào report.

Các lỗi setup đã xác minh:

- URL host gốc trả `HTTP 200`, content-type `text/html`; đó không phải response API hợp lệ.
- Thử `gpt-4o-mini` ở endpoint `/v1` trả `model_not_found`: `No available channel for model gpt-4o-mini under group default (distributor)`.
- `text-embedding-3-small` không có trong danh sách model proxy đã kiểm tra; evaluation dùng embedding local theo lựa chọn của user.
- Với client lấy đúng key/URL project và `gpt-5.6-luna`, chat smoke đã trả answer hợp lệ.

Đánh giá sử dụng RAGAS 0.1.22. `answer_relevancy` giữ strictness mặc định 3. Judge dùng temperature 0, 2 workers, timeout 120 giây mỗi tác vụ và tối đa 1 retry. Embedding dùng model đã cache và normalize embeddings.

Chạy baseline từ root repo bằng PowerShell:

```powershell
& .\.venv\Scripts\python.exe naive_baseline.py
```

RAGAS là đánh giá có gọi API và có thể mất nhiều phút. Unit tests cô lập external judge; kết quả mocked của tests không được dùng làm điểm bài nộp. Smoke một câu nằm riêng tại `reports/m4_ragas_smoke.json` và ghi rõ answer fixture viết tay, không phải baseline/production benchmark.

Trong reports, chỉ coi `eval_status=complete` và đủ `metric_counts` là lượt đo đầy đủ. `partial`/`failed` lưu điểm thiếu bằng `null`; không dùng các giá trị fallback trong giao diện scaffold làm điểm thật. `num_questions` là số câu có evidence, không tự chứng minh chấm thành công. `generation_modes` và `generation_success_count` phân biệt LLM generation với câu trả lời trích nguyên văn fallback.

Baseline và production phải dùng cùng bộ 20 câu gốc và cùng cấu hình generator/judge/embedding. Báo cáo production được tạo ở bước tích hợp sau M5.
