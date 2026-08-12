# Smart CV AI Core Engine
---

## 1. YÊU CẦU HỆ THỐNG & CÀI ĐẶT

### Yêu cầu phần mềm:
- **Python:** Version `3.10+`.
- **Hệ điều hành:** Linux (Ubuntu 22.04 LTS / Debian) hoặc Windows 10/11.

### Các bước cài đặt:
```bash
# 1. Clone repository
git clone <URL_REPO_AI_CORE>
cd smart-cv-ai-core

# 2. Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# Hoặc trên Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 3. Cài đặt các thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. CẤU HÌNH BIẾN MÔI TRƯỜNG

Tạo file `.env` tại thư mục gốc của dự án (tham khảo file mẫu `.env.example`):

---

## 3. BACKEND INTEGRATION

Khi Backend gửi yêu cầu xử lý CV sang AI Core qua API:

### 1. Endpoint Tiếp nhận (`POST /api/v1/cv/ai-handle`):
- **Content-Type:** `multipart/form-data` (gửi `id` và `file` PDF) hoặc `application/json` (gửi `cvId` và `fileUrl`).

### 2. 3 HTTP PUT Callbacks:

1. **Callback 1 — Dữ liệu bóc tách (`PUT /api/v1/cvs/{id}/extracted`):**
   - Trả về JSON Profile đầy đủ (`candidateName`, `skills`, `companies`, `jobTitles`, `education`, `projects`, `piiFields`...) để Backend lưu DB và hiển thị UI.
2. **Callback 2 — Điểm số Rubric (`PUT /api/v1/cvs/{id}/scored`):**
   - Trả về điểm số tổng (`Score: float`) và chi tiết từng tiêu chí phục vụ xếp hạng ứng viên.
3. **Callback 3 — Vector Embedding (`PUT /api/v1/cvs/{id}/embedded`):**
   - Trả về mảng L2-Normalized Vector 1024 float (`embedding: [1024 floats]`, `dimension: 1024`, `model: "BAAI/bge-m3"`) để Backend lưu vào cột `pgvector` phục vụ Vector Search.

---

## 4. SCRIPT KIỂM THỬ

Để kiểm tra độc lập luồng xử lý của 1 file CV PDF mà không cần bật server:

```powershell
python inspect_pipeline_stages.py "<path_to_cv.pdf>" "<output_folder>"
```
