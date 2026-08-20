# 📐 TÀI LIỆU KỸ THUẬT: IT SCORING V3 ENGINE & PHÂN VỊ CẤP BẬC

Tài liệu này quy định chi tiết về kiến trúc thuật toán chấm điểm **IT Scoring V3**, công thức tính toán theo trọng số, cơ chế bảo vệ điểm số (**Ceiling Gate Guardrails**), phân vị cấp bậc ứng viên và cơ chế cấu hình trọng số động.

---

## 1. CÔNG THỨC CHẤM ĐIỂM & TRỌNG SỐ CHUẨN NGÀNH IT

Thuật toán Scoring V3 đánh giá ứng viên độc lập trên **6 Tiêu chí (Thang điểm 0 - 100)**:

$$\text{Score} = 30\% \cdot \text{Kỹ thuật} + 25\% \cdot \text{Thành tựu} + 15\% \cdot \text{Doanh nghiệp} + 15\% \cdot \text{Học vấn} + 10\% \cdot \text{Chứng chỉ} + 5\% \cdot \text{Ngoại ngữ}$$

| Mã Tiêu chí | Tên Tiêu chí (Tiếng Việt) | Trọng số Mặc định | Ý nghĩa Đánh giá |
| :--- | :--- | :---: | :--- |
| `TECHNICAL_DEPTH` | **Kỹ năng & Kiến trúc** | **30%** | Chiều sâu công nghệ lõi, năng lực thiết kế hệ thống phân tán, xử lý tải lớn (production architecture). |
| `IMPACT_METRICS` | **Dự án & Thành tựu** | **25%** | Thành tựu đo lường được bằng số liệu định lượng cụ thể (% tăng trưởng, giảm latency ms, giảm chi phí $). |
| `ENTERPRISE_SCALE` | **Quy mô Doanh nghiệp** | **15%** | Uy tín và môi trường thực chiến (Big Tech, Unicorn, Bank Tier 1, Product vs Outsourcing/SME). |
| `EDUCATION` | **Học vấn & CS Foundation** | **15%** | Bằng cấp và uy tín trường đại học đào tạo (Tier 1A, Tier 1B, Accredited, Non-degree). |
| `CERTIFICATIONS` | **Chứng chỉ Chuyên môn** | **10%** | Chứng chỉ quốc tế có giá trị cao (AWS Pro, CKA, GCP Pro, PMP, TOGAF). |
| `LANGUAGE_PROFICIENCY`| **Năng lực Ngoại ngữ** | **5%** | Khả năng làm việc trong môi trường quốc tế (IELTS, TOEIC, kinh nghiệm dự án nước ngoài). |

---

## 2. BẢNG PHÂN VỊ CẤP BẬC KINH NGHIỆM (SENIORITY CALIBRATED BANDS)

Cấp bậc được ánh xạ trực tiếp từ **Điểm tổng hợp V3** (Weighted Score) theo phân vị thực tế của thị trường tuyển dụng CNTT:

| Thang Điểm V3 | Bậc Kinh nghiệm (`seniorityCalibratedLevel`) | Chân dung Ứng viên Điển hình |
| :---: | :--- | :--- |
| **$\ge 85.0\text{đ}$** | **`SENIOR_LEAD`** | Senior/Lead/Architect dày dặn $\ge 5\text{ năm}$, thiết kế hệ thống lớn, thành tựu định lượng rõ rệt. |
| **$72.0 - 84.9\text{đ}$** | **`SENIOR`** | Senior vững vàng $\ge 3-5\text{ năm}$, thành thạo stack production, giải quyết bài toán phức tạp. |
| **$58.0 - 71.9\text{đ}$** | **`MID_LEVEL`** | Mid-level $2-3\text{ năm}$ kinh nghiệm thực chiến, hoàn thành công việc độc lập. |
| **$45.0 - 57.9\text{đ}$** | **`FRESHER_JUNIOR`** | Junior $1\text{ năm}$ hoặc Fresher mới ra trường có nền tảng cơ bản (ví dụ: Huỳnh Thanh Tùng $50.5\text{đ}$). |
| **$< 45.0\text{đ}$** | **`INTERN_TRAINEE`** | Thực tập sinh, sinh viên chưa có kinh nghiệm thực tế. |

---

## 3. CƠ CHẾ BẢO VỆ ĐIỂM SỐ CHỐNG LẠM PHÁT (CEILING GATE GUARDRAILS)

Nhằm tránh trường hợp AI chấm điểm phóng đại không có căn cứ, hệ thống tự động khóa cận trên điểm số (**Ceiling Gate**) theo bằng chứng thực tế:
* **Không có bằng chứng thiết kế kiến trúc hệ thống:** `TECHNICAL_DEPTH` $\le 83.0\text{đ}$.
* **Dự án không có số liệu định lượng đo lường:** `IMPACT_METRICS` $\le 79.0\text{đ}$.
* **Chỉ làm việc tại công ty nhỏ / freelance:** `ENTERPRISE_SCALE` $\le 64.0\text{đ}$.
* **Không có bằng đại học chính quy:** `EDUCATION` $\le 54.0\text{đ}$.
* **Không có chứng chỉ CNTT:** `CERTIFICATIONS` $\le 39.0\text{đ}$.
* **Không có thông tin ngoại ngữ:** `LANGUAGE_PROFICIENCY` $\le 49.0\text{đ}$.

---

## 4. CƠ CHẾ CẤU HÌNH TRỌNG SỐ ĐỘNG (DYNAMIC HR WEIGHTING)

Khi Nhà tuyển dụng tùy chỉnh trọng số hoặc chỉ chọn $k$ trên 6 tiêu chí ($k \le 6$), hệ thống áp dụng công thức **Chuẩn hóa Trọng số Tỷ lệ**:

$$\text{Weight Normalization:}\quad w_i' = \frac{w_i}{\sum_{j \in \text{Selected}} w_j}$$

$$\text{Final Score:}\quad \text{Score} = \sum_{i \in \text{Selected}} w_i' \cdot \text{Score}_i$$

* **Nếu HR tăng/giảm trọng số bất kỳ:** Tổng trọng số chuẩn hóa $\sum w_i'$ luôn bằng $1.0$ ($100\%$), thang điểm luôn nằm trong dải $[0, 100]$.
* **Nếu HR chỉ chọn 3 tiêu chí (ví dụ: Kỹ năng 30%, Dự án 25%, Doanh nghiệp 15%):**
  * Tổng trọng số thô $= 70\%$.
  * Tỷ trọng chuẩn hóa: Kỹ năng $= \frac{30}{70} \approx 42.86\%$, Dự án $= \frac{25}{70} \approx 35.71\%$, Doanh nghiệp $= \frac{15}{70} \approx 21.43\%$.
  * Điểm tổng luôn đảm bảo chuẩn xác trên thang 100 điểm.

---

## 5. CHUẨN ĐẦU RA API (`PUT /api/v1/cvs/{id}/scored`)

Toàn bộ kết quả phản hồi của Core tuân thủ cấu trúc hợp đồng chuẩn:
- `Score`: Điểm tổng hợp chuẩn ngành (thang 100).
- `criteriaScores`: Chi tiết 6 tiêu chí kèm điểm số, tier và giải thích tiếng Việt.
- `summary`: Tóm tắt 2-3 câu tiếng Việt súc tích, 100% PII-free.
- `scoringRationale`: Lời giải trình điểm tổng hợp và cấp bậc.
- `seniorityCalibratedLevel`: Cấp bậc phân vị chuẩn ngành.
- `scoringVersion`: `"v3_rubric"`.
