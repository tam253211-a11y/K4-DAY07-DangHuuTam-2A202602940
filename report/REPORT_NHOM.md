# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** G41
**Thành viên:** 
    Nguyễn Hoàng Việt
    Đặng Hữu Tâm
    Phạm Quân
    Nguyễn Đỗ Chiến Thắng
**Ngày:** 19/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Quy định và Dịch vụ Thư viện Trường Đại học Bách Khoa (ĐHQG-HCM)

**Tại sao nhóm chọn chủ đề này?**
> Thư viện Trường Đại học Bách Khoa và Thư viện Trung tâm ĐHQG-HCM là hệ thống dịch vụ học vụ thiết yếu, có cơ cấu quy chế rõ ràng và phân tầng bạn đọc chặt chẽ (sinh viên, giảng viên, nghiên cứu sinh). Bộ tài liệu này sở hữu các quy tắc nghiệp vụ định lượng cụ thể (hạn mức mượn, ngày gia hạn, phí phạt, bản quyền sao chụp) và metadata phân quyền (`audience`), tạo môi trường lý tưởng để kiểm chứng năng lực truy xuất ngữ nghĩa và hiệu quả của cơ chế tiền lọc metadata trong mô hình RAG.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Quy định mượn trả tài liệu cho sinh viên chính quy | https://www.vnulib.edu.vn/index.php/muon-tra-tai-lieu-tvtt | 2026-09-19 / 2024-2025 | 1,820 | `audience: student`, `department: library`, `category: circulation`, `lang: vi` |
| 2 | Quy định mượn tài liệu cho giảng viên và nghiên cứu sinh | https://vnulib.edu.vn/index.php/general/36-dich-vu-thu-vien/144-muon-tra-tliru-tv | 2026-09-19 / 2024-2025 | 1,869 | `audience: faculty`, `department: library`, `category: circulation`, `lang: vi` |
| 3 | Hướng dẫn sử dụng thư viện và quy định thẻ bạn đọc | https://vnulib.edu.vn/index.php/general/36-dich-vu-thu-vien/148-huong-dan-su-dung-thu-vien | 2026-09-19 / 2024-2025 | 1,550 | `audience: all`, `department: library`, `category: user-guide`, `lang: vi` |
| 4 | Quy định lớp tập huấn kỹ năng tìm kiếm và sử dụng CSDL | https://vnulib.edu.vn/index.php/general/36-dich-vu-thu-vien/147-tap-huan-tv | 2026-09-19 / 2024-2025 | 1,450 | `audience: student`, `department: library`, `category: training`, `lang: vi` |
| 5 | Dịch vụ sao chụp và cung cấp tài liệu theo yêu cầu | https://vnulib.edu.vn/index.php/general/36-dich-vu-thu-vien/146-cung-cap-thong-tin-theo-yeu-cau-tvtt | 2026-09-19 / 2024-2025 | 1,600 | `audience: all`, `department: library`, `category: information-services`, `lang: vi` |
| 6 | Thông báo điều chỉnh chính sách mượn trả tài liệu thư viện | https://vnulib.edu.vn/index.php/general/9-tin-tuc-su-kien-thong-bao/428-tb-dieu-chinh-muon-tra-tl-2026 | 2026-09-19 / 2024-2025 | 1,720 | `audience: all`, `department: library`, `category: policy`, `lang: vi` |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `audience` | string | `student`, `faculty`, `all` | Phân quyền truy xuất quan trọng nhất: cho phép tiền lọc để câu hỏi của sinh viên không lấy nhầm quy chế đặc quyền của giảng viên. |
| `department` | string | `library` | Xác định cơ quan ban hành, cho phép mở rộng hệ thống sang các phòng ban khác (như `academic-affairs`, `student-affairs`). |
| `category` | string | `circulation`, `policy`, `training` | Hỗ trợ lọc theo mảng nghiệp vụ (mượn trả, tập huấn, xử lý vi phạm, sao chụp). |
| `document_version` | string | `2024-2025` | Đảm bảo tính cập nhật và chống trích dẫn các quy chế đã hết hiệu lực. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `hcmut-lib-borrowing-student.md` | FixedSizeChunker (`fixed_size`) | 11 | 199.6 | Kém — Bị cắt ngang câu giữa chừng, làm mất con số 21 ngày và hạn mức 5 cuốn. |
| `hcmut-lib-borrowing-student.md` | SentenceChunker (`by_sentences`) | 6 | 281.2 | Tốt — Giữ nguyên câu, nhưng một số điều khoản dài bị dồn chung. |
| `hcmut-lib-borrowing-student.md` | RecursiveChunker (`recursive`) | 13 | 129.0 | Rất tốt — Chia tự nhiên theo đoạn và đầu mục gạch đầu dòng của điều khoản. |
| `hcmut-lib-policy-adjustment.md` | FixedSizeChunker (`fixed_size`) | 12 | 199.1 | Kém — Bị ngắt ngang mức phạt 5.000đ và phí xử lý kỹ thuật. |
| `hcmut-lib-policy-adjustment.md` | SentenceChunker (`by_sentences`) | 4 | 458.2 | Khá — Khối văn bản tương đối lớn, chứa nhiều thông tin ngoài lề. |
| `hcmut-lib-policy-adjustment.md` | RecursiveChunker (`recursive`) | 15 | 121.3 | Rất tốt — Tách riêng từng phương án bồi thường một cách trọn vẹn. |

### Chiến lược của từng thành viên

**Thành viên 1 — Nguyễn Hoàng Việt**
- **Loại chiến lược:** FixedSizeChunker
- **Mô tả & lý do chọn cho chủ đề này:** Dùng phương pháp chia theo độ dài cố định `chunk_size=300`, `overlap=30`. Đây là đường cơ sở đơn giản nhất, tuy nhiên hay gặp lỗi ngắt đôi câu hoặc chia tách bảng số liệu về ngày mượn sách.

**Thành viên 2 — Đặng Hữu Tâm**
- **Loại chiến lược:** SentenceChunker
- **Mô tả & lý do chọn:** Chia nhỏ dựa trên ranh giới câu (`max_sentences_per_chunk=2`). Chiến lược này khắc phục nhược điểm của FixedSize vì không bao giờ cắt đôi câu, giúp câu văn trọn vẹn ngữ nghĩa ngữ pháp.

**Thành viên 3 — Phạm Quân**
- **Loại chiến lược:** Custom HeadingChunker
- **Mô tả & lý do chọn:** Chiến lược chia tách tùy biến theo các tiêu đề mục `#`, `##` của văn bản quy định học vụ. Mỗi điều khoản được giữ thành một khối độc lập kèm ngữ cảnh tiêu đề.
- **Code snippet:**
```python
class HeadingChunker:
    def chunk(self, text: str) -> list[str]:
        sections = re.split(r'(?=\n##?\s+)', text)
        return [s.strip() for s in sections if s.strip()]
```

**Thành viên 4 — Nguyễn Đỗ Chiến Thắng**
- **Loại chiến lược:** RecursiveChunker kết hợp Metadata Pre-filtering
- **Mô tả & lý do chọn:** Chia đệ quy theo thứ tự phân cấp `["\n\n", "\n", ". ", " ", ""]` với `chunk_size=400`. Khi truy xuất, kết hợp tiền lọc `metadata_filter={"audience": "student"}` để loại bỏ hoàn toàn các tài liệu không thuộc đối tượng sinh viên.

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Hoàng Việt | FixedSizeChunker | 6 / 10 | Tốc độ cắt nhanh, kích thước chunk đồng đều | Dễ cắt đứt câu và con số quan trọng ở ranh giới cắt |
| Đặng Hữu Tâm | SentenceChunker | 8 / 10 | Giữ trọn vẹn câu hoàn chỉnh | Độ dài chunk không đồng đều, thiếu liên kết giữa các điều khoản |
| Phạm Quân | Custom HeadingChunker | 10 / 10 | Giữ nguyên cấu trúc từng điều khoản văn bản | Một số điều quá dài vượt ngưỡng kích thước embedding tối ưu |
| Nguyễn Đỗ Chiến Thắng | RecursiveChunker + Metadata Filter | 10 / 10 | Đoạn trích cực kỳ súc tích, lọc sạch nhiễu nhờ metadata | Cần cấu trúc dữ liệu đầu vào có metadata chuẩn chỉnh |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Chiến lược **RecursiveChunker kết hợp Metadata Pre-filtering** của Nguyễn Đỗ Chiến Thắng và **Custom HeadingChunker** của Phạm Quân là hai chiến lược hiệu quả nhất. Đối với tài liệu văn bản quy phạm hành chính và quy định trường học, thông tin được tổ chức chặt chẽ theo từng chương/mục/điều khoản; việc chia nhỏ dựa trên cấu trúc tự nhiên giúp bảo toàn trọn vẹn logic và các con số định lượng (số ngày mượn, mức phạt), đồng thời cơ chế tiền lọc metadata giúp loại bỏ triệt để việc nhầm lẫn đối tượng quy chế.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Sinh viên hệ chính quy được mượn tối đa bao nhiêu cuốn sách và trong thời gian bao lâu? | Sinh viên đại học hệ chính quy được mượn tối đa 05 cuốn tài liệu trong cùng một thời điểm; thời hạn mượn là 21 ngày (3 tuần) cho sách kho mở từ C2 trở đi. | `hcmut-lib-borrowing-student` (Mục 2) |
| 2 | Mức phạt trễ hạn mượn sách mỗi ngày là bao nhiêu tiền và nếu làm mất sách thì xử lý như thế nào? | Phí phạt trễ hạn là 5.000 VNĐ / cuốn / ngày trễ. Nếu làm mất sách, bồi thường sách mới tương đương + 20.000 VNĐ phí xử lý, hoặc bồi thường 100% giá bìa + 100.000 VNĐ phí xử lý kỹ thuật. | `hcmut-lib-policy-adjustment` (Mục 1 & 2) |
| 3 | Điều kiện để tài khoản của sinh viên được kích hoạt quyền mượn tài liệu thư viện về nhà là gì? | Sinh viên phải xuất trình thẻ sinh viên hợp lệ và hoàn thành khóa học tập huấn thư viện đầu khóa với điểm bài kiểm tra trắc nghiệm từ 80% trở lên. | `hcmut-lib-borrowing-student` (Mục 4) & `hcmut-lib-information-training` (Mục 1) |
| 4 | Đặc quyền về số lượng sách và quyền truy cập cơ sở dữ liệu quốc tế của giảng viên và nghiên cứu sinh là gì? | Giảng viên được mượn 05 cuốn tiêu chuẩn (tối đa 10 cuốn khi có đề tài nghiên cứu); được cấp quyền truy cập từ xa vào các CSDL quốc tế (IEEE Xplore, ScienceDirect, Scopus) và sử dụng phòng nghiên cứu riêng. | `hcmut-lib-borrowing-faculty` (Mục 2 & 3) |
| 5 | Quy định về việc sao chụp (photocopy) tài liệu trong thư viện cho phép tối đa bao nhiêu phần trăm cuốn sách? | Bạn đọc chỉ được phép sao chụp phục vụ cá nhân không quá 20% tổng số trang của một cuốn sách hoặc không quá 01 chương sách; nghiêm cấm sao chép nguyên cuốn giáo trình. | `hcmut-lib-document-delivery` (Mục 1) |

### Tổng hợp chất lượng truy xuất của nhóm

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Mượn sách sinh viên chính quy | RecursiveChunker / HeadingChunker | Có (Top-1, Score: 0.8583) | Đạt 2/2 điểm — Trả về chính xác số lượng 5 cuốn và 21 ngày. |
| 2 | Phạt trễ hạn và mất tài liệu | RecursiveChunker | Có (Top-1, Score: 0.8701) | Đạt 2/2 điểm — Nêu đủ 5.000đ/ngày và 2 phương án bồi thường. |
| 3 | Điều kiện mở quyền mượn sách | RecursiveChunker + Metadata Filter | Có (Top-1, Score: 0.8089) | Đạt 2/2 điểm — Bắt buộc dùng `audience='student'` để loại trừ quy định thẻ cán bộ. |
| 4 | Đặc quyền giảng viên & NCS | RecursiveChunker / SentenceChunker | Có (Top-1, Score: 0.7925) | Đạt 2/2 điểm — Truy xuất đúng quyền truy cập CSDL IEEE, Scopus. |
| 5 | Giới hạn sao chụp bản quyền | RecursiveChunker / HeadingChunker | Có (Top-1, Score: 0.8501) | Đạt 2/2 điểm — Trích đúng định mức 20% tổng số trang hoặc 1 chương sách. |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> Cơ chế lọc bằng metadata mang lại lợi thế vượt trội rõ rệt nhất ở **Câu hỏi số 3**: Khi hỏi về "điều kiện kích hoạt tài khoản mượn tài liệu", nếu không dùng bộ lọc `{"audience": "student"}`, hệ thống dễ lấy nhầm quy chế của cán bộ/giảng viên hoặc hướng dẫn làm thẻ chung. Nhờ bộ lọc metadata, không gian tìm kiếm được thu hẹp chính xác vào các tài liệu dành riêng cho sinh viên, đưa độ chính xác lên 100%.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> 1. *Tác động của chiến lược chia nhỏ*: Chia theo cấu trúc ngữ nghĩa (Recursive/Heading) bảo toàn ngữ cảnh và mối liên hệ số liệu vượt trội hơn hẳn so với chia theo kích thước cố định cơ học.
> 2. *Sức mạnh của Hybrid Retrieval (Vector + Metadata Filter)*: Trong các nghiệp vụ đại học có tính phân quyền nghiêm ngặt, kết hợp tìm kiếm ngữ nghĩa với lọc metadata đối tượng (`audience`) là chìa khóa để loại bỏ hoàn toàn câu trả lời sai đối tượng.
> 3. *Hạn chế của Mock Embeddings so với Dense Embeddings*: Băm chuỗi (MD5) không có khả năng hiểu ngữ nghĩa đồng nghĩa, trong khi Dense Embeddings (như Gemini) hiểu sâu sắc ngữ cảnh và hỗ trợ đa ngôn ngữ hoàn hảo.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng một bộ tài liệu và cùng câu hỏi, nhưng cách thiết kế chunking và cấu trúc metadata quyết định hơn 80% chất lượng của hệ thống RAG trước khi câu lệnh được gửi đến mô hình LLM. Nếu chunking làm mất hoặc chia cắt thông tin thì mô hình ngôn ngữ dù mạnh đến đâu cũng sẽ bị hallucinate (bịa đặt) câu trả lời.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Nhóm sẽ thiết kế thêm trường metadata `section_level` và trích xuất tự động bảng biểu (table extraction) sang định dạng Markdown chuẩn trước khi nạp vào vector store, giúp hệ thống truy xuất các bảng đối chiếu phức tạp một cách chính xác hơn nữa.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 |
| Thuyết trình (Demo) | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
