# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đặng Hữu Tâm
**Nhóm:** G41
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai văn bản có cosine cao khi vector embedding của chúng cùng hướng, tức là mang ý nghĩa hoặc chủ đề gần nhau, kể cả khi dùng từ khác nhau. Giá trị gần 1 là rất giống, gần 0 là không liên quan, âm là ngược hướng.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên được mượn tối đa 3 tài liệu về nhà.
- Câu B: Học viên chỉ có thể đem về nhà không quá ba cuốn sách cùng lúc.
- Tại sao tương đồng: Hai câu gần như không trùng từ ("tài liệu" và "cuốn sách", "tối đa" và "không quá") nhưng cùng nghĩa là giới hạn số lượng mượn về nhà. Embedding có ngữ nghĩa sẽ cho cosine cao, còn so khớp từ khóa thì không. Đây là nhận định về khái niệm, không phải số đo, vì `MockEmbedder` băm chuỗi ký tự nên không thể hiện được ngữ nghĩa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Sinh viên được mượn tối đa 3 tài liệu về nhà.
- Câu B: Phòng học nhóm ở tầng 3 có máy lạnh và bảng viết.
- Tại sao khác: Hai câu nói về hai chủ đề không liên quan (quy định mượn sách và trang thiết bị phòng), không chia sẻ ý nghĩa nên vector của chúng chỉ theo hai hướng xa nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài vector, trong khi ý nghĩa của embedding nằm ở hướng chứ không phải độ lớn. Khoảng cách Euclid bị lệch khi hai vector cùng hướng nhưng khác độ dài; với vector đã chuẩn hóa thì hai cách cho cùng thứ tự xếp hạng, nhưng cosine vẫn an toàn hơn khi vector chưa chuẩn hóa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Mỗi chunk mới tiến thêm `chunk_size - overlap = 500 - 50 = 450` ký tự, nên số chunk = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22,11).
> Đáp án: **23 chunk**. Mình đã kiểm lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a' * 10000)` và kết quả cũng là 23.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Số chunk tăng lên 25, vì ceil((10000 - 100) / (500 - 100)) = ceil(24,75) = 25 (kiểm bằng code cũng ra 25): mỗi bước tiến ngắn hơn nên cần nhiều chunk hơn để phủ hết văn bản, đổi lại tốn thêm chi phí lưu trữ và embedding. Overlap lớn hơn vẫn đáng dùng vì một ý nằm sát ranh giới giữa hai chunk sẽ có mặt trọn vẹn trong ít nhất một chunk thay vì bị cắt đôi, nên tăng cơ hội được truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Mình tách câu bằng `re.split(r"(?<=[.!?])\s+", text)`: lookbehind cắt *sau* dấu `.`, `!`, `?` khi có khoảng trắng hoặc xuống dòng theo sau, nên dấu câu vẫn nằm trong câu (dùng `[.!?]\s+` sẽ làm mất dấu và chunk thành câu cụt). Sau đó mình strip từng câu, bỏ câu rỗng và ghép mỗi nhóm `max_sentences_per_chunk` câu lại bằng dấu cách; text rỗng hoặc chỉ có khoảng trắng trả `[]`. Số thập phân như `1.000` không bị cắt vì sau dấu chấm không có khoảng trắng. Trường hợp mình chưa xử lý được: chữ viết tắt (`TS.`, `v.v.`) vẫn bị cắt sai, và trên markdown các dòng danh sách không kết thúc bằng dấu chấm bị gộp thành một "câu" rất dài (trên file `hcmut-lib-borrowing-student` của corpus hiện tại, `SentenceChunker` với 3 câu mỗi chunk cho 7 chunk mà chunk dài nhất tới 848 ký tự).

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán chạy theo hai chiều. Chiều xuống: thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`, mảnh nào vẫn dài hơn `chunk_size` thì gọi đệ quy `_split` với các separator còn lại. Chiều lên: các mảnh nhỏ liền kề được gom lại cho tới sát `chunk_size`, nếu thiếu bước này sẽ sinh ra hàng trăm chunk vụn vài ký tự. Có ba base case: text đã ngắn hơn `chunk_size` thì trả nguyên; hết separator (kể cả `separators=[]`) hoặc gặp separator rỗng thì cắt cứng theo `chunk_size`; separator không có trong text thì hạ xuống separator kế tiếp. Separator được giữ ở cuối mảnh phía trước để không mất dấu câu hay xuống dòng khi ghép lại. Mình kiểm chứng trên 5 file corpus với `chunk_size=500`: ghép các chunk lại không mất chữ nào, chunk dài nhất là 496 ký tự và không có chunk nào dưới 40 ký tự.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mình chỉ dùng kho lưu trữ in-memory (bỏ hẳn nhánh ChromaDB vì không test nào cần nó và có nguy cơ làm sập cả bộ test nếu máy có cài `chromadb`). `_make_record` chuẩn hóa mỗi `Document` thành một record gồm `id`, `content`, bản sao của `metadata` (luôn có khóa `doc_id`, mặc định lấy `doc.id`) và `embedding`. `search` embed câu hỏi rồi tính dot product với từng record: vì vector đã được chuẩn hóa nên dot product bằng đúng cosine similarity. Kết quả được sắp xếp giảm dần theo điểm, cắt lấy `top_k` và bỏ vector `embedding` đi để output gọn.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc metadata **trước** rồi mới search trên tập ứng viên đã lọc. Nếu lấy top-k rồi mới lọc, k vị trí có thể bị các tài liệu sai chiếm hết và kết quả cuối còn 0 dù store vẫn có tài liệu hợp lệ; mình đã kiểm chứng bằng cách để 20 tài liệu `faculty` và 1 tài liệu `student`, tìm `top_k=1` với filter `student` vẫn ra đúng tài liệu đó. `search` và `search_with_filter` cùng đi qua `_search_records` nên không thể lệch kết quả. `delete_document` giữ lại các record có `metadata['doc_id']` khác `doc_id` được yêu cầu, rồi trả `True` nếu số record giảm đi và `False` nếu không xóa được gì.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `answer` làm ba bước: truy xuất top-k chunk từ store, dựng prompt, rồi gọi `llm_fn`. Prompt đánh số từng chunk `[1]`, `[2]`... kèm tên nguồn (`doc_id` và `source_url` nếu có) trong phần NGỮ CẢNH, và yêu cầu model chỉ dùng ngữ cảnh đó, không suy đoán, nếu không có câu trả lời thì nói rõ là không tìm thấy, đồng thời ghi số thứ tự đoạn đã dùng để câu trả lời truy vết được về đúng chunk và đúng file. Khi store rỗng, `answer` trả một thông báo cố định và không gọi LLM. Khi chạy LLM thật mình thấy prompt đầu tiên chưa đủ: agent có lúc chỉ trả về "[1]", có lúc lấy nhầm số liệu của một dịch vụ khác (khi thử trên bộ dữ liệu thư viện HUIT ban đầu, agent lấy mức phạt của dịch vụ mượn liên thư viện thay vì của mượn sách). Nên mình bổ sung ba yêu cầu chung: trả lời bằng câu hoàn chỉnh (không chỉ trả số thứ tự), nêu đủ mọi mục nếu câu hỏi hỏi nhiều mục, và nêu từng đáp án kèm điều kiện áp dụng khi câu hỏi chưa nói rõ đối tượng hoặc dịch vụ.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0 -- d:\AI THỰC CHIẾN\K4-DAY07-DangHuuTam-2A202602940-main\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: d:\AI THỰC CHIẾN\K4-DAY07-DangHuuTam-2A202602940-main
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.12s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Điểm thực tế tính bằng `compute_similarity()` với embedder local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (hỗ trợ tiếng Việt). Dự đoán được ghi lại trước khi chạy, với quy ước "cao" là cosine ≥ 0.5 và "thấp" là cosine < 0.5. Các câu ví dụ lấy từ quy định của thư viện HUIT (bộ dữ liệu ban đầu của nhóm); phép đo cosine không phụ thuộc vào corpus hiện tại.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên được mượn 3 tài liệu trong 10 ngày. | Học viên được đem về nhà ba cuốn sách trong mười ngày. | cao | 0.815 | Đúng |
| 2 | Sinh viên được phép mượn tài liệu về nhà. | Sinh viên không được phép mượn tài liệu về nhà. | cao | 0.614 | Đúng |
| 3 | Giảng viên được mượn sách trong 180 ngày. | Sinh viên được mượn sách trong 10 ngày. | cao | 0.622 | Đúng |
| 4 | Thư viện có 4 tầng và wifi miễn phí. | Hôm nay trời nắng, nhiệt độ khoảng 35 độ. | thấp | 0.153 | Đúng |
| 5 | Thư viện mở cửa lúc 7 giờ sáng. | Phòng học nhóm ở tầng 3 có 4 phòng. | thấp | 0.147 | Đúng |

Kết quả: dự đoán đúng 5/5. Để đối chứng, cùng 5 cặp này với `MockEmbedder` cho 0.313, 0.045, 0.060, -0.105 và 0.232, tức là toàn bộ đều dưới 0.5 và không phân biệt được cặp "cao" với cặp "thấp" (cặp 5, vốn thuộc nhóm thấp, còn có điểm cao hơn cặp 2 và cặp 3).

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 2 và cặp 3: câu phủ định ("không được phép") và câu đổi đối tượng kèm con số (giảng viên 180 ngày so với sinh viên 10 ngày) vẫn có cosine khoảng 0.61–0.62, gần bằng nhau và thấp hơn cặp diễn đạt lại cùng nghĩa (0.815). Điều này cho thấy embedding biểu diễn chủ đề và ngữ cảnh chung của câu tốt hơn là các chi tiết quyết định đúng sai như phủ định, con số hay đối tượng áp dụng, nên retrieval chỉ dựa vào độ giống có thể lẫn tài liệu của sinh viên với tài liệu của giảng viên, nên với dữ liệu có nhiều đối tượng thì cần thêm cách khác ngoài độ giống (ví dụ lọc `audience`). Tuy vậy trên corpus hiện tại, lọc `audience` không đổi kết quả (mục 5). Cặp 5 cũng lệch kỳ vọng của mình: hai câu cùng nói về thư viện nhưng khác thông tin chỉ đạt 0.147, gần bằng cặp về thời tiết hoàn toàn không liên quan (0.153).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Cấu hình chạy:** corpus là 5 tài liệu của Thư viện Trung tâm ĐHQG-HCM (`vnulib.edu.vn`, crawl ngày 19/09/2026). Chiến lược riêng của mình là `hybrid` (lớp `ParentIndex` trong `bench.py`): chia mỗi mục thành đoạn nhỏ ≤ 300 ký tự có gắn "tiêu đề tài liệu > tiêu đề mục", viết lại mỗi hàng bảng thành một dòng "cột: giá trị", xếp hạng bằng embedding kết hợp BM25 (reciprocal rank fusion), rồi trả về cho agent nguyên mục cha chứa đoạn tốt nhất. Embedder local `paraphrase-multilingual-MiniLM-L12-v2`, `top_k=3`, 54 đoạn nhỏ thuộc 25 mục cha; agent gọi LLM thật `nvidia/nemotron-3-super-120b-a12b` qua NVIDIA API với chế độ suy nghĩ bật (`python bench.py --answers`); câu 3 chạy với `metadata_filter={"audience": "student"}`. Toàn bộ top-3 và câu trả lời nằm trong `ket_qua_benchmark.txt`.

**Kết quả:** điểm truy xuất theo nội dung chunk là **9/10** (Q1, Q2, Q4, Q5 được 2 điểm; Q3 được 1 điểm vì chunk chứa đáp án ở hạng 3). Chấm cả câu trả lời của agent theo `docs/SCORING.md` cũng được **9/10**, vì agent trả lời đúng cả 5 câu, Q3 mất 1 điểm chỉ vì đoạn đáp án không ở top-1. So với các chiến lược khác trên cùng 5 câu: `fixed_size` 6, `sentence` 5, `recursive` 5, `heading` 5, `small_to_big` (chưa có BM25) 7, `hybrid` 9. Trên 8 câu kiểm tra riêng (`python bench.py all --heldout`, tối đa 16): 10, 13, 11, 15, 16 và 16.

**Lưu ý:** (1) `hybrid` được thiết kế từ các lỗi quan sát trên bộ dữ liệu HUIT ban đầu chứ không phải từ dữ liệu này, nhưng corpus chỉ có 5 tài liệu nên dễ hơn và điểm cao chưa chứng tỏ hệ thống mạnh. (2) Lọc `audience=student` **không thay đổi kết quả** ở Q3: cả có lọc và không lọc đều đưa đoạn đáp án xuống hạng 3, vì corpus chỉ có một tài liệu `student` và trang nguồn gộp mọi nhóm đối tượng trong một bảng. (3) Bộ 5 câu này lấy từ nội dung thật của các trang: Q1 và Q2 cùng chủ đề với báo cáo nhóm, còn ba câu còn lại khác vì đáp án chuẩn trước đó (điều kiện tập huấn ≥ 80%, quyền giảng viên, giới hạn sao chụp 20%) không có trên trang nguồn nào. (4) Cột Score là cosine của đoạn nhỏ tốt nhất, không phải xác suất; ở chế độ hybrid thứ hạng do rank fusion quyết định nên điểm không luôn giảm dần.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên hệ chính quy được mượn tối đa bao nhiêu tài liệu và trong thời gian bao lâu? | `borrowing-student#p2`, mục "Đối tượng thuộc ĐHQG-HCM (hệ chính quy)": bảng số lượng, số ngày và gia hạn cho giảng viên, cán bộ, học viên sau đại học, nghiên cứu sinh và sinh viên (5 tài liệu, 21 ngày, gia hạn 1 lần 21 ngày) | 0.806 | Có, đủ đáp án | Tối đa 5 tài liệu, thời gian mượn 21 ngày [1]. Đúng |
| 2 | Trả tài liệu trễ hạn thì bị phạt bao nhiêu tiền một ngày? | `borrowing-student#p9`, mục "Phạt và bồi thường hư hỏng tài liệu": phạt trễ hạn 5.000đ/tài liệu/ngày | 0.623 | Có, đủ đáp án | Phạt 5.000đ cho mỗi tài liệu mỗi ngày [1]. Đúng |
| 3 | Có phải đặt tiền cọc khi mượn sách không? (lọc `audience=student`) | `borrowing-student#p6`, mục "Phí mượn": phí 1.000đ/cuốn/ngày và các lưu ý về biên lai và khấu trừ tiền cọc. Đoạn đáp án (`#p2`, dòng "(*) không phải đặt tiền cọc") ở hạng 3 | 0.708 | Một phần: đáp án ở hạng 3 | Phụ thuộc đối tượng: hệ chính quy thuộc ĐHQG-HCM (gồm sinh viên) không phải đặt cọc và phí mượn, đối tượng khác phải đặt cọc theo bậc giá bìa [2][3]. Đúng và có căn cứ |
| 4 | Làm mất tài liệu thì phải đền như thế nào? | `borrowing-student#p9`, mục "Phạt và bồi thường hư hỏng tài liệu" cùng bảng mức bồi thường | 0.455 | Có, đủ đáp án | Hai cách: đền bằng tiền (giá bìa x 3 + 50.000đ/tài liệu + phí trễ hạn nếu có) hoặc đền bằng tài liệu như bản gốc/mới hơn + 50.000đ/tài liệu + phí trễ hạn [1]. Đúng và đủ |
| 5 | Photocopy tài liệu ở thư viện tính phí bao nhiêu một trang? | `document-delivery#p1`, mục "Nội dung giao dịch và phí dịch vụ" | 0.668 | Có, đủ đáp án | Photocopy 250đ mỗi trang A4 [1]. Đúng |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (Q1, Q2, Q4, Q5 ở hạng 1; Q3 ở hạng 3)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | / 5 |
| Hướng tiếp cận của tôi (My Approach) | / 10 |
| Hoàn thiện code (Core Implementation — tests) | / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | / 5 |
| Kết quả truy xuất của tôi (Competition Results) | / 10 |
| **Tổng phần cá nhân** | **/ 60** |
