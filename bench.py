"""Benchmark 5 câu hỏi trên corpus Thư viện Trung tâm ĐHQG-HCM (data/thu-vien-vnulib/).

Chạy:
    python bench.py              # chiến lược đặt ở dòng STRATEGY bên dưới
    python bench.py heading      # chọn nhanh một chiến lược khác
    python bench.py small_to_big # chiến lược tự thiết kế: đoạn nhỏ có tiêu đề, trả về nguyên mục cha
    python bench.py hybrid       # như small_to_big, xếp hạng bằng embedding + từ khóa BM25
    python bench.py all          # chạy tất cả chiến lược để so sánh
    python bench.py all --heldout  # 8 câu hỏi kiểm tra riêng, để xem cải tiến có tổng quát hóa không
    python bench.py --answers    # thêm câu trả lời của KnowledgeBaseAgent bằng LLM thật (cần NVIDIA_API_KEY)

Mỗi người trong nhóm chỉ đổi DÒNG `STRATEGY = ...`; mọi thứ khác giữ nguyên để so sánh công bằng.
Kết quả được in ra màn hình và ghi vào ket_qua_benchmark.txt.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

DATA_DIR = Path("data/thu-vien-vnulib")
RESULT_FILE = Path("ket_qua_benchmark.txt")
TOP_K = 3


class HeadingChunker:
    """Chia theo tiêu đề markdown (`##`, `###`): mỗi mục là một chunk.

    Lý do thiết kế: quy định và FAQ được biên soạn theo mục ("## 7. Quy định đền bù...",
    "## 3. Để có Thẻ thư viện...") nên mỗi mục đã là một đơn vị ngữ nghĩa trọn vẹn.
    Mục nào dài hơn `chunk_size` thì hạ xuống RecursiveChunker và gắn lại tiêu đề vào từng
    mảnh con, để mảnh thứ hai trở đi vẫn biết nó thuộc mục nào.
    """

    def __init__(self, chunk_size: int = 800) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        carried = ""  # a title-only section (e.g. the "# Title" line) is joined to the next one
        for section in re.split(r"(?m)^(?=#{2,3} )", text):
            section = section.strip()
            if not section:
                continue
            heading, _, body = section.partition("\n")
            heading = heading.strip() if heading.startswith("#") else ""
            if heading and not body.strip():
                carried = f"{carried}\n\n{section}".strip()
                continue
            if carried:
                section, carried = f"{carried}\n\n{section}", ""
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue
            body = section[len(heading):].strip() if heading else section
            room = max(50, self.chunk_size - len(heading) - 2)
            for piece in RecursiveChunker(chunk_size=room).chunk(body):
                chunks.append(f"{heading}\n\n{piece}" if heading else piece)
        if carried:
            chunks.append(carried)
        return chunks


STRATEGIES = {
    "fixed_size": FixedSizeChunker(chunk_size=500, overlap=50),
    "sentence": SentenceChunker(max_sentences_per_chunk=3),
    "recursive": RecursiveChunker(chunk_size=500),
    "heading": HeadingChunker(chunk_size=800),
}

STRATEGY = "hybrid"  # <- mỗi người chỉ đổi đúng dòng này


@dataclass
class Query:
    text: str
    gold_docs: tuple[str, ...]  # tài liệu chứa đáp án
    # Answer is complete when one chunk holds EVERY entry. An entry may be a tuple of alternatives (any of them
    # counts), because a table row reads differently as raw markdown and as "column: value" text.
    must_contain: tuple[str | tuple[str, ...], ...]
    gold_answer: str
    metadata_filter: dict | None = None


QUERIES = [
    Query(
        text="Sinh viên hệ chính quy được mượn tối đa bao nhiêu tài liệu và trong thời gian bao lâu?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=(("Sinh viên | 5 | 21", "Đối tượng: Sinh viên; Số lượng tài liệu: 5; Số ngày: 21"),),
        gold_answer="Sinh viên được mượn tối đa 5 tài liệu trong 21 ngày, gia hạn 1 lần 21 ngày.",
    ),
    Query(
        text="Trả tài liệu trễ hạn thì bị phạt bao nhiêu tiền một ngày?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=("Mức phạt: 5.000đ/tài liệu/ngày",),
        gold_answer="Phạt trễ hạn 5.000đ/tài liệu/ngày (khác với phí mượn 1.000đ/cuốn/ngày của độc giả phải đóng phí).",
    ),
    Query(
        # Does not say who is asking. The notice on deposits applies to outside readers, second-degree students and
        # visiting lecturers, so an unfiltered search can answer a regular student with the wrong rule.
        text="Có phải đặt tiền cọc khi mượn sách không?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=("(*) Đối tượng không phải đặt tiền cọc và phí mượn sách",),
        gold_answer="Sinh viên hệ chính quy thuộc ĐHQG-HCM không phải đặt tiền cọc và phí mượn sách (chỉ hệ không chính quy và độc giả ngoài ĐHQG-HCM mới phải đặt cọc).",
        metadata_filter={"audience": "student"},
    ),
    Query(
        text="Làm mất tài liệu thì phải đền như thế nào?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=("Giá bìa x 3 + 50.000đ/tài liệu", "Như tài liệu gốc/năm xuất bản mới hơn + 50.000đ/tài liệu"),
        gold_answer="Đền bằng tiền: giá bìa x 3 + 50.000đ/tài liệu + phí phạt trễ hạn (nếu có); hoặc đền bằng tài liệu như bản gốc hoặc năm xuất bản mới hơn + 50.000đ/tài liệu + phí phạt trễ hạn (nếu có).",
    ),
    Query(
        text="Photocopy tài liệu ở thư viện tính phí bao nhiêu một trang?",
        gold_docs=("hcmut-lib-document-delivery",),
        must_contain=("Photocopy 250đ/trang A4",),
        gold_answer="Photocopy 250đ/trang A4 (scan và in là 1.000đ/trang A4).",
    ),
]


# Extra queries written from the corpus BEFORE the small-to-big fixes were tried and never used to
# diagnose or tune them. They only check that an improvement measured on QUERIES is not memorisation.
# Answer strings are chosen to appear in both the raw table and the "column: value" form.
HELDOUT_QUERIES = [
    Query(
        text="Độc giả ngoài ĐHQG-HCM được mượn tối đa mấy tài liệu và trong bao nhiêu ngày?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=(("Độc giả ngoài ĐQHG-HCM | 2 | 30", "Đối tượng: Độc giả ngoài ĐQHG-HCM; Số lượng tài liệu: 2; Số ngày: 30"),),
        gold_answer="Tối đa 2 tài liệu trong 30 ngày, không được gia hạn.",
    ),
    Query(
        text="Phí mượn sách mỗi ngày là bao nhiêu?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=("1.000đ /1 cuốn /1 ngày",),
        gold_answer="Phí mượn 1.000đ/cuốn/ngày (áp dụng cho đối tượng phải đóng phí mượn).",
    ),
    Query(
        text="Mượn quá hạn 30 ngày thì tiền cọc bị xử lý thế nào?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=("khấu trừ vào tiền cọc 10%/ngày",),
        gold_answer="Bị khấu trừ vào tiền cọc 10%/ngày và phải thanh toán phí mượn trong thời gian lưu giữ sách.",
    ),
    Query(
        text="Lớp tập huấn sử dụng thư viện kéo dài bao lâu?",
        gold_docs=("hcmut-lib-information-training",),
        must_contain=("Lớp tập huấn sử dụng thư viện (30 phút)",),
        gold_answer="Lớp tập huấn sử dụng thư viện kéo dài 30 phút.",
    ),
    Query(
        text="Sách có giá bìa 150.000 đồng thì tiền thế chân là bao nhiêu?",
        gold_docs=("hcmut-lib-borrowing-student", "hcmut-lib-policy-adjustment"),
        must_contain=(("Từ 100.000 - dưới 200.000 | 1.500.000", "Giá bìa sách (VNĐ): Từ 100.000 - dưới 200.000; Số tiền thế chân (VNĐ): 1.500.000"),),
        gold_answer="Giá bìa từ 100.000 đến dưới 200.000 đồng thì tiền thế chân là 1.500.000 đồng.",
    ),
    Query(
        text="Nếu không trả tài liệu quá hạn 3 tháng thì tiền thế chân xử lý thế nào?",
        gold_docs=("hcmut-lib-policy-adjustment",),
        must_contain=("tiền thế chân sẽ được TVTT sử dụng để mua lại tài liệu",),
        gold_answer="Tiền thế chân được thư viện dùng để mua lại tài liệu đã thất thoát và không hoàn trả cho độc giả.",
    ),
    Query(
        text="Phí scan tài liệu là bao nhiêu một trang?",
        gold_docs=("hcmut-lib-document-delivery",),
        must_contain=("Scan (số hóa) 1.000đ/trang A4",),
        gold_answer="Scan (số hóa) 1.000đ/trang A4.",
    ),
    Query(
        text="Những loại tài liệu nào không được mượn về nhà?",
        gold_docs=("hcmut-lib-borrowing-student",),
        must_contain=("Tài liệu tra cứu như từ điển, bách khoa toàn thư",),
        gold_answer="Tài liệu tra cứu (từ điển, bách khoa toàn thư, cẩm nang) có đóng dấu Không mượn về, tài liệu phòng Tham khảo, phòng Báo - Tạp chí, băng đĩa và tài liệu lưu chiểu.",
    ),
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def has_key(text: str, key) -> bool:
    alternatives = (key,) if isinstance(key, str) else tuple(key)
    return any(norm(alternative) in norm(text) for alternative in alternatives)


def load_corpus() -> list[tuple[str, dict, str]]:
    corpus = []
    for path in sorted(DATA_DIR.glob("*.md")):
        head, body = path.read_text(encoding="utf-8").split("\n---\n", 1)
        metadata = {key: value.strip().strip('"') for key, value in re.findall(r"^(\w+):\s*(.+)$", head, re.M)}
        corpus.append((path.stem, metadata, body.strip()))
    return corpus


def build_documents(chunker, corpus: list[tuple[str, dict, str]]) -> list[Document]:
    """Chunk OUTSIDE the store: one Document per chunk, frontmatter spread into every chunk."""
    documents = []
    for doc_id, metadata, body in corpus:
        for index, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{doc_id}#{index}",
                    content=chunk,
                    metadata={**metadata, "doc_id": doc_id, "chunk_index": index},
                )
            )
    return documents


def build_embedder():
    load_dotenv(override=False)
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    try:
        if provider == "local":
            from src import LocalEmbedder

            return LocalEmbedder()
        if provider == "openai":
            from src import OpenAIEmbedder

            return OpenAIEmbedder()
        if provider == "gemini":
            from src import GeminiEmbedder

            return GeminiEmbedder()
    except Exception as error:  # missing package or key: fall back instead of crashing
        print(f"CẢNH BÁO: không nạp được embedder '{provider}' ({error}).")
    if provider != "mock":
        print("CẢNH BÁO: đang dùng MockEmbedder, điểm số bị chi phối bởi mock, không phản ánh ngữ nghĩa.")
    return _mock_embed


NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def build_llm():
    """LLM for --answers. The key is read from NVIDIA_API_KEY (env var or .env), never hard-coded."""
    load_dotenv(override=False)
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None
    model = os.getenv("NVIDIA_MODEL", NVIDIA_MODEL)

    def llm(prompt: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 3000,
                # Thinking on: measured over 5 repeats, Q2 was right 5/5 with it and only 3/5 without.
                # The reasoning goes to `reasoning_content`; `content` is just the final answer.
                "chat_template_kwargs": {"enable_thinking": True},
            }
        ).encode()
        request = urllib.request.Request(
            NVIDIA_URL, data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        for attempt in range(3):  # the endpoint sometimes resets the connection
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    content = (json.load(response)["choices"][0]["message"].get("content") or "").strip()
                if not content:  # e.g. the token budget was used up by the reasoning
                    raise RuntimeError("model returned an empty answer")
                return content
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"HTTP {error.code}") from error
                time.sleep(3 * (attempt + 1))  # transient server error: wait and retry
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if attempt == 2:
                    raise RuntimeError(str(error)) from error
                time.sleep(2 * (attempt + 1))

    llm.model_name = model
    return llm


class _FilteredView:
    """Lets KnowledgeBaseAgent (which calls store.search) retrieve with a metadata filter."""

    def __init__(self, index, metadata_filter: dict | None) -> None:
        self._index = index
        self._metadata_filter = metadata_filter

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return self._index.search(query, top_k=top_k, metadata_filter=self._metadata_filter)


def linearize_tables(text: str) -> str:
    """Rewrite each markdown table row as one "column: value; ..." line so a row stands alone.

    A table read row by row loses its header, and its later rows fall outside the embedder's
    128-token window; a self-contained row keeps both the column names and the values.
    """
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip()):
            headers = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                out.append("- " + "; ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c))
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (heading, text) sections at `##`/`###` lines.

    The H1 line is dropped (the title is in the metadata); text before the first heading
    becomes a section with an empty heading.
    """
    sections = []
    for part in re.split(r"(?m)^(?=#{2,3} )", body):
        part = part.strip()
        if part.startswith("# "):
            part = part.split("\n", 1)[1].strip() if "\n" in part else ""
        if not part:
            continue
        first_line = part.split("\n", 1)[0]
        sections.append((first_line.lstrip("#").strip() if first_line.startswith("#") else "", part))
    return sections


def _tokens(text: str) -> list[str]:
    """Lower-case syllables plus adjacent pairs; Vietnamese words are made of 1-3 syllables."""
    words = re.findall(r"\w+", text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class BM25:
    """Minimal Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.term_counts = [Counter(_tokens(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.term_counts]
        self.avg_length = sum(self.lengths) / len(documents)
        document_frequency = Counter(term for c in self.term_counts for term in c)
        n = len(documents)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()}

    def score(self, query: str, index: int) -> float:
        counts, length = self.term_counts[index], self.lengths[index]
        total = 0.0
        for term in set(_tokens(query)):
            frequency = counts.get(term, 0)
            if frequency:
                norm_length = self.k1 * (1 - self.b + self.b * length / self.avg_length)
                total += self.idf[term] * frequency * (self.k1 + 1) / (frequency + norm_length)
        return total


class FlatIndex:
    """One vector per chunk; the agent reads back exactly the chunk that was embedded."""

    def __init__(self, name: str, chunker, corpus, embedder) -> None:
        self.documents = build_documents(chunker, corpus)
        self.store = EmbeddingStore(collection_name=f"bench_{name}", embedding_fn=embedder)
        self.store.add_documents(self.documents)
        self.size = len(self.documents)

    def search(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict]:
        return self.store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)

    def describe(self) -> str:
        lengths = [len(d.content) for d in self.documents]
        return f"{self.size} chunk (dài trung bình {sum(lengths) / len(lengths):.0f}, lớn nhất {max(lengths)} ký tự)"


class ParentIndex:
    """Small-to-big retrieval, designed for the failures found on Q3, Q4 and Q5.

    * Small pieces (<= CHILD_SIZE chars) are embedded, each prefixed with the document title and
      section heading: a step list no longer loses its topic when it is cut (Q3), and a piece
      fits the embedder's 128-token window (Q4).
    * Table rows are rewritten as "column: value" lines before chunking (Q4).
    * The agent is handed the whole parent section (<= PARENT_MAX chars) of the best pieces, so
      a list or table is never cut in the middle (Q5).
    * hybrid=True ranks the pieces by reciprocal rank fusion of the embedding order and BM25, which
      rescues a specific document that a broader one out-scores on meaning alone (Q4). The printed
      score stays the cosine of the best piece, so in hybrid mode it is not monotonic with the rank.
    """

    CHILD_SIZE = 300
    PARENT_MAX = 1200

    def __init__(self, corpus, embedder, hybrid: bool = False) -> None:
        self.hybrid = hybrid
        self.store = EmbeddingStore(collection_name="bench_small_to_big", embedding_fn=embedder)
        self.parents: dict[str, dict] = {}
        children: list[Document] = []
        splitter = RecursiveChunker(chunk_size=self.CHILD_SIZE)
        for doc_id, metadata, body in corpus:
            index = 0
            for heading, section in split_sections(linearize_tables(body)):
                for text in self._parent_texts(heading, section):
                    parent_id = f"{doc_id}#p{index}"
                    index += 1
                    self.parents[parent_id] = {
                        "content": text,
                        "metadata": {**metadata, "doc_id": doc_id, "chunk_index": index - 1},
                    }
                    header = f"{metadata['title']} > {heading}" if heading else metadata["title"]
                    inner = text.split("\n", 1)[1].strip() if heading and "\n" in text else ("" if heading else text)
                    for number, piece in enumerate(splitter.chunk(inner) or [""]):
                        children.append(
                            Document(
                                id=f"{parent_id}.{number}",
                                content=f"{header}\n{piece}".strip(),
                                metadata={**metadata, "doc_id": doc_id, "parent_id": parent_id},
                            )
                        )
        self.children = children
        self.store.add_documents(children)
        self.size = len(children)
        self._bm25 = BM25([d.content for d in children]) if hybrid else None
        self._child_position = {d.id: i for i, d in enumerate(children)}

    def _parent_texts(self, heading: str, section: str) -> list[str]:
        if len(section) <= self.PARENT_MAX:
            return [section]
        prefix = f"## {heading}\n\n" if heading else ""
        inner = section.split("\n", 1)[1].strip() if heading else section
        return [prefix + piece for piece in RecursiveChunker(chunk_size=self.PARENT_MAX - len(prefix)).chunk(inner)]

    def search(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict]:
        hits = self.store.search_with_filter(query, top_k=len(self.children), metadata_filter=metadata_filter)
        if self.hybrid:
            hits = self._fuse(query, hits)
        results, seen = [], set()
        for hit in hits:  # best piece first; several pieces of one section count once
            parent_id = hit["metadata"]["parent_id"]
            if parent_id in seen:
                continue
            seen.add(parent_id)
            parent = self.parents[parent_id]
            results.append(
                {"id": parent_id, "content": parent["content"], "metadata": parent["metadata"], "score": hit["score"]}
            )
            if len(results) == top_k:
                break
        return results

    def _fuse(self, query: str, hits: list[dict], k: int = 60) -> list[dict]:
        """Reciprocal rank fusion of the embedding order and the BM25 keyword order."""
        by_keyword = sorted(hits, key=lambda h: self._bm25.score(query, self._child_position[h["id"]]), reverse=True)
        fused: dict[str, float] = {}
        for ranking in (hits, by_keyword):
            for rank, hit in enumerate(ranking, start=1):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (k + rank)
        return sorted(hits, key=lambda h: fused[h["id"]], reverse=True)

    def describe(self) -> str:
        small = [len(d.content) for d in self.children]
        big = [len(p["content"]) for p in self.parents.values()]
        return (
            f"{self.size} đoạn nhỏ (dài trung bình {sum(small) / len(small):.0f}, lớn nhất {max(small)}) "
            f"-> {len(self.parents)} mục cha trả về cho agent (trung bình {sum(big) / len(big):.0f}, lớn nhất {max(big)} ký tự)"
        )


ALL_STRATEGIES = [*STRATEGIES, "small_to_big", "hybrid"]


def build_index(name: str, corpus, embedder):
    if name in ("small_to_big", "hybrid"):
        return ParentIndex(corpus, embedder, hybrid=name == "hybrid")
    return FlatIndex(name, STRATEGIES[name], corpus, embedder)


def grade(query: Query, results: list[dict]) -> tuple[int, int | None, bool, int]:
    """Return (score, rank of the best gold chunk, naive doc-level hit, ideas covered).

    Content-level, following docs/SCORING.md: 2 if the top-1 chunk is from a gold doc and holds
    the WHOLE answer, 1 if a gold chunk in the top-3 holds only part of it or ranks 2-3, 0 if
    no gold chunk in the top-3 holds any of it. The naive doc-level hit only checks the doc_id,
    which inflates results: a gold doc can fill the top-3 without any chunk answering.
    """
    doc_hit = any(r["metadata"]["doc_id"] in query.gold_docs for r in results)
    needed = len(query.must_contain)
    best = None  # (ideas covered, rank) of the gold chunk that covers the most required strings
    for rank, result in enumerate(results, start=1):
        if result["metadata"]["doc_id"] not in query.gold_docs:
            continue
        found = sum(has_key(result["content"], key) for key in query.must_contain)
        if found and (best is None or found > best[0]):
            best = (found, rank)
    if best is None:
        return 0, None, doc_hit, 0
    found, rank = best
    return (2 if found == needed and rank == 1 else 1), rank, doc_hit, found


def run_strategy(name: str, corpus, embedder, out, llm=None, queries=None) -> dict:
    queries = queries or QUERIES
    index = build_index(name, corpus, embedder)
    out(f"\n{'=' * 78}\nChiến lược: {name}  |  đã nạp {index.describe()}\n{'=' * 78}")

    total, naive_hits = 0, 0
    for number, query in enumerate(queries, start=1):
        results = index.search(query.text, top_k=TOP_K, metadata_filter=query.metadata_filter)
        score, rank, doc_hit, found = grade(query, results)
        total += score
        naive_hits += doc_hit
        label = f"lọc {query.metadata_filter}" if query.metadata_filter else "không lọc"
        out(f"\n[Q{number}] ({label}) {query.text}")
        out(f"  gold: {', '.join(query.gold_docs)} | dấu hiệu đáp án: {query.must_contain!r}")
        for position, result in enumerate(results, start=1):
            preview = norm(result["content"])[:95]
            out(f"  #{position} score={result['score']:.3f} {result['id']:<44} {preview}")
        out(f"  -> điểm {score}/2 (chunk gold tốt nhất ở hạng {rank}, chứa {found}/{len(query.must_contain)} ý của đáp án; gold có trong top-{TOP_K}: {doc_hit})")

        if llm:
            agent = KnowledgeBaseAgent(_FilteredView(index, query.metadata_filter), llm)
            try:
                answer = agent.answer(query.text, top_k=TOP_K)
            except RuntimeError as error:
                answer = f"[không gọi được LLM: {error}]"
            out("  Agent (LLM): " + answer.replace("\n", "\n    "))
            out(f"  Gold answer: {query.gold_answer}")

        if query.metadata_filter:  # A/B: the same query without the filter
            unfiltered = index.search(query.text, top_k=TOP_K, metadata_filter=None)
            a_score, a_rank, _, _ = grade(query, unfiltered)
            out(f"  [A/B không lọc] điểm {a_score}/2 (hạng có đáp án: {a_rank})")
            for position, result in enumerate(unfiltered, start=1):
                out(f"    #{position} score={result['score']:.3f} {result['id']:<44} {norm(result['content'])[:75]}")

    out(f"\nTỔNG {name}: {total}/{2 * len(queries)} (chấm theo nội dung) | doc_id đúng trong top-{TOP_K}: {naive_hits}/{len(queries)} (chấm ngây thơ)")
    return {"strategy": name, "chunks": index.size, "score": total, "naive": naive_hits}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    choice = positional[0] if positional else STRATEGY
    names = ALL_STRATEGIES if choice == "all" else [choice]
    if any(name not in ALL_STRATEGIES for name in names):
        print(f"Chiến lược không hợp lệ: {choice}. Chọn một trong {ALL_STRATEGIES} hoặc 'all'.")
        return 2

    lines: list[str] = []

    def out(line: str = "") -> None:
        print(line)
        lines.append(line)

    heldout = "--heldout" in sys.argv
    queries = HELDOUT_QUERIES if heldout else QUERIES
    corpus = load_corpus()
    for query in queries:  # every gold answer must be quotable from the real documents
        text = " ".join(body for doc_id, _, body in corpus if doc_id in query.gold_docs)
        missing = [key for key in query.must_contain if not has_key(text, key)]
        if missing:
            out(f"LỖI GOLD: không tìm thấy {missing!r} trong {query.gold_docs}")
            return 1

    llm = None
    if "--answers" in sys.argv:
        llm = build_llm()
        if llm is None:
            print("Thiếu NVIDIA_API_KEY: đặt biến môi trường hoặc thêm vào file .env (đã có trong .gitignore).")
            return 2

    embedder = build_embedder()
    out(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    if llm:
        out(f"LLM cho agent: {llm.model_name}")
    out(f"Corpus: {len(corpus)} tài liệu trong {DATA_DIR}")

    summary = [run_strategy(name, corpus, embedder, out, llm, queries) for name in names]
    if len(summary) > 1:
        out(f"\n{'=' * 78}\nTỔNG HỢP\n{'=' * 78}")
        out(f"{'chiến lược':<13} {'số chunk':>9} {'điểm nội dung':>19} {'doc_id đúng':>16}")
        for row in summary:
            out(f"{row['strategy']:<13} {row['chunks']:>9} {row['score']:>19} {row['naive']:>16}")

    if heldout:  # never overwrite the deliverable with the extra check
        return 0
    RESULT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
