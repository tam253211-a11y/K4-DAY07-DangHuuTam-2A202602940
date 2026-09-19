from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    NO_CONTEXT_MESSAGE = "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            # Empty store: nothing to ground an answer on, so skip the LLM call.
            return self.NO_CONTEXT_MESSAGE
        return self.llm_fn(self._build_prompt(question, chunks))

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        # Number every chunk and name its source so the answer can cite "[1]" and be
        # traced back to the exact chunk and file.
        context_blocks = []
        for number, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("doc_id") or metadata.get("source") or chunk.get("id")
            if metadata.get("source_url"):
                source = f"{source} - {metadata['source_url']}"
            context_blocks.append(f"[{number}] (nguồn: {source})\n{chunk['content']}")
        context = "\n\n".join(context_blocks)

        return (
            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu.\n"
            "Chỉ dùng thông tin trong phần NGỮ CẢNH bên dưới, không dùng kiến thức bên ngoài "
            "và không suy đoán. Nếu ngữ cảnh không chứa câu trả lời, hãy nói rõ là không tìm "
            "thấy thông tin trong tài liệu.\n"
            "Các đoạn có thể nói về những chủ đề hoặc dịch vụ khác nhau: chỉ dùng đoạn thực sự "
            "trả lời đúng câu hỏi và không lấy số liệu của một chủ đề khác. Nếu câu hỏi hỏi về "
            "nhiều mục (các loại, các bước, các điều kiện), hãy nêu đủ tất cả các mục có trong ngữ cảnh.\n"
            "Nếu câu hỏi chưa nói rõ đối tượng hoặc dịch vụ mà ngữ cảnh có nhiều đáp án khác nhau, hãy nêu "
            "từng đáp án kèm điều kiện áp dụng của nó (loại tài liệu, dịch vụ hoặc đối tượng), không chọn "
            "một đáp án duy nhất.\n"
            "Trả lời bằng tiếng Việt, bằng câu văn hoàn chỉnh, sau đó ghi số thứ tự đoạn đã dùng "
            "ở cuối, ví dụ [1], để có thể kiểm chứng nguồn. Không được chỉ trả về số thứ tự.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )
