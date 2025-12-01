"""
RAG Module - Retrieval-Augmented Generation
Kết hợp retrieval với LLM để trả lời câu hỏi dựa trên audio transcripts
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
from openai import OpenAI
from datetime import datetime
import json
import os


class RAGSystem:
    """
    Hệ thống RAG kết hợp retrieval và LLM
    """

    def __init__(
        self,
        vector_db,
        embedder,
        llm_model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_k: int = 5
    ):
        """
        Khởi tạo RAG System

        Args:
            vector_db: VectorDatabase instance
            embedder: TextEmbedding instance
            llm_model: Tên model LLM
            api_key: OpenAI API key
            temperature: Temperature cho LLM
            max_tokens: Max tokens cho response
            top_k: Số lượng chunks retrieve
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Setup OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            print("WARNING: OpenAI API key chưa được cấu hình!")

        self.prompt_template = """Bạn là một trợ lý AI thông minh. Dựa trên các đoạn văn bản sau đây được trích xuất từ âm thanh, hãy trả lời câu hỏi của người dùng một cách chính xác và chi tiết.

Các đoạn văn bản liên quan:
{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên thông tin được cung cấp. Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó. Nếu có thông tin về thời gian (timestamp) trong audio, hãy đề cập đến nó.

Trả lời:"""

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Thực hiện query với RAG pipeline

        Args:
            question: Câu hỏi của người dùng
            top_k: Số lượng chunks retrieve (override default)
            return_sources: Có trả về sources không
            filter_dict: Bộ lọc cho retrieval

        Returns:
            Dict chứa answer, sources và metadata
        """
        k = top_k or self.top_k

        # Step 1: Tạo embedding cho query
        query_embedding = self.embedder.encode_text(question, show_progress=False)

        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=k,
            filter_dict=filter_dict
        )

        if not retrieved_chunks:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này.",
                "sources": [],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }

        # Step 3: Tạo context từ retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Step 4: Tạo prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        # Step 5: Gọi LLM
        try:
            answer = self._call_llm(prompt)
        except Exception as e:
            print(f"Lỗi khi gọi LLM: {str(e)}")
            answer = f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

        # Step 6: Format response
        response = {
            "answer": answer,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "num_sources": len(retrieved_chunks)
        }

        if return_sources:
            response["sources"] = self._format_sources(retrieved_chunks)

        return response

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Format các chunks thành context cho LLM

        Args:
            chunks: List các retrieved chunks

        Returns:
            Context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")

            # Thêm timestamp info nếu có
            metadata = chunk.get("metadata", {})
            if "start_time" in metadata and "end_time" in metadata:
                time_info = f"[Thời gian: {self._format_timestamp(metadata['start_time'])} - {self._format_timestamp(metadata['end_time'])}]"
                context_parts.append(f"{i}. {time_info}\n{text}\n")
            else:
                context_parts.append(f"{i}. {text}\n")

        return "\n".join(context_parts)

    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Format sources để trả về cho user

        Args:
            chunks: List các retrieved chunks

        Returns:
            List sources đã format
        """
        sources = []

        for chunk in chunks:
            source = {
                "text": chunk.get("text", ""),
                "similarity": chunk.get("similarity", 0.0)
            }

            # Thêm metadata
            metadata = chunk.get("metadata", {})
            if metadata:
                source["audio_file"] = metadata.get("audio_file", "")
                source["start_time"] = metadata.get("start_time")
                source["end_time"] = metadata.get("end_time")

                # Format timestamps
                if source["start_time"] is not None:
                    source["start_time_formatted"] = self._format_timestamp(source["start_time"])
                if source["end_time"] is not None:
                    source["end_time_formatted"] = self._format_timestamp(source["end_time"])

            sources.append(source)

        return sources

    def _call_llm(self, prompt: str) -> str:
        """
        Gọi LLM để sinh câu trả lời

        Args:
            prompt: Prompt cho LLM

        Returns:
            Câu trả lời từ LLM
        """
        if not self.client:
            raise ValueError("OpenAI client chưa được khởi tạo. Vui lòng cung cấp API key.")

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý AI thông minh và hữu ích."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        answer = response.choices[0].message.content.strip()
        return answer

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp từ giây sang HH:MM:SS

        Args:
            seconds: Số giây

        Returns:
            Timestamp formatted
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    def batch_query(
        self,
        questions: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Xử lý nhiều queries cùng lúc

        Args:
            questions: List các câu hỏi
            top_k: Số lượng chunks retrieve

        Returns:
            List các responses
        """
        responses = []

        for question in questions:
            print(f"\nProcessing: {question}")
            response = self.query(question, top_k=top_k)
            responses.append(response)

        return responses

    def save_conversation(
        self,
        responses: Union[Dict, List[Dict]],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Lưu conversation ra file

        Args:
            responses: Response hoặc list responses
            output_path: Đường dẫn file output

        Returns:
            Path của file đã lưu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(responses, dict):
            responses = [responses]

        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "llm_model": self.llm_model,
            "conversations": responses
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        print(f"Đã lưu conversation tại: {output_path}")
        return output_path

    def set_prompt_template(self, template: str):
        """
        Cập nhật prompt template

        Args:
            template: Template mới (phải có {context} và {question})
        """
        if "{context}" not in template or "{question}" not in template:
            raise ValueError("Template phải chứa {context} và {question}")

        self.prompt_template = template
        print("Đã cập nhật prompt template")


# Test function
if __name__ == "__main__":
    # Example usage (requires proper setup)
    print("RAG Module initialized successfully!")
    print("Note: Cần setup VectorDB, Embedder và OpenAI API key để test đầy đủ")

    # Example template
    custom_template = """Bạn là chuyên gia phân tích audio.

Context:
{context}

Question: {question}

Answer:"""

    print(f"\nExample custom template:\n{custom_template}")
