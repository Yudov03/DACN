"""
RAG Module - Retrieval-Augmented Generation
Ho tro ca OpenAI va Google LLM thong qua LangChain
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


class RAGSystem:
    """
    He thong RAG ho tro ca OpenAI va Google LLM
    - OpenAI: gpt-4, gpt-4o, gpt-4o-mini
    - Google: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash
    """

    def __init__(
        self,
        vector_db,
        embedder,
        llm_model: Optional[str] = None,
        provider: str = "google",  # "openai" or "google"
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_k: int = 5
    ):
        """
        Khoi tao RAG System

        Args:
            vector_db: VectorDatabase instance (Qdrant)
            embedder: TextEmbedding instance
            llm_model: Ten model LLM (neu None se dung default theo provider)
            provider: "openai" hoac "google"
            api_key: API key (neu khong truyen se lay tu env)
            temperature: Temperature cho LLM
            max_tokens: Max tokens cho response
            top_k: So luong chunks retrieve
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Default models
        if llm_model is None:
            if self.provider == "google":
                llm_model = "gemini-2.0-flash"
            else:
                llm_model = "gpt-4o-mini"

        self.llm_model = llm_model

        # Get API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            key_name = "GOOGLE_API_KEY" if self.provider == "google" else "OPENAI_API_KEY"
            raise ValueError(f"{key_name} chua duoc cau hinh!")

        # Initialize LLM
        self._init_llm()

        # Setup prompt template
        self.prompt_template = self._get_default_prompt_template()

        print(f"Da khoi tao RAG System - Provider: {self.provider}, Model: {self.llm_model}")

    def _init_llm(self):
        """Initialize LLM based on provider"""
        print(f"Dang khoi tao {self.provider.upper()} LLM '{self.llm_model}'...")

        if self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                google_api_key=self.api_key
            )
        else:  # openai
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )

    def _get_default_prompt_template(self) -> PromptTemplate:
        """Tao prompt template mac dinh"""
        template = """Ban la mot tro ly AI thong minh. Dua tren cac doan van ban sau day duoc trich xuat tu am thanh, hay tra loi cau hoi cua nguoi dung mot cach chinh xac va chi tiet.

Cac doan van ban lien quan (kem timestamp):
{context}

Cau hoi: {question}

Yeu cau:
1. Tra loi dua tren thong tin duoc cung cap
2. Neu thong tin khong du de tra loi, hay noi ro "Toi khong tim thay thong tin lien quan"
3. Trich dan nguon voi format [file_id:start_time-end_time] khi co the
4. Tra loi ngan gon, chinh xac

Tra loi:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        filter_dict: Optional[Dict] = None,
        use_mmr: bool = False
    ) -> Dict:
        """
        Thuc hien query voi RAG pipeline

        Args:
            question: Cau hoi cua nguoi dung
            top_k: So luong chunks retrieve (override default)
            return_sources: Co tra ve sources khong
            filter_dict: Bo loc cho retrieval
            use_mmr: Su dung MMR de tang diversity

        Returns:
            Dict chua answer, sources va metadata
        """
        k = top_k or self.top_k

        # Step 1: Tao embedding cho query
        query_embedding = self.embedder.encode_query(question)

        # Step 2: Retrieve relevant chunks
        if use_mmr:
            retrieved_chunks = self.vector_db.search_with_mmr(
                query_embedding=query_embedding,
                top_k=k,
                filter_dict=filter_dict
            )
        else:
            retrieved_chunks = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=k,
                filter_dict=filter_dict
            )

        if not retrieved_chunks:
            return {
                "answer": "Toi khong tim thay thong tin lien quan de tra loi cau hoi nay.",
                "sources": [],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }

        # Step 3: Format context tu retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Step 4: Tao prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        # Step 5: Goi LLM
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            print(f"Loi khi goi LLM: {str(e)}")
            answer = f"Xin loi, da co loi xay ra khi xu ly cau hoi: {str(e)}"

        # Step 6: Format response
        response_data = {
            "answer": answer,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "num_sources": len(retrieved_chunks),
            "model": self.llm_model,
            "provider": self.provider
        }

        if return_sources:
            response_data["sources"] = self._format_sources(retrieved_chunks)

        return response_data

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format cac chunks thanh context cho LLM"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            audio_file = metadata.get("audio_file", "")

            if start_time is not None and end_time is not None:
                time_info = f"[{audio_file}:{self._format_timestamp(start_time)}-{self._format_timestamp(end_time)}]"
                context_parts.append(f"{i}. {time_info}\n{text}\n")
            else:
                context_parts.append(f"{i}. {text}\n")

        return "\n".join(context_parts)

    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format sources de tra ve cho user"""
        sources = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            source = {
                "text": chunk.get("text", ""),
                "similarity": chunk.get("similarity", 0.0),
                "audio_file": metadata.get("audio_file", ""),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "chunk_id": metadata.get("chunk_id")
            }

            if source["start_time"] is not None:
                source["start_time_formatted"] = self._format_timestamp(source["start_time"])
            if source["end_time"] is not None:
                source["end_time_formatted"] = self._format_timestamp(source["end_time"])

            sources.append(source)

        return sources

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp tu giay sang HH:MM:SS"""
        if seconds is None:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    def query_with_chain(
        self,
        question: str,
        top_k: Optional[int] = None
    ) -> Dict:
        """Query su dung LangChain chain"""
        k = top_k or self.top_k
        return self.query(question, top_k=k)

    def batch_query(
        self,
        questions: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Xu ly nhieu queries cung luc"""
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
        """Luu conversation ra file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(responses, dict):
            responses = [responses]

        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "llm_model": self.llm_model,
            "provider": self.provider,
            "conversations": responses
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        print(f"Da luu conversation tai: {output_path}")
        return output_path

    def set_prompt_template(self, template: str):
        """Cap nhat prompt template"""
        if "{context}" not in template or "{question}" not in template:
            raise ValueError("Template phai chua {context} va {question}")

        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        print("Da cap nhat prompt template")

    def get_retriever_stats(self) -> Dict:
        """Lay thong ke ve retriever"""
        return {
            "llm_model": self.llm_model,
            "llm_provider": self.provider,
            "embedding_model": self.embedder.model_name,
            "embedding_provider": self.embedder.provider,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# Test function
if __name__ == "__main__":
    print("RAG Module initialized successfully!")
    print("Supported providers: openai, google")
    print("\nOpenAI models: gpt-4, gpt-4o, gpt-4o-mini")
    print("Google models: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash")
    print("\nNote: Can setup VectorDB, Embedder va API key de test day du")
