"""
RAG Module - Retrieval-Augmented Generation
Ho tro nhieu providers:
- Local: Ollama (llama3.2, mistral, qwen2.5, ...)
- Cloud: OpenAI, Google (LangChain)

Enhanced features:
- Anti-hallucination with answer verification
- Conflict detection and resolution
- Date-aware retrieval (prefer newer info)
- Multiple prompt templates (strict, citation_required, conflict_aware)
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Import new modules for enhanced RAG
try:
    from .answer_verification import AnswerVerifier, AbstentionChecker, verify_rag_answer
    from .conflict_detection import ConflictDetector, detect_conflicts, ResolutionStrategy
    from .prompt_templates import PromptTemplateManager
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False


# Supported Ollama models with descriptions
OLLAMA_MODELS = {
    # Llama models
    "llama3.2": "Meta Llama 3.2 (3B) - Fast, good for general tasks",
    "llama3.2:1b": "Meta Llama 3.2 (1B) - Very fast, lightweight",
    "llama3.1": "Meta Llama 3.1 (8B) - Better quality, slower",
    "llama3.1:70b": "Meta Llama 3.1 (70B) - Best quality, requires high VRAM",
    # Mistral models
    "mistral": "Mistral 7B - Good balance of speed and quality",
    "mistral-nemo": "Mistral Nemo (12B) - Better reasoning",
    # Qwen models (good for multilingual/Vietnamese)
    "qwen2.5": "Qwen 2.5 (7B) - Excellent multilingual support",
    "qwen2.5:3b": "Qwen 2.5 (3B) - Fast multilingual",
    "qwen2.5:14b": "Qwen 2.5 (14B) - Better quality multilingual",
    # Gemma models
    "gemma2": "Google Gemma 2 (9B) - Good general purpose",
    "gemma2:2b": "Google Gemma 2 (2B) - Very fast",
    # Phi models (Microsoft)
    "phi3": "Microsoft Phi-3 (3.8B) - Efficient, good reasoning",
    # Vietnamese-optimized
    "vinallama": "VinaLlama - Vietnamese fine-tuned",
}


class RAGSystem:
    """
    He thong RAG ho tro nhieu providers:
    - Local (Ollama): llama3.2, mistral, qwen2.5, ...
    - OpenAI: gpt-4, gpt-4o, gpt-4o-mini
    - Google: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash
    """

    def __init__(
        self,
        vector_db,
        embedder,
        llm_model: Optional[str] = None,
        provider: str = None,  # "ollama", "openai", or "google" - reads from LLM_PROVIDER
        api_key: Optional[str] = None,
        temperature: float = None,  # Reads from LLM_TEMPERATURE
        max_tokens: int = None,  # Reads from LLM_MAX_TOKENS
        top_k: int = None,  # Reads from TOP_K
        ollama_base_url: str = None,  # Reads from OLLAMA_BASE_URL
        # Enhanced features
        enable_verification: bool = True,
        enable_conflict_detection: bool = True,
        enable_date_boost: bool = True,
        prompt_template_name: str = "strict_qa",  # strict_qa, citation_required, conflict_aware, safe_abstention
        language: str = "vi",
        min_confidence: float = 0.4,
        date_boost_weight: float = 0.2
    ):
        """
        Khoi tao RAG System

        Args:
            vector_db: VectorDatabase instance (Qdrant)
            embedder: TextEmbedding instance
            llm_model: Ten model LLM (neu None se dung default theo provider)
            provider: "ollama" (local), "openai", hoac "google"
            api_key: API key (chi can cho openai/google)
            temperature: Temperature cho LLM
            max_tokens: Max tokens cho response
            top_k: So luong chunks retrieve
            ollama_base_url: URL cua Ollama server (default: http://localhost:11434)

            # Enhanced RAG features (anti-hallucination)
            enable_verification: Bat answer verification (default: True)
            enable_conflict_detection: Bat conflict detection (default: True)
            enable_date_boost: Uu tien thong tin moi hon (default: True)
            prompt_template_name: Ten prompt template (strict_qa, citation_required, conflict_aware, safe_abstention)
            language: Ngon ngu cho prompts (vi/en)
            min_confidence: Nguong confidence toi thieu (default: 0.4)
            date_boost_weight: Trong so cho date boost (default: 0.2)
        """
        self.vector_db = vector_db
        self.embedder = embedder
        # Read from .env with defaults
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "500"))
        self.top_k = top_k if top_k is not None else int(os.getenv("TOP_K", "5"))
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Enhanced features settings
        self.enable_verification = enable_verification and HAS_ENHANCED_FEATURES
        self.enable_conflict_detection = enable_conflict_detection and HAS_ENHANCED_FEATURES
        self.enable_date_boost = enable_date_boost
        self.prompt_template_name = prompt_template_name
        self.language = language
        self.min_confidence = min_confidence
        self.date_boost_weight = date_boost_weight

        # Initialize enhanced components
        if HAS_ENHANCED_FEATURES:
            self.verifier = AnswerVerifier(min_confidence=min_confidence) if enable_verification else None
            self.abstention_checker = AbstentionChecker() if enable_verification else None
            self.conflict_detector = ConflictDetector() if enable_conflict_detection else None
            self.prompt_manager = PromptTemplateManager(language=language)
        else:
            self.verifier = None
            self.abstention_checker = None
            self.conflict_detector = None
            self.prompt_manager = None

        # Default models (read from .env)
        if llm_model is None:
            if self.provider == "ollama":
                llm_model = os.getenv("OLLAMA_MODEL", "llama3.2")
            elif self.provider == "google":
                llm_model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash")
            else:
                llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

        self.llm_model = llm_model

        # Get API key (only needed for cloud providers)
        if self.provider in ["openai", "google"]:
            if api_key:
                self.api_key = api_key
            elif self.provider == "google":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.api_key:
                key_name = "GOOGLE_API_KEY" if self.provider == "google" else "OPENAI_API_KEY"
                raise ValueError(f"{key_name} chua duoc cau hinh!")
        else:
            self.api_key = None

        # Initialize LLM
        self._init_llm()

        # Setup prompt template
        self.prompt_template = self._get_default_prompt_template()

        print(f"Da khoi tao RAG System - Provider: {self.provider}, Model: {self.llm_model}")

    def _init_llm(self):
        """Initialize LLM based on provider"""
        print(f"Dang khoi tao {self.provider.upper()} LLM '{self.llm_model}'...")

        if self.provider == "ollama":
            self._init_ollama_llm()

        elif self.provider == "google":
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

    def _init_ollama_llm(self):
        """Initialize Ollama local LLM"""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama chua duoc cai dat. "
                "Chay: pip install langchain-ollama"
            )

        # Check if Ollama server is running
        if not self._check_ollama_server():
            raise ConnectionError(
                f"Khong the ket noi toi Ollama server tai {self.ollama_base_url}. "
                "Hay chac chan Ollama dang chay: ollama serve"
            )

        # Check if model is available
        if not self._check_ollama_model():
            print(f"Model '{self.llm_model}' chua duoc tai. Dang pull...")
            self._pull_ollama_model()

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            base_url=self.ollama_base_url
        )

    def _check_ollama_server(self) -> bool:
        """Check if Ollama server is running"""
        import requests
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _check_ollama_model(self) -> bool:
        """Check if model is available in Ollama"""
        import requests
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                # Check both exact name and base name
                base_model = self.llm_model.split(":")[0]
                return self.llm_model in model_names or base_model in model_names
            return False
        except Exception:
            return False

    def _pull_ollama_model(self):
        """Pull model from Ollama registry"""
        import requests
        print(f"Dang tai model '{self.llm_model}' tu Ollama... (co the mat vai phut)")
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": self.llm_model},
                stream=True,
                timeout=600  # 10 minutes timeout for large models
            )
            for line in response.iter_lines():
                if line:
                    import json as json_lib
                    data = json_lib.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "downloading" in status:
                        print(f"  {status}")
            print(f"Da tai xong model '{self.llm_model}'")
        except Exception as e:
            raise RuntimeError(f"Loi khi tai model: {e}")

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
        use_mmr: bool = False,
        # Enhanced options
        verify_answer: Optional[bool] = None,
        detect_conflicts: Optional[bool] = None,
        use_date_boost: Optional[bool] = None,
        prompt_template: Optional[str] = None
    ) -> Dict:
        """
        Thuc hien query voi RAG pipeline (Enhanced)

        Args:
            question: Cau hoi cua nguoi dung
            top_k: So luong chunks retrieve (override default)
            return_sources: Co tra ve sources khong
            filter_dict: Bo loc cho retrieval
            use_mmr: Su dung MMR de tang diversity

            # Enhanced options (override defaults)
            verify_answer: Bat/tat verification cho query nay
            detect_conflicts: Bat/tat conflict detection cho query nay
            use_date_boost: Bat/tat date boost cho query nay
            prompt_template: Override prompt template (strict_qa, citation_required, conflict_aware, safe_abstention)

        Returns:
            Dict chua answer, sources, metadata, verification_result, conflicts
        """
        k = top_k or self.top_k

        # Resolve feature flags (use query-level override or instance defaults)
        do_verify = verify_answer if verify_answer is not None else self.enable_verification
        do_conflict = detect_conflicts if detect_conflicts is not None else self.enable_conflict_detection
        do_date_boost = use_date_boost if use_date_boost is not None else self.enable_date_boost
        template_name = prompt_template or self.prompt_template_name

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

        # Step 2.5: Check abstention (should we refuse to answer?)
        if do_verify and self.abstention_checker and retrieved_chunks:
            should_abstain, abstain_reason = self.abstention_checker.should_abstain(
                question, retrieved_chunks
            )
            if should_abstain:
                return {
                    "answer": self.abstention_checker.get_abstention_response(abstain_reason, question),
                    "sources": [],
                    "question": question,
                    "timestamp": datetime.now().isoformat(),
                    "abstained": True,
                    "abstain_reason": abstain_reason
                }

        if not retrieved_chunks:
            return {
                "answer": "Toi khong tim thay thong tin lien quan de tra loi cau hoi nay.",
                "sources": [],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }

        # Step 2.6: Apply date boost (prefer newer info)
        if do_date_boost:
            retrieved_chunks = self._apply_date_boost(retrieved_chunks)

        # Step 3: Detect and resolve conflicts
        conflict_result = None
        if do_conflict and self.conflict_detector:
            conflict_result = self.conflict_detector.detect_and_resolve(
                retrieved_chunks, question, ResolutionStrategy.PREFER_NEWER
            )
            if conflict_result.has_conflicts:
                # Use resolved context
                context = conflict_result.resolved_context
            else:
                context = self._format_context(retrieved_chunks)
        else:
            context = self._format_context(retrieved_chunks)

        # Step 4: Tao prompt (use enhanced template if available)
        if self.prompt_manager and template_name:
            system_prompt, user_prompt = self.prompt_manager.format_prompt(
                template_name, context, question
            )
        else:
            system_prompt = "Ban la tro ly AI tra loi cau hoi dua tren tai lieu."
            user_prompt = self.prompt_template.format(
                context=context,
                question=question
            )

        # Step 5: Goi LLM (use proper message format for chat models)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            print(f"Loi khi goi LLM: {str(e)}")
            answer = f"Xin loi, da co loi xay ra khi xu ly cau hoi: {str(e)}"

        # Step 6: Verify answer (anti-hallucination)
        verification_result = None
        if do_verify and self.verifier:
            verification_result = self.verifier.verify(
                answer, context, question,
                sources=[{"similarity": c.get("similarity", 0)} for c in retrieved_chunks]
            )

            # If should abstain based on verification, modify answer
            if verification_result.should_abstain:
                answer = (
                    f"[Luu y: Do tin cay thap ({verification_result.confidence_score:.0%})]\n\n"
                    f"{answer}\n\n"
                    f"---\n"
                    f"Canh bao: Cau tra loi co the khong chinh xac do thong tin trong tai lieu khong du ro rang. "
                    f"Vui long kiem tra lai nguon goc."
                )

        # Step 7: Format response
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

        # Add verification info
        if verification_result:
            response_data["verification"] = {
                "confidence": verification_result.confidence_score,
                "grounding_level": verification_result.grounding_level.value,
                "warnings": verification_result.warnings,
                "suggested_action": verification_result.suggested_action
            }

        # Add conflict info
        if conflict_result and conflict_result.has_conflicts:
            response_data["conflicts"] = {
                "detected": True,
                "count": len(conflict_result.conflicts),
                "resolution_notes": conflict_result.resolution_notes,
                "warnings": conflict_result.warnings
            }

        return response_data

    def _apply_date_boost(self, chunks: List[Dict]) -> List[Dict]:
        """Apply date boost to prefer newer information"""
        if not chunks:
            return chunks

        # Extract and score by date
        scored_chunks = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            metadata = chunk.get("metadata", {})

            # Original similarity score
            original_score = chunk.get("similarity", chunk.get("score", 0.5))

            # Date score (newer = higher)
            date_score = 0.5  # Default neutral
            if self.conflict_detector:
                date = self.conflict_detector._extract_date(text, metadata)
                if date:
                    days_ago = (datetime.now() - date).days
                    date_score = max(0, 1 - days_ago / 1825)  # 5 year scale

            # Check for "new/updated" keywords
            text_lower = text.lower()
            if any(kw in text_lower for kw in ['mới', 'cập nhật', 'hiện hành', 'new', 'updated', 'current']):
                date_score += 0.2
            if any(kw in text_lower for kw in ['cũ', 'trước đây', 'old', 'previous', 'deprecated']):
                date_score -= 0.2

            date_score = max(0, min(1, date_score))

            # Combined score
            final_score = (original_score * (1 - self.date_boost_weight) +
                          date_score * self.date_boost_weight)

            chunk["_date_boosted_score"] = final_score
            chunk["_date_score"] = date_score
            scored_chunks.append(chunk)

        # Sort by boosted score
        scored_chunks.sort(key=lambda x: x.get("_date_boosted_score", 0), reverse=True)
        return scored_chunks

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

            # TODO: Implement deep_link for audio/video sources (Chapter 4 design)
            # Format: ?doc=doc_id&t=start_time (giay)
            # Cho phep user click vao citation de nhay den dung timestamp trong audio/video player
            # Example:
            #   if source["start_time"] is not None:
            #       doc_id = metadata.get("doc_id", source["chunk_id"])
            #       source["deep_link"] = f"?doc={doc_id}&t={int(source['start_time'])}"

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


# Utility functions
def list_ollama_models():
    """List all supported Ollama models"""
    print("\n=== SUPPORTED OLLAMA MODELS ===\n")
    for model, desc in OLLAMA_MODELS.items():
        print(f"  {model}: {desc}")
    print("\nCach su dung:")
    print("  1. Cai dat Ollama: https://ollama.ai/download")
    print("  2. Chay Ollama: ollama serve")
    print("  3. Tai model: ollama pull llama3.2")
    print("  4. Dung trong code: RAGSystem(provider='ollama', llm_model='llama3.2')")


def check_ollama_status(base_url: str = "http://localhost:11434") -> Dict:
    """Check Ollama server status and available models"""
    import requests

    status = {
        "server_running": False,
        "available_models": [],
        "base_url": base_url
    }

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            status["server_running"] = True
            models = response.json().get("models", [])
            status["available_models"] = [m.get("name", "") for m in models]
    except Exception as e:
        status["error"] = str(e)

    return status


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Module - Multi-provider LLM Support")
    print("=" * 60)

    print("\nSupported providers:")
    print("  - ollama (local): Free, offline, privacy-friendly")
    print("  - google (cloud): High quality, requires API key")
    print("  - openai (cloud): High quality, requires API key")

    print("\n--- OLLAMA MODELS ---")
    list_ollama_models()

    print("\n--- CLOUD MODELS ---")
    print("\nOpenAI: gpt-4, gpt-4o, gpt-4o-mini")
    print("Google: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash")

    # Check Ollama status
    print("\n--- OLLAMA STATUS ---")
    status = check_ollama_status()
    if status["server_running"]:
        print(f"Ollama server: RUNNING at {status['base_url']}")
        if status["available_models"]:
            print(f"Available models: {', '.join(status['available_models'])}")
        else:
            print("No models installed. Run: ollama pull llama3.2")
    else:
        print("Ollama server: NOT RUNNING")
        print("To start: ollama serve")

    print("\n" + "=" * 60)
