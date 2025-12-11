"""
Context Compression Module - Nen context truoc khi dua vao LLM
Ho tro:
- Extractive compression (trich xuat cau quan trong)
- LLM-based compression (tom tat bang LLM)
- Sentence filtering (loc cau theo relevance)
- Token limiting (gioi han so tokens)
"""

from typing import List, Dict, Optional, Tuple
import re


class ContextCompressor:
    """
    Nen context de giam tokens va tang chat luong answer
    """

    def __init__(
        self,
        method: str = "extractive",  # "extractive", "llm", "filter", "hybrid"
        embedder=None,
        llm=None,
        max_tokens: int = 2000,
        min_similarity: float = 0.3
    ):
        """
        Args:
            method: Phuong phap compression
            embedder: TextEmbedding instance (cho filter)
            llm: LLM instance (cho llm compression)
            max_tokens: Gioi han tokens cho context
            min_similarity: Nguong similarity cho filtering
        """
        self.method = method
        self.embedder = embedder
        self.llm = llm
        self.max_tokens = max_tokens
        self.min_similarity = min_similarity

    def compress(
        self,
        query: str,
        contexts: List[Dict],
        max_contexts: Optional[int] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Nen contexts thanh context string ngan gon

        Args:
            query: Query cua user
            contexts: List cac retrieved chunks
            max_contexts: So luong contexts toi da

        Returns:
            Tuple (compressed_context_string, filtered_contexts)
        """
        if not contexts:
            return "", []

        if max_contexts:
            contexts = contexts[:max_contexts]

        if self.method == "extractive":
            return self._extractive_compress(query, contexts)
        elif self.method == "llm" and self.llm:
            return self._llm_compress(query, contexts)
        elif self.method == "filter" and self.embedder:
            return self._filter_compress(query, contexts)
        elif self.method == "hybrid":
            return self._hybrid_compress(query, contexts)
        else:
            return self._extractive_compress(query, contexts)

    def _extractive_compress(
        self,
        query: str,
        contexts: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Extractive compression - trich xuat cac cau quan trong nhat
        """
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        compressed_contexts = []

        for ctx in contexts:
            text = ctx.get("text", "")
            sentences = self._split_sentences(text)

            # Score each sentence
            scored_sentences = []
            for sent in sentences:
                score = self._sentence_relevance_score(sent, query_terms)
                if score > 0:
                    scored_sentences.append((sent, score))

            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:5]]

            if top_sentences:
                compressed_text = " ".join(top_sentences)
                compressed_ctx = ctx.copy()
                compressed_ctx["text"] = compressed_text
                compressed_ctx["original_length"] = len(text)
                compressed_ctx["compressed_length"] = len(compressed_text)
                compressed_contexts.append(compressed_ctx)

        # Build context string with token limit
        context_string = self._build_context_string(compressed_contexts)
        return context_string, compressed_contexts

    def _filter_compress(
        self,
        query: str,
        contexts: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Filter compression - loc cau theo semantic similarity
        """
        if not self.embedder:
            return self._extractive_compress(query, contexts)

        query_emb = self.embedder.encode_query(query)
        filtered_contexts = []

        for ctx in contexts:
            text = ctx.get("text", "")
            sentences = self._split_sentences(text)

            if not sentences:
                continue

            # Encode sentences
            sent_embs = self.embedder.encode_text(sentences, show_progress=False)

            # Filter by similarity
            relevant_sentences = []
            for sent, emb in zip(sentences, sent_embs):
                sim = self.embedder.compute_similarity(query_emb, emb)
                if sim >= self.min_similarity:
                    relevant_sentences.append((sent, sim))

            # Sort by similarity
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:5]]

            if top_sentences:
                compressed_text = " ".join(top_sentences)
                compressed_ctx = ctx.copy()
                compressed_ctx["text"] = compressed_text
                filtered_contexts.append(compressed_ctx)

        context_string = self._build_context_string(filtered_contexts)
        return context_string, filtered_contexts

    def _llm_compress(
        self,
        query: str,
        contexts: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        LLM compression - tom tat context bang LLM
        """
        if not self.llm:
            return self._extractive_compress(query, contexts)

        compressed_contexts = []

        for ctx in contexts:
            text = ctx.get("text", "")

            # Only compress if text is long enough
            if len(text.split()) < 50:
                compressed_contexts.append(ctx)
                continue

            prompt = f"""Tom tat doan van sau, giu lai thong tin lien quan den cau hoi.
Cau hoi: {query}

Doan van:
{text}

Tom tat (chi giu thong tin quan trong):"""

            try:
                response = self.llm.invoke(prompt)
                summary = response.content.strip()

                compressed_ctx = ctx.copy()
                compressed_ctx["text"] = summary
                compressed_ctx["original_length"] = len(text)
                compressed_ctx["compressed_length"] = len(summary)
                compressed_contexts.append(compressed_ctx)
            except Exception as e:
                print(f"LLM compression error: {e}")
                compressed_contexts.append(ctx)

        context_string = self._build_context_string(compressed_contexts)
        return context_string, compressed_contexts

    def _hybrid_compress(
        self,
        query: str,
        contexts: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Hybrid compression - ket hop extractive + filter
        """
        # First: extractive compression
        _, extracted = self._extractive_compress(query, contexts)

        # Then: filter by similarity if embedder available
        if self.embedder and extracted:
            context_string, filtered = self._filter_compress(query, extracted)
            return context_string, filtered

        return self._build_context_string(extracted), extracted

    def _split_sentences(self, text: str) -> List[str]:
        """Tach van ban thanh cau"""
        # Vietnamese and English sentence splitting
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    def _sentence_relevance_score(self, sentence: str, query_terms: set) -> float:
        """Tinh diem relevance cua cau voi query"""
        sent_terms = set(re.findall(r'\b\w+\b', sentence.lower()))
        if not sent_terms:
            return 0.0

        # Jaccard-like score
        overlap = len(query_terms & sent_terms)
        score = overlap / len(query_terms) if query_terms else 0

        # Boost for longer overlap
        if overlap > 2:
            score *= 1.5

        return score

    def _build_context_string(self, contexts: List[Dict]) -> str:
        """Xay dung context string voi token limit"""
        parts = []
        total_tokens = 0

        for i, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "")
            metadata = ctx.get("metadata", {})

            # Estimate tokens (rough: 1 token ~ 4 chars for Vietnamese)
            estimated_tokens = len(text) // 4

            if total_tokens + estimated_tokens > self.max_tokens:
                # Truncate
                remaining_tokens = self.max_tokens - total_tokens
                char_limit = remaining_tokens * 4
                text = text[:char_limit] + "..."

            # Format with timestamp if available
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            audio_file = metadata.get("audio_file", "")

            if start_time is not None and end_time is not None:
                time_str = f"[{self._format_time(start_time)}-{self._format_time(end_time)}]"
                parts.append(f"{i}. {time_str} {text}")
            else:
                parts.append(f"{i}. {text}")

            total_tokens += estimated_tokens

            if total_tokens >= self.max_tokens:
                break

        return "\n\n".join(parts)

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def estimate_tokens(self, text: str) -> int:
        """Uoc luong so tokens"""
        # Rough estimation: 1 token ~ 4 chars for Vietnamese/English mixed
        return len(text) // 4


class ContextualCompressor:
    """
    LangChain-style contextual compressor
    Ket hop voi retriever de compress ngay khi retrieve
    """

    def __init__(
        self,
        base_retriever,
        compressor: ContextCompressor
    ):
        self.base_retriever = base_retriever
        self.compressor = compressor

    def retrieve_and_compress(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve va compress trong 1 buoc

        Args:
            query: Query cua user
            top_k: So ket qua retrieve

        Returns:
            Tuple (compressed_context, filtered_contexts)
        """
        # Retrieve
        query_emb = self.base_retriever["embedder"].encode_query(query)
        results = self.base_retriever["vector_db"].search(query_emb, top_k=top_k * 2)

        # Compress
        compressed_context, filtered = self.compressor.compress(
            query, results, max_contexts=top_k
        )

        return compressed_context, filtered


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Context Compression Module")
    print("=" * 60)

    # Test extractive compression
    compressor = ContextCompressor(method="extractive", max_tokens=500)

    test_contexts = [
        {
            "text": "Machine learning la mot nhanh cua tri tue nhan tao. No cho phep may tinh hoc tu du lieu. Cac ung dung bao gom nhan dien hinh anh, xu ly ngon ngu tu nhien. Deep learning la mot dang dac biet cua machine learning. Hom nay troi dep qua.",
            "metadata": {"start_time": 0.0, "end_time": 30.0, "audio_file": "lecture.mp3"}
        },
        {
            "text": "NLP giup may tinh hieu ngon ngu con nguoi. Cac mo hinh nhu BERT va GPT da tao ra dot pha. Transformer la kien truc quan trong trong NLP. An sang rat ngon.",
            "metadata": {"start_time": 30.0, "end_time": 60.0, "audio_file": "lecture.mp3"}
        }
    ]

    query = "Machine learning la gi?"

    print(f"\nQuery: {query}")
    print(f"Original contexts: {len(test_contexts)}")

    context_str, compressed = compressor.compress(query, test_contexts)

    print(f"\nCompressed context:\n{context_str}")
    print(f"\nCompression ratio: {len(compressed)}/{len(test_contexts)}")

    for ctx in compressed:
        orig = ctx.get("original_length", len(ctx["text"]))
        comp = ctx.get("compressed_length", len(ctx["text"]))
        print(f"  - {orig} -> {comp} chars ({comp/orig*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Context Compression Module ready!")
