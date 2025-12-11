"""
Query Expansion Module - Mo rong query de cai thien recall
Ho tro:
- Synonym expansion (tu dong nghia)
- Paraphrase generation (viet lai cau hoi)
- Pseudo-relevance feedback (PRF)
- Multi-query retrieval
"""

from typing import List, Dict, Optional, Tuple
import re


class QueryExpander:
    """
    Mo rong query de cai thien retrieval recall
    """

    def __init__(
        self,
        method: str = "multi_query",  # "synonym", "paraphrase", "prf", "multi_query"
        embedder=None,
        llm=None,
        num_expansions: int = 3
    ):
        """
        Args:
            method: Phuong phap expansion
            embedder: TextEmbedding instance (cho PRF)
            llm: LLM instance (cho paraphrase/multi_query)
            num_expansions: So luong queries mo rong
        """
        self.method = method
        self.embedder = embedder
        self.llm = llm
        self.num_expansions = num_expansions

        # Vietnamese synonyms dictionary
        self.vi_synonyms = {
            "la gi": ["la cai gi", "nghia la gi", "duoc hieu nhu the nao"],
            "nhu the nao": ["ra sao", "bang cach nao", "lam sao"],
            "tai sao": ["vi sao", "ly do gi", "nguyen nhan gi"],
            "khi nao": ["luc nao", "thoi diem nao", "bao gio"],
            "o dau": ["tai dau", "vi tri nao", "cho nao"],
            "ai": ["nguoi nao", "doi tuong nao"],
            "cai gi": ["dieu gi", "thu gi", "vat gi"],
            "bao nhieu": ["so luong bao nhieu", "may", "bao lau"],
            # Technical terms
            "ai": ["tri tue nhan tao", "artificial intelligence"],
            "ml": ["machine learning", "hoc may"],
            "nlp": ["xu ly ngon ngu tu nhien", "natural language processing"],
            "deep learning": ["hoc sau", "mang neural sau"],
        }

        # English synonyms
        self.en_synonyms = {
            "what is": ["define", "explain", "describe"],
            "how": ["in what way", "by what means"],
            "why": ["for what reason", "what causes"],
            "when": ["at what time", "what date"],
            "where": ["in what place", "at what location"],
            "who": ["which person", "what individual"],
        }

    def expand(self, query: str) -> List[str]:
        """
        Mo rong query thanh nhieu queries

        Args:
            query: Query goc

        Returns:
            List queries (bao gom query goc)
        """
        if self.method == "synonym":
            return self.expand_with_synonyms(query)
        elif self.method == "paraphrase" and self.llm:
            return self.expand_with_paraphrase(query)
        elif self.method == "prf" and self.embedder:
            return [query]  # PRF is done during retrieval
        elif self.method == "multi_query" and self.llm:
            return self.expand_with_multi_query(query)
        else:
            return self.expand_with_synonyms(query)

    def expand_with_synonyms(self, query: str) -> List[str]:
        """Mo rong query bang tu dong nghia"""
        queries = [query]
        query_lower = query.lower()

        # Check Vietnamese synonyms
        for term, synonyms in self.vi_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:self.num_expansions]:
                    expanded = query_lower.replace(term, syn)
                    if expanded not in queries:
                        queries.append(expanded)

        # Check English synonyms
        for term, synonyms in self.en_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:self.num_expansions]:
                    expanded = query_lower.replace(term, syn)
                    if expanded not in queries:
                        queries.append(expanded)

        return queries[:self.num_expansions + 1]

    def expand_with_paraphrase(self, query: str) -> List[str]:
        """Mo rong query bang cach viet lai (can LLM)"""
        if not self.llm:
            return [query]

        prompt = f"""Viet lai cau hoi sau thanh {self.num_expansions} cach khac nhau, giu nguyen y nghia.
Chi tra ve cac cau hoi, moi cau mot dong, khong danh so.

Cau hoi goc: {query}

Cac cau hoi viet lai:"""

        try:
            response = self.llm.invoke(prompt)
            paraphrases = response.content.strip().split('\n')
            paraphrases = [p.strip() for p in paraphrases if p.strip()]

            queries = [query] + paraphrases[:self.num_expansions]
            return queries
        except Exception as e:
            print(f"Paraphrase error: {e}")
            return [query]

    def expand_with_multi_query(self, query: str) -> List[str]:
        """
        Multi-query expansion - tao nhieu goc nhin khac nhau cho cung 1 cau hoi
        """
        if not self.llm:
            return self.expand_with_synonyms(query)

        prompt = f"""Nguoi dung dang hoi: "{query}"

Hay tao {self.num_expansions} cau hoi lien quan de tim kiem thong tin tu nhieu goc do khac nhau.
Cac cau hoi nen:
1. Giu y nghia chinh cua cau hoi goc
2. Su dung tu khoa khac nhau
3. Hoi tu cac khia canh khac nhau

Chi tra ve cac cau hoi, moi cau mot dong, khong giai thich.

Cac cau hoi:"""

        try:
            response = self.llm.invoke(prompt)
            expanded = response.content.strip().split('\n')
            expanded = [q.strip().lstrip('0123456789.-) ') for q in expanded if q.strip()]

            queries = [query] + expanded[:self.num_expansions]
            return queries
        except Exception as e:
            print(f"Multi-query error: {e}")
            return self.expand_with_synonyms(query)

    def expand_with_prf(
        self,
        query: str,
        initial_results: List[Dict],
        top_k: int = 3,
        num_terms: int = 5
    ) -> str:
        """
        Pseudo-Relevance Feedback - mo rong query tu ket qua ban dau

        Args:
            query: Query goc
            initial_results: Ket qua retrieval ban dau
            top_k: So documents dung de expand
            num_terms: So terms them vao

        Returns:
            Expanded query
        """
        if not initial_results:
            return query

        # Extract terms from top results
        top_docs = initial_results[:top_k]
        all_text = " ".join([doc.get("text", "") for doc in top_docs])

        # Simple term extraction (frequency-based)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Remove query terms
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        for term in query_terms:
            word_freq.pop(term, None)

        # Get top terms
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_terms]
        expansion_terms = [term for term, freq in top_terms]

        # Combine with original query
        expanded_query = query + " " + " ".join(expansion_terms)
        return expanded_query


class MultiQueryRetriever:
    """
    Retriever su dung nhieu queries va merge ket qua
    """

    def __init__(
        self,
        vector_db,
        embedder,
        query_expander: Optional[QueryExpander] = None
    ):
        self.vector_db = vector_db
        self.embedder = embedder
        self.query_expander = query_expander or QueryExpander()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand: bool = True,
        fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "score"
    ) -> List[Dict]:
        """
        Retrieve voi multi-query expansion

        Args:
            query: Query goc
            top_k: So ket qua tra ve
            expand: Co expand query khong
            fusion_method: Phuong phap merge ket qua

        Returns:
            List documents da merge va re-rank
        """
        if expand:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]

        # Retrieve for each query
        all_results = []
        for q in queries:
            query_emb = self.embedder.encode_query(q)
            results = self.vector_db.search(query_emb, top_k=top_k * 2)
            all_results.append(results)

        # Merge results
        if fusion_method == "rrf":
            merged = self._reciprocal_rank_fusion(all_results)
        else:
            merged = self._score_fusion(all_results)

        return merged[:top_k]

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion - merge ket qua tu nhieu queries

        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        doc_scores = {}
        doc_data = {}

        for results in result_lists:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get("id", str(rank))
                rrf_score = 1.0 / (k + rank)

                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_data[doc_id] = doc

        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result list
        results = []
        for doc_id, score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc["rrf_score"] = score
            doc["original_similarity"] = doc.get("similarity", 0)
            doc["similarity"] = score  # Use RRF score as similarity
            results.append(doc)

        return results

    def _score_fusion(self, result_lists: List[List[Dict]]) -> List[Dict]:
        """Merge by averaging scores"""
        doc_scores = {}
        doc_counts = {}
        doc_data = {}

        for results in result_lists:
            for doc in results:
                doc_id = doc.get("id", "")
                score = doc.get("similarity", 0)

                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                    doc_counts[doc_id] += 1
                else:
                    doc_scores[doc_id] = score
                    doc_counts[doc_id] = 1
                    doc_data[doc_id] = doc

        # Average scores
        for doc_id in doc_scores:
            doc_scores[doc_id] /= doc_counts[doc_id]

        # Sort
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc["similarity"] = score
            results.append(doc)

        return results


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Query Expansion Module")
    print("=" * 60)

    # Test synonym expansion
    expander = QueryExpander(method="synonym", num_expansions=3)

    test_queries = [
        "AI la gi?",
        "Machine learning hoat dong nhu the nao?",
        "Tai sao deep learning quan trong?",
        "What is NLP?",
    ]

    print("\n--- Synonym Expansion ---")
    for query in test_queries:
        expanded = expander.expand(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded}")

    print("\n" + "=" * 60)
    print("Query Expansion Module ready!")
