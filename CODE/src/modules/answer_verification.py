"""
Answer Verification Module - Anti-Hallucination System
=======================================================

Cung cấp các cơ chế để:
1. Kiểm tra câu trả lời có được grounded trong context không
2. Phát hiện potential hallucinations
3. Quyết định abstention khi context không đủ tin cậy
4. Tính confidence score cho câu trả lời
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class GroundingLevel(Enum):
    """Mức độ grounding của câu trả lời"""
    FULLY_GROUNDED = "fully_grounded"      # Hoàn toàn từ context
    PARTIALLY_GROUNDED = "partially_grounded"  # Một phần từ context
    LIKELY_HALLUCINATED = "likely_hallucinated"  # Có thể bịa
    UNVERIFIABLE = "unverifiable"          # Không thể xác minh


@dataclass
class VerificationResult:
    """Kết quả verification của câu trả lời"""

    # Overall assessment
    grounding_level: GroundingLevel
    confidence_score: float  # 0.0 - 1.0
    should_abstain: bool     # Có nên từ chối trả lời không

    # Details
    grounded_claims: List[str] = field(default_factory=list)
    ungrounded_claims: List[str] = field(default_factory=list)
    source_coverage: float = 0.0  # % câu trả lời có source

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Suggested action
    suggested_action: str = ""

    def to_dict(self) -> Dict:
        return {
            "grounding_level": self.grounding_level.value,
            "confidence_score": self.confidence_score,
            "should_abstain": self.should_abstain,
            "grounded_claims": self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "source_coverage": self.source_coverage,
            "warnings": self.warnings,
            "suggested_action": self.suggested_action,
        }


class AnswerVerifier:
    """
    Verifier để kiểm tra câu trả lời có được ground trong context không.

    Usage:
        verifier = AnswerVerifier()
        result = verifier.verify(answer, context, question)

        if result.should_abstain:
            return "Không có đủ thông tin để trả lời câu hỏi này."
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_source_coverage: float = 0.3,
        abstain_threshold: float = 0.4,
        embedder=None  # Optional: for semantic similarity
    ):
        """
        Args:
            min_confidence: Ngưỡng confidence tối thiểu
            min_source_coverage: % tối thiểu câu trả lời phải có trong context
            abstain_threshold: Dưới ngưỡng này sẽ recommend abstain
            embedder: TextEmbedding instance cho semantic matching
        """
        self.min_confidence = min_confidence
        self.min_source_coverage = min_source_coverage
        self.abstain_threshold = abstain_threshold
        self.embedder = embedder

        # Các từ/cụm từ cho thấy LLM đang suy luận thay vì trích dẫn
        self.inference_indicators = [
            "có thể", "có lẽ", "chắc là", "dường như", "hình như",
            "tôi nghĩ", "tôi cho rằng", "theo tôi",
            "probably", "maybe", "perhaps", "i think", "i believe",
            "it seems", "likely", "possibly",
            "suy ra", "từ đó", "do đó có thể thấy",
        ]

        # Cụm từ cho thấy không có thông tin
        self.no_info_phrases = [
            "không có thông tin", "không tìm thấy", "không đề cập",
            "không được nêu", "không có trong", "no information",
            "not found", "not mentioned", "không rõ",
        ]

    def verify(
        self,
        answer: str,
        context: str,
        question: str,
        sources: List[Dict] = None
    ) -> VerificationResult:
        """
        Verify câu trả lời có được ground trong context không.

        Args:
            answer: Câu trả lời từ LLM
            context: Context đã retrieve
            question: Câu hỏi gốc
            sources: List các source chunks (optional)

        Returns:
            VerificationResult với đánh giá chi tiết
        """
        warnings = []

        # 1. Check for explicit "no info" response
        if self._is_no_info_response(answer):
            return VerificationResult(
                grounding_level=GroundingLevel.FULLY_GROUNDED,
                confidence_score=1.0,
                should_abstain=False,
                suggested_action="accept",
                warnings=["LLM đã tự nhận không có thông tin - đây là phản hồi hợp lệ"]
            )

        # 2. Extract claims from answer
        claims = self._extract_claims(answer)

        # 3. Check each claim against context
        grounded_claims = []
        ungrounded_claims = []

        for claim in claims:
            is_grounded, evidence = self._check_claim_grounding(claim, context)
            if is_grounded:
                grounded_claims.append(claim)
            else:
                ungrounded_claims.append(claim)

        # 4. Calculate source coverage
        if claims:
            source_coverage = len(grounded_claims) / len(claims)
        else:
            source_coverage = 0.0

        # 5. Check for inference indicators
        inference_count = self._count_inference_indicators(answer)
        if inference_count > 0:
            warnings.append(f"Phát hiện {inference_count} cụm từ suy luận - có thể không hoàn toàn từ context")

        # 6. Check retrieval quality
        retrieval_confidence = self._assess_retrieval_quality(sources) if sources else 0.5

        # 7. Calculate overall confidence
        confidence_score = self._calculate_confidence(
            source_coverage=source_coverage,
            retrieval_confidence=retrieval_confidence,
            inference_count=inference_count,
            num_claims=len(claims)
        )

        # 8. Determine grounding level
        grounding_level = self._determine_grounding_level(
            source_coverage, confidence_score, len(ungrounded_claims)
        )

        # 9. Decide abstention
        should_abstain = confidence_score < self.abstain_threshold

        # 10. Generate suggested action
        suggested_action = self._suggest_action(
            grounding_level, confidence_score, should_abstain, warnings
        )

        return VerificationResult(
            grounding_level=grounding_level,
            confidence_score=confidence_score,
            should_abstain=should_abstain,
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            source_coverage=source_coverage,
            warnings=warnings,
            suggested_action=suggested_action
        )

    def _is_no_info_response(self, answer: str) -> bool:
        """Check if answer is explicitly saying no info found"""
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in self.no_info_phrases)

    def _extract_claims(self, answer: str) -> List[str]:
        """
        Trích xuất các claims/facts từ câu trả lời.
        Mỗi câu được coi là một claim.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', answer)

        # Filter out very short sentences and clean
        claims = []
        for sent in sentences:
            sent = sent.strip()
            # Bỏ qua câu quá ngắn hoặc chỉ là transitional phrases
            if len(sent) > 15 and not self._is_transitional(sent):
                claims.append(sent)

        return claims

    def _is_transitional(self, sentence: str) -> bool:
        """Check if sentence is just a transitional phrase"""
        transitional = [
            "dưới đây là", "sau đây là", "cụ thể như sau",
            "here is", "below are", "the following",
            "theo như", "based on",
        ]
        sent_lower = sentence.lower()
        return any(t in sent_lower for t in transitional)

    def _check_claim_grounding(
        self,
        claim: str,
        context: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a claim is grounded in the context.

        Returns:
            Tuple (is_grounded, evidence_snippet)
        """
        claim_lower = claim.lower()
        context_lower = context.lower()

        # Extract key terms from claim (nouns, numbers, key phrases)
        key_terms = self._extract_key_terms(claim)

        if not key_terms:
            return False, None

        # Check how many key terms appear in context
        matches = sum(1 for term in key_terms if term.lower() in context_lower)
        coverage = matches / len(key_terms)

        # If most key terms found, consider it grounded
        if coverage >= 0.5:
            # Find evidence snippet
            for term in key_terms:
                idx = context_lower.find(term.lower())
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(context), idx + len(term) + 50)
                    evidence = context[start:end]
                    return True, evidence
            return True, None

        return False, None

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms (numbers, proper nouns, important words)"""
        terms = []

        # Extract numbers and percentages
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        terms.extend(numbers)

        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        terms.extend(quoted)

        # Extract capitalized words (proper nouns)
        # For Vietnamese, also look for key content words
        words = text.split()
        for word in words:
            # Skip common words
            if len(word) > 3 and word[0].isupper():
                terms.append(word)

        # Extract key Vietnamese content words
        vietnamese_content = re.findall(
            r'\b(sinh viên|học phí|điểm|môn học|quy định|thời gian|'
            r'học kỳ|tín chỉ|đăng ký|deadline|hạn|yêu cầu)\b',
            text.lower()
        )
        terms.extend(vietnamese_content)

        return list(set(terms))

    def _count_inference_indicators(self, answer: str) -> int:
        """Count inference/speculation indicators in answer"""
        answer_lower = answer.lower()
        return sum(1 for ind in self.inference_indicators if ind in answer_lower)

    def _assess_retrieval_quality(self, sources: List[Dict]) -> float:
        """
        Assess quality of retrieved sources.

        Returns score 0.0 - 1.0
        """
        if not sources:
            return 0.0

        scores = []
        for source in sources:
            similarity = source.get("similarity", 0.5)
            scores.append(similarity)

        # Average of top sources, weighted toward highest
        if scores:
            scores.sort(reverse=True)
            # Weighted average: top source counts more
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight

        return 0.5

    def _calculate_confidence(
        self,
        source_coverage: float,
        retrieval_confidence: float,
        inference_count: int,
        num_claims: int
    ) -> float:
        """Calculate overall confidence score"""

        # Base score from source coverage and retrieval quality
        base_score = (source_coverage * 0.6 + retrieval_confidence * 0.4)

        # Penalty for inference indicators
        inference_penalty = min(0.3, inference_count * 0.1)

        # Penalty for too few or too many claims
        if num_claims == 0:
            claims_penalty = 0.3
        elif num_claims > 10:
            claims_penalty = 0.1  # Might be over-explaining
        else:
            claims_penalty = 0

        confidence = base_score - inference_penalty - claims_penalty

        return max(0.0, min(1.0, confidence))

    def _determine_grounding_level(
        self,
        source_coverage: float,
        confidence: float,
        ungrounded_count: int
    ) -> GroundingLevel:
        """Determine grounding level based on metrics"""

        if source_coverage >= 0.8 and confidence >= 0.7:
            return GroundingLevel.FULLY_GROUNDED
        elif source_coverage >= 0.5 or confidence >= 0.5:
            return GroundingLevel.PARTIALLY_GROUNDED
        elif ungrounded_count > 3 or confidence < 0.3:
            return GroundingLevel.LIKELY_HALLUCINATED
        else:
            return GroundingLevel.UNVERIFIABLE

    def _suggest_action(
        self,
        grounding_level: GroundingLevel,
        confidence: float,
        should_abstain: bool,
        warnings: List[str]
    ) -> str:
        """Suggest action based on verification result"""

        if should_abstain:
            return "abstain - Recommend không trả lời do confidence thấp"

        if grounding_level == GroundingLevel.FULLY_GROUNDED:
            return "accept - Câu trả lời được ground tốt trong context"

        if grounding_level == GroundingLevel.PARTIALLY_GROUNDED:
            if warnings:
                return "accept_with_warning - Chấp nhận nhưng cần cảnh báo user"
            return "accept - Câu trả lời được ground một phần"

        if grounding_level == GroundingLevel.LIKELY_HALLUCINATED:
            return "reject - Có khả năng hallucination, cần review"

        return "manual_review - Cần kiểm tra thủ công"


class AbstentionChecker:
    """
    Checker để quyết định có nên từ chối trả lời không.

    Abstention scenarios:
    1. Retrieval scores quá thấp
    2. Question không liên quan đến corpus
    3. Context không đủ để trả lời
    """

    def __init__(
        self,
        min_retrieval_score: float = 0.3,
        min_relevant_chunks: int = 1,
        embedder=None
    ):
        self.min_retrieval_score = min_retrieval_score
        self.min_relevant_chunks = min_relevant_chunks
        self.embedder = embedder

    def should_abstain(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        context: str = None
    ) -> Tuple[bool, str]:
        """
        Quyết định có nên abstain không.

        Returns:
            Tuple (should_abstain, reason)
        """
        # Check 1: No chunks retrieved
        if not retrieved_chunks:
            return True, "Không tìm thấy thông tin liên quan trong knowledge base"

        # Check 2: All chunks have low similarity
        similarities = [c.get("similarity", 0) for c in retrieved_chunks]
        max_sim = max(similarities) if similarities else 0

        if max_sim < self.min_retrieval_score:
            return True, f"Độ liên quan cao nhất ({max_sim:.2f}) dưới ngưỡng ({self.min_retrieval_score})"

        # Check 3: Too few relevant chunks
        relevant_count = sum(1 for s in similarities if s >= self.min_retrieval_score)
        if relevant_count < self.min_relevant_chunks:
            return True, f"Chỉ có {relevant_count} chunk liên quan (cần ít nhất {self.min_relevant_chunks})"

        # Check 4: Context too short
        if context and len(context) < 50:
            return True, "Context quá ngắn để đưa ra câu trả lời tin cậy"

        return False, "Đủ điều kiện để trả lời"

    def get_abstention_response(self, reason: str, question: str) -> str:
        """Generate appropriate abstention response"""

        templates = [
            f"Xin lỗi, tôi không tìm thấy thông tin đủ tin cậy để trả lời câu hỏi này.\n\nLý do: {reason}\n\nBạn có thể thử:\n- Đặt câu hỏi cụ thể hơn\n- Kiểm tra xem tài liệu liên quan đã được upload chưa",

            f"Tôi không thể trả lời câu hỏi này vì:\n{reason}\n\nĐể tôi có thể hỗ trợ tốt hơn, vui lòng cung cấp thêm tài liệu liên quan hoặc diễn đạt câu hỏi theo cách khác.",
        ]

        return templates[0]


# Convenience function
def verify_rag_answer(
    answer: str,
    context: str,
    question: str,
    sources: List[Dict] = None,
    min_confidence: float = 0.5
) -> VerificationResult:
    """
    Quick function để verify RAG answer.

    Usage:
        result = verify_rag_answer(answer, context, question)
        if result.should_abstain:
            answer = "Không có đủ thông tin..."
    """
    verifier = AnswerVerifier(min_confidence=min_confidence)
    return verifier.verify(answer, context, question, sources)


def should_abstain_from_answering(
    retrieved_chunks: List[Dict],
    min_score: float = 0.3
) -> Tuple[bool, str]:
    """
    Quick function để check abstention.

    Usage:
        abstain, reason = should_abstain_from_answering(chunks)
        if abstain:
            return generate_abstention_response(reason)
    """
    checker = AbstentionChecker(min_retrieval_score=min_score)
    return checker.should_abstain("", retrieved_chunks)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Answer Verification Module")
    print("=" * 60)

    verifier = AnswerVerifier()

    # Test case 1: Well-grounded answer
    context1 = """
    Điểm tối thiểu để pass môn học là 5.0 điểm.
    Sinh viên cần đăng ký môn học trước ngày 15 hàng tháng.
    Học phí phải đóng đầy đủ trước khi đăng ký.
    """

    answer1 = "Điểm tối thiểu để pass là 5.0 điểm. Sinh viên phải đăng ký trước ngày 15."

    result1 = verifier.verify(answer1, context1, "Điểm tối thiểu là bao nhiêu?")
    print(f"\nTest 1 - Grounded answer:")
    print(f"  Grounding: {result1.grounding_level.value}")
    print(f"  Confidence: {result1.confidence_score:.2f}")
    print(f"  Should abstain: {result1.should_abstain}")

    # Test case 2: Potentially hallucinated
    answer2 = "Điểm tối thiểu là 4.0 điểm theo quy định năm 2020. Có thể sẽ thay đổi."

    result2 = verifier.verify(answer2, context1, "Điểm tối thiểu là bao nhiêu?")
    print(f"\nTest 2 - Potentially hallucinated:")
    print(f"  Grounding: {result2.grounding_level.value}")
    print(f"  Confidence: {result2.confidence_score:.2f}")
    print(f"  Ungrounded claims: {result2.ungrounded_claims}")
    print(f"  Warnings: {result2.warnings}")

    # Test case 3: No info response
    answer3 = "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."

    result3 = verifier.verify(answer3, context1, "Thời gian nghỉ hè là khi nào?")
    print(f"\nTest 3 - No info response:")
    print(f"  Grounding: {result3.grounding_level.value}")
    print(f"  Action: {result3.suggested_action}")

    print("\n" + "=" * 60)
    print("Answer Verification Module ready!")
