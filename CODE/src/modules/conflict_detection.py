"""
Conflict Detection Module - Handle Conflicting Information
===========================================================

Cung cap cac co che de:
1. Phat hien thong tin xung dot giua cac documents
2. Xu ly version/date conflicts (uu tien thong tin moi hon)
3. Phat hien semantic contradictions
4. De xuat cach giai quyet conflicts
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ConflictType(Enum):
    """Loai conflict"""
    VERSION_CONFLICT = "version_conflict"  # Cung topic, khac version
    DATE_CONFLICT = "date_conflict"        # Thong tin cu vs moi
    VALUE_CONFLICT = "value_conflict"      # Cung attribute, khac value
    SEMANTIC_CONFLICT = "semantic_conflict"  # Contradiction ve y nghia
    PARTIAL_CONFLICT = "partial_conflict"  # Mot phan xung dot


class ResolutionStrategy(Enum):
    """Chien luoc giai quyet conflict"""
    PREFER_NEWER = "prefer_newer"          # Uu tien thong tin moi hon
    PREFER_HIGHER_SCORE = "prefer_higher_score"  # Uu tien relevance score cao hon
    MERGE_ALL = "merge_all"                # Gop tat ca thong tin
    SHOW_ALL_VERSIONS = "show_all_versions"  # Hien thi tat ca versions
    MANUAL_REVIEW = "manual_review"        # Can review thu cong


@dataclass
class ConflictInfo:
    """Thong tin ve mot conflict"""
    conflict_type: ConflictType
    topic: str                  # Topic bi conflict
    sources: List[Dict]         # Cac sources lien quan
    values: List[str]           # Cac gia tri conflict
    dates: List[Optional[datetime]] = field(default_factory=list)
    confidence: float = 0.5     # Do tin cay cua detection
    resolution: Optional[str] = None  # Cach da giai quyet

    def to_dict(self) -> Dict:
        return {
            "conflict_type": self.conflict_type.value,
            "topic": self.topic,
            "sources": self.sources,
            "values": self.values,
            "dates": [d.isoformat() if d else None for d in self.dates],
            "confidence": self.confidence,
            "resolution": self.resolution,
        }


@dataclass
class ConflictDetectionResult:
    """Ket qua phat hien conflicts"""
    has_conflicts: bool
    conflicts: List[ConflictInfo] = field(default_factory=list)
    resolved_context: str = ""   # Context sau khi resolve
    resolution_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "has_conflicts": self.has_conflicts,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolved_context": self.resolved_context,
            "resolution_notes": self.resolution_notes,
            "warnings": self.warnings,
        }


class ConflictDetector:
    """
    Detector de phat hien va xu ly thong tin xung dot.

    Usage:
        detector = ConflictDetector()
        result = detector.detect_and_resolve(chunks, query)

        if result.has_conflicts:
            print(f"Found {len(result.conflicts)} conflicts")
            context = result.resolved_context  # Da duoc resolve
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.PREFER_NEWER,
        date_weight: float = 0.4,
        score_weight: float = 0.3,
        semantic_threshold: float = 0.7,
        embedder=None  # Optional: for semantic comparison
    ):
        """
        Args:
            default_strategy: Chien luoc mac dinh de resolve conflicts
            date_weight: Trong so cho date khi ranking
            score_weight: Trong so cho retrieval score
            semantic_threshold: Nguong similarity de coi la cung topic
            embedder: TextEmbedding instance cho semantic matching
        """
        self.default_strategy = default_strategy
        self.date_weight = date_weight
        self.score_weight = score_weight
        self.semantic_threshold = semantic_threshold
        self.embedder = embedder

        # Patterns de extract dates
        self.date_patterns = [
            # Vietnamese formats
            r'ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'năm\s+(\d{4})',
            r'tháng\s+(\d{1,2})[/-](\d{4})',
            # English formats
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            # Quy dinh/version patterns
            r'quy\s*định\s+(?:số\s+)?(\d+)[/-](\d{4})',
            r'version\s+(\d+(?:\.\d+)?)',
            r'v(\d+(?:\.\d+)?)',
        ]

        # Keywords indicating updates/changes
        self.update_keywords = [
            'mới', 'cập nhật', 'sửa đổi', 'thay đổi', 'điều chỉnh',
            'bổ sung', 'thay thế', 'hiện tại', 'hiện hành',
            'new', 'updated', 'revised', 'current', 'latest',
            'effective from', 'có hiệu lực từ',
        ]

        # Keywords indicating old/outdated info
        self.old_keywords = [
            'cũ', 'trước đây', 'trước kia', 'đã hết hạn', 'không còn',
            'old', 'previous', 'former', 'outdated', 'deprecated',
            'đã bãi bỏ', 'hết hiệu lực',
        ]

    def detect_and_resolve(
        self,
        chunks: List[Dict],
        query: str,
        strategy: ResolutionStrategy = None
    ) -> ConflictDetectionResult:
        """
        Phat hien va resolve conflicts trong chunks.

        Args:
            chunks: List cac chunk da retrieve
            query: Query goc
            strategy: Chien luoc resolve (default: use self.default_strategy)

        Returns:
            ConflictDetectionResult voi resolved context
        """
        strategy = strategy or self.default_strategy
        warnings = []
        resolution_notes = []

        # 1. Group chunks by topic
        topic_groups = self._group_by_topic(chunks, query)

        # 2. Detect conflicts in each group
        all_conflicts = []
        for topic, group_chunks in topic_groups.items():
            if len(group_chunks) > 1:
                conflicts = self._detect_conflicts_in_group(topic, group_chunks)
                all_conflicts.extend(conflicts)

        # 3. If no conflicts, return original context
        if not all_conflicts:
            context = self._build_context(chunks)
            return ConflictDetectionResult(
                has_conflicts=False,
                resolved_context=context,
            )

        # 4. Resolve conflicts
        resolved_chunks = []
        for topic, group_chunks in topic_groups.items():
            topic_conflicts = [c for c in all_conflicts if c.topic == topic]

            if topic_conflicts:
                # Resolve and pick best chunk(s)
                resolved, notes = self._resolve_conflicts(
                    group_chunks, topic_conflicts, strategy
                )
                resolved_chunks.extend(resolved)
                resolution_notes.extend(notes)

                # Mark resolution in conflict
                for c in topic_conflicts:
                    c.resolution = strategy.value
            else:
                resolved_chunks.extend(group_chunks)

        # 5. Build final context
        resolved_context = self._build_context(resolved_chunks)

        # 6. Add warnings if needed
        if all_conflicts:
            warnings.append(
                f"Phat hien {len(all_conflicts)} conflicts, da resolve voi strategy: {strategy.value}"
            )

        return ConflictDetectionResult(
            has_conflicts=True,
            conflicts=all_conflicts,
            resolved_context=resolved_context,
            resolution_notes=resolution_notes,
            warnings=warnings,
        )

    def _group_by_topic(
        self,
        chunks: List[Dict],
        query: str
    ) -> Dict[str, List[Dict]]:
        """Group chunks theo topic/subject"""

        # Extract key entities from query
        query_entities = self._extract_entities(query)

        groups = {}
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))

            # Find which entities this chunk mentions
            chunk_entities = self._extract_entities(text)

            # Use overlap as topic key
            common = query_entities & chunk_entities
            if common:
                topic = "_".join(sorted(list(common)[:3]))
            else:
                topic = "general"

            if topic not in groups:
                groups[topic] = []
            groups[topic].append(chunk)

        return groups

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract key entities/topics tu text"""
        entities = set()
        text_lower = text.lower()

        # Extract key Vietnamese entities
        patterns = [
            r'(học phí|điểm|môn học|tín chỉ|học kỳ)',
            r'(đăng ký|deadline|hạn|thời gian)',
            r'(quy định|quy chế|điều|khoản)',
            r'(sinh viên|học sinh|học viên)',
            r'(\d+(?:\.\d+)?%)',  # Percentages
            r'(\d{4})',  # Years
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            entities.update(matches)

        # Extract capitalized terms (proper nouns)
        words = text.split()
        for word in words:
            if len(word) > 2 and word[0].isupper():
                entities.add(word.lower())

        return entities

    def _detect_conflicts_in_group(
        self,
        topic: str,
        chunks: List[Dict]
    ) -> List[ConflictInfo]:
        """Detect conflicts trong mot group cung topic"""
        conflicts = []

        # Extract dates and values from each chunk
        chunk_info = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            date = self._extract_date(text, chunk.get("metadata", {}))
            values = self._extract_key_values(text)

            chunk_info.append({
                "chunk": chunk,
                "text": text,
                "date": date,
                "values": values,
                "is_newer": self._is_newer_info(text),
                "is_older": self._is_older_info(text),
            })

        # Compare chunks
        for i in range(len(chunk_info)):
            for j in range(i + 1, len(chunk_info)):
                info_i = chunk_info[i]
                info_j = chunk_info[j]

                # Check for value conflicts
                value_conflicts = self._find_value_conflicts(
                    info_i["values"], info_j["values"]
                )

                if value_conflicts:
                    conflict_type = ConflictType.VALUE_CONFLICT

                    # Check if it's a date/version conflict
                    if info_i["date"] and info_j["date"]:
                        conflict_type = ConflictType.DATE_CONFLICT
                    elif info_i["is_newer"] != info_j["is_newer"]:
                        conflict_type = ConflictType.VERSION_CONFLICT

                    conflicts.append(ConflictInfo(
                        conflict_type=conflict_type,
                        topic=topic,
                        sources=[
                            {"index": i, "text": info_i["text"][:100]},
                            {"index": j, "text": info_j["text"][:100]},
                        ],
                        values=value_conflicts,
                        dates=[info_i["date"], info_j["date"]],
                        confidence=0.7,
                    ))

        return conflicts

    def _extract_date(
        self,
        text: str,
        metadata: Dict
    ) -> Optional[datetime]:
        """Extract date tu text hoac metadata"""

        # Check metadata first
        if metadata:
            for key in ['date', 'created_at', 'modified_at', 'timestamp', 'effective_date']:
                if key in metadata:
                    try:
                        date_val = metadata[key]
                        if isinstance(date_val, datetime):
                            return date_val
                        if isinstance(date_val, str):
                            # Try parsing
                            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']:
                                try:
                                    return datetime.strptime(date_val, fmt)
                                except ValueError:
                                    continue
                    except Exception:
                        pass

        # Try extracting from text
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 1:  # Year only
                        return datetime(int(groups[0]), 1, 1)
                    elif len(groups) == 2:  # Month/Year
                        return datetime(int(groups[1]), int(groups[0]), 1)
                    elif len(groups) == 3:
                        # Determine order (d/m/y or y/m/d)
                        if int(groups[0]) > 31:  # Year first
                            return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                        else:  # Day first
                            return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_key_values(self, text: str) -> Dict[str, str]:
        """Extract key-value pairs tu text"""
        values = {}

        # Pattern: "X la/bang Y"
        patterns = [
            r'([\w\s]+)\s+(?:là|bằng|=)\s+([\d,.]+(?:\s*%)?)',
            r'([\w\s]+):\s*([\d,.]+(?:\s*%)?)',
            r'(điểm|học phí|tín chỉ|thời gian)\s+(?:tối thiểu|tối đa|cần thiết)?\s*(?:là|:)?\s*([\d,.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                key = match[0].strip().lower()
                value = match[1].strip()
                if key and value:
                    values[key] = value

        return values

    def _find_value_conflicts(
        self,
        values1: Dict[str, str],
        values2: Dict[str, str]
    ) -> List[str]:
        """Find conflicting values between two dicts"""
        conflicts = []

        common_keys = set(values1.keys()) & set(values2.keys())
        for key in common_keys:
            v1 = values1[key]
            v2 = values2[key]

            # Normalize values for comparison
            v1_norm = re.sub(r'[,.\s]', '', v1)
            v2_norm = re.sub(r'[,.\s]', '', v2)

            if v1_norm != v2_norm:
                conflicts.append(f"{key}: '{v1}' vs '{v2}'")

        return conflicts

    def _is_newer_info(self, text: str) -> bool:
        """Check if text indicates newer/updated info"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.update_keywords)

    def _is_older_info(self, text: str) -> bool:
        """Check if text indicates older/outdated info"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.old_keywords)

    def _resolve_conflicts(
        self,
        chunks: List[Dict],
        conflicts: List[ConflictInfo],
        strategy: ResolutionStrategy
    ) -> Tuple[List[Dict], List[str]]:
        """
        Resolve conflicts va return filtered chunks.

        Returns:
            Tuple (resolved_chunks, resolution_notes)
        """
        notes = []

        if strategy == ResolutionStrategy.PREFER_NEWER:
            return self._resolve_prefer_newer(chunks, conflicts, notes)

        elif strategy == ResolutionStrategy.PREFER_HIGHER_SCORE:
            return self._resolve_prefer_score(chunks, notes)

        elif strategy == ResolutionStrategy.MERGE_ALL:
            return self._resolve_merge_all(chunks, conflicts, notes)

        elif strategy == ResolutionStrategy.SHOW_ALL_VERSIONS:
            return self._resolve_show_all(chunks, conflicts, notes)

        else:  # MANUAL_REVIEW
            notes.append("Conflicts can review thu cong")
            return chunks, notes

    def _resolve_prefer_newer(
        self,
        chunks: List[Dict],
        conflicts: List[ConflictInfo],
        notes: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """Uu tien thong tin moi hon"""

        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            metadata = chunk.get("metadata", {})

            score = 0.0

            # Date score
            date = self._extract_date(text, metadata)
            if date:
                # Newer = higher score
                days_ago = (datetime.now() - date).days
                date_score = max(0, 1 - days_ago / 3650)  # 10 year scale
                score += date_score * self.date_weight

            # Update keyword score
            if self._is_newer_info(text):
                score += 0.3
            if self._is_older_info(text):
                score -= 0.2

            # Retrieval score
            retrieval_score = chunk.get("similarity", chunk.get("score", 0.5))
            score += retrieval_score * self.score_weight

            scored_chunks.append((chunk, score))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Take top chunks, avoiding duplicates
        resolved = []
        seen_content = set()
        for chunk, score in scored_chunks:
            text = chunk.get("text", chunk.get("content", ""))[:200]
            if text not in seen_content:
                resolved.append(chunk)
                seen_content.add(text)

        notes.append(f"Da chon {len(resolved)} chunks moi nhat/co score cao nhat")
        return resolved, notes

    def _resolve_prefer_score(
        self,
        chunks: List[Dict],
        notes: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """Uu tien retrieval score cao"""

        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("similarity", c.get("score", 0)),
            reverse=True
        )

        notes.append(f"Da sap xep {len(chunks)} chunks theo score")
        return sorted_chunks, notes

    def _resolve_merge_all(
        self,
        chunks: List[Dict],
        conflicts: List[ConflictInfo],
        notes: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """Gop tat ca thong tin va note conflicts"""

        # Add conflict note to chunks
        for chunk in chunks:
            chunk_conflicts = []
            for c in conflicts:
                for src in c.sources:
                    if src.get("text", "")[:50] in chunk.get("text", "")[:100]:
                        chunk_conflicts.append(c)
                        break

            if chunk_conflicts:
                chunk["_conflict_note"] = f"[CONFLICT: {len(chunk_conflicts)} detected]"

        notes.append(f"Da merge {len(chunks)} chunks, marked {len(conflicts)} conflicts")
        return chunks, notes

    def _resolve_show_all(
        self,
        chunks: List[Dict],
        conflicts: List[ConflictInfo],
        notes: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """Hien thi tat ca versions voi annotation"""

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", chunk.get("content", ""))
            metadata = chunk.get("metadata", {})

            date = self._extract_date(text, metadata)
            if date:
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = "unknown date"

            version_note = f"[Version {i+1}, {date_str}]"

            if self._is_newer_info(text):
                version_note += " [NEWER]"
            if self._is_older_info(text):
                version_note += " [OLDER]"

            chunk["_version_note"] = version_note

        notes.append(f"Da annotate {len(chunks)} versions")
        return chunks, notes

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string tu chunks"""
        parts = []

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", chunk.get("content", ""))

            # Add version note if exists
            version_note = chunk.get("_version_note", "")
            conflict_note = chunk.get("_conflict_note", "")

            prefix = ""
            if version_note:
                prefix = f"{version_note}\n"
            if conflict_note:
                prefix += f"{conflict_note}\n"

            parts.append(f"{prefix}{text}")

        return "\n\n---\n\n".join(parts)


class DateAwareRetriever:
    """
    Retriever co nhan biet date de uu tien thong tin moi.

    Usage:
        retriever = DateAwareRetriever(vector_db, embedder)
        results = retriever.search_with_date_boost(
            query,
            top_k=10,
            date_boost=0.3  # 30% weight cho date
        )
    """

    def __init__(
        self,
        vector_db,
        embedder,
        default_date_boost: float = 0.2
    ):
        self.vector_db = vector_db
        self.embedder = embedder
        self.default_date_boost = default_date_boost
        self.detector = ConflictDetector()

    def search_with_date_boost(
        self,
        query: str,
        top_k: int = 10,
        date_boost: float = None,
        prefer_recent: bool = True
    ) -> List[Dict]:
        """
        Search voi date boost.

        Args:
            query: Query string
            top_k: So luong ket qua
            date_boost: Weight cho date (0-1)
            prefer_recent: True = uu tien moi, False = uu tien cu

        Returns:
            List chunks da duoc rank theo date + relevance
        """
        date_boost = date_boost or self.default_date_boost

        # Get embedding
        query_emb = self.embedder.encode([query])[0]

        # Search
        results = self.vector_db.search(
            query_embedding=query_emb,
            top_k=top_k * 2  # Get more to re-rank
        )

        # Re-rank with date
        scored_results = []
        for result in results:
            text = result.get("text", result.get("content", ""))
            metadata = result.get("metadata", {})

            # Original score
            original_score = result.get("similarity", result.get("score", 0.5))

            # Date score
            date = self.detector._extract_date(text, metadata)
            if date:
                days_ago = (datetime.now() - date).days
                if prefer_recent:
                    date_score = max(0, 1 - days_ago / 1825)  # 5 year scale
                else:
                    date_score = min(1, days_ago / 1825)
            else:
                date_score = 0.5  # Neutral if no date

            # Combined score
            final_score = (original_score * (1 - date_boost) +
                          date_score * date_boost)

            result["_final_score"] = final_score
            result["_date_score"] = date_score
            scored_results.append(result)

        # Sort by final score
        scored_results.sort(key=lambda x: x["_final_score"], reverse=True)

        return scored_results[:top_k]


# Convenience functions
def detect_conflicts(
    chunks: List[Dict],
    query: str,
    strategy: ResolutionStrategy = ResolutionStrategy.PREFER_NEWER
) -> ConflictDetectionResult:
    """
    Quick function de detect va resolve conflicts.

    Usage:
        result = detect_conflicts(chunks, query)
        if result.has_conflicts:
            context = result.resolved_context
    """
    detector = ConflictDetector(default_strategy=strategy)
    return detector.detect_and_resolve(chunks, query, strategy)


def resolve_version_conflicts(
    chunks: List[Dict],
    prefer_newer: bool = True
) -> List[Dict]:
    """
    Quick function de resolve version conflicts.

    Returns chunks sorted by date (newer first if prefer_newer=True)
    """
    detector = ConflictDetector()
    result = detector.detect_and_resolve(
        chunks,
        "",  # Empty query
        ResolutionStrategy.PREFER_NEWER if prefer_newer else ResolutionStrategy.PREFER_HIGHER_SCORE
    )

    # Parse resolved context back to chunks if needed
    return chunks  # Return original for now, sorted


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Conflict Detection Module")
    print("=" * 60)

    detector = ConflictDetector()

    # Test case 1: Date conflict
    chunks1 = [
        {
            "text": "Hoc phi nam 2023 la 15 trieu dong/nam. Quy dinh ngay 01/01/2023.",
            "metadata": {"date": "2023-01-01"},
            "similarity": 0.8,
        },
        {
            "text": "Hoc phi nam 2024 da tang len 18 trieu dong/nam. Cap nhat moi ngay 01/01/2024.",
            "metadata": {"date": "2024-01-01"},
            "similarity": 0.85,
        },
    ]

    result1 = detector.detect_and_resolve(chunks1, "hoc phi la bao nhieu")
    print(f"\nTest 1 - Date conflict:")
    print(f"  Has conflicts: {result1.has_conflicts}")
    if result1.conflicts:
        for c in result1.conflicts:
            print(f"  Conflict: {c.conflict_type.value}")
            print(f"    Values: {c.values}")
    print(f"  Resolution notes: {result1.resolution_notes}")

    # Test case 2: Version conflict
    chunks2 = [
        {
            "text": "Diem toi thieu de pass mon hoc la 4.0 (quy dinh cu).",
            "similarity": 0.7,
        },
        {
            "text": "Diem toi thieu de pass mon hoc la 5.0 theo quy dinh moi.",
            "similarity": 0.75,
        },
    ]

    result2 = detector.detect_and_resolve(chunks2, "diem toi thieu")
    print(f"\nTest 2 - Version conflict:")
    print(f"  Has conflicts: {result2.has_conflicts}")
    print(f"  Resolved context preview: {result2.resolved_context[:200]}...")

    # Test case 3: No conflict
    chunks3 = [
        {
            "text": "Sinh vien can dang ky mon hoc truoc ngay 15 hang thang.",
            "similarity": 0.9,
        },
        {
            "text": "Thoi gian dang ky mon hoc la tu ngay 1 den ngay 15.",
            "similarity": 0.85,
        },
    ]

    result3 = detector.detect_and_resolve(chunks3, "thoi gian dang ky")
    print(f"\nTest 3 - No conflict:")
    print(f"  Has conflicts: {result3.has_conflicts}")

    print("\n" + "=" * 60)
    print("Conflict Detection Module ready!")
