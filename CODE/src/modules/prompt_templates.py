"""
Prompt Templates Module - Cac prompt templates toi uu cho RAG
Ho tro:
- Vietnamese prompts
- English prompts
- Multi-turn conversation prompts
- Specialized prompts (QA, Summary, Analysis)
- Anti-hallucination prompts (Strict citation required)
- Conflict handling prompts
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template cho 1 prompt"""
    name: str
    description: str
    system_prompt: str
    user_prompt: str
    variables: List[str]


class PromptTemplateManager:
    """Manager quan ly cac prompt templates"""

    def __init__(self, language: str = "vi"):
        self.language = language
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        if self.language == "vi":
            self._load_vietnamese_templates()
        else:
            self._load_english_templates()

    def _load_vietnamese_templates(self):
        """Load Vietnamese templates"""

        self.templates["basic_qa"] = PromptTemplate(
            name="basic_qa",
            description="Template co ban cho QA",
            system_prompt="Ban la tro ly AI thong minh. Chi tra loi dua tren context. Neu khong tim thay, noi khong co thong tin.",
            user_prompt="Context:\n{context}\n\nCau hoi: {question}\n\nTra loi:",
            variables=["context", "question"]
        )

        self.templates["audio_qa"] = PromptTemplate(
            name="audio_qa",
            description="Template cho QA tren audio transcript",
            system_prompt="Ban la tro ly AI xu ly audio. Luon trich dan moc thoi gian [MM:SS] khi tra loi.",
            user_prompt="Ban ghi audio:\n{context}\n\nCau hoi: {question}\n\nTra loi:",
            variables=["context", "question"]
        )

        self.templates["factual_qa"] = PromptTemplate(
            name="factual_qa",
            description="Template QA nghiem ngat",
            system_prompt="Chi tra loi tu context. Khong suy doan. Neu khong co -> 'Khong co thong tin'.",
            user_prompt="[CONTEXT]\n{context}\n\n[CAU HOI]\n{question}\n\n[TRA LOI]",
            variables=["context", "question"]
        )

        self.templates["cot_qa"] = PromptTemplate(
            name="cot_qa",
            description="Chain of Thought QA",
            system_prompt="Suy luan tung buoc truoc khi tra loi.",
            user_prompt="Context:\n{context}\n\nCau hoi: {question}\n\nSuy luan:\n1. Thong tin lien quan:\n2. Phan tich:\n3. Ket luan:",
            variables=["context", "question"]
        )

        # === ANTI-HALLUCINATION TEMPLATES ===
        self.templates["strict_qa"] = PromptTemplate(
            name="strict_qa",
            description="Template chong hallucination - bat buoc trich dan nguon",
            system_prompt="""Ban la tro ly AI tra loi cau hoi dua tren tai lieu. Chi tra loi tu thong tin trong tai lieu. Neu khong tim thay, noi "Khong co thong tin".""",
            user_prompt="""Tai lieu:
{context}

Cau hoi: {question}

Tra loi (chi dua tren tai lieu):""",
            variables=["context", "question"]
        )

        self.templates["citation_required"] = PromptTemplate(
            name="citation_required",
            description="Template bat buoc co trich dan nguon chi tiet",
            system_prompt="""Ban la tro ly AI ho tro tra cuu tai lieu. NGUYEN TAC:

1. MOI cau tra loi PHAI co trich dan nguon cu the
2. Format trich dan: [Nguon: ten_file, trang/doan X]
3. Neu nhieu nguon -> liet ke tat ca nguon lien quan
4. Phan biet RO giua:
   - Thong tin TRUC TIEP tu tai lieu (co trich dan)
   - Suy luan/Ket luan tu thong tin (danh dau la "Suy luan:")
5. KHONG tra loi neu khong co nguon

Uu tien thong tin:
- Tai lieu MOI hon > cu hon
- Tai lieu CHINH THUC > khong chinh thuc
- Thong tin CU THE > chung chung""",
            user_prompt="""[CONTEXT - Cac tai lieu da truy xuat]
{context}

[CAU HOI CAN TRA LOI]
{question}

[DINH DANG TRA LOI]
Tra loi:
[Noi dung tra loi voi trich dan nguon]

Nguon tham khao:
- [Liet ke cac nguon da su dung]

[BAT DAU TRA LOI]""",
            variables=["context", "question"]
        )

        # === CONFLICT HANDLING TEMPLATES ===
        self.templates["conflict_aware"] = PromptTemplate(
            name="conflict_aware",
            description="Template xu ly thong tin xung dot/mau thuan",
            system_prompt="""Ban la tro ly AI xu ly thong tin co the co MAU THUAN. Quy tac:

1. Neu phat hien THONG TIN XUNG DOT (cung chu de, khac gia tri):
   - Neu co DATE: uu tien thong tin MOI HON
   - Neu khong co date: trinh bay CA HAI va neu ro su khac biet
   - Chi ro nguon cua moi thong tin

2. Format khi co xung dot:
   "[Luu y: Co thong tin khac nhau giua cac nguon]
    - Theo [Nguon A, date]: ...
    - Theo [Nguon B, date]: ...
    Thong tin moi nhat cho thay: ..."

3. UU TIEN thong tin:
   - Co date MOI > cu
   - Co chu "cap nhat", "moi", "hien hanh" > khong co
   - Tu tai lieu CHINH THUC > khong chinh thuc

4. Neu khong the xac dinh dau moi hon -> trinh bay tat ca va de nghi nguoi dung xac nhan""",
            user_prompt="""[TAI LIEU - Co the chua thong tin tu nhieu thoi diem khac nhau]
{context}

[CAU HOI]
{question}

[HUONG DAN]
- Kiem tra xem co thong tin mau thuan khong
- Neu co: uu tien thong tin moi hon va giai thich
- Neu khong the xac dinh: trinh bay tat ca versions

[TRA LOI]""",
            variables=["context", "question"]
        )

        self.templates["version_compare"] = PromptTemplate(
            name="version_compare",
            description="Template so sanh cac phien ban thong tin",
            system_prompt="""Ban la tro ly AI chuyen so sanh cac phien ban thong tin.

Nhiem vu:
1. Xac dinh cac PHIEN BAN khac nhau cua thong tin trong context
2. SO SANH va chi ra:
   - Diem GIONG nhau
   - Diem KHAC nhau
   - Version nao MOI HON (dua tren date, keyword)
3. Dua ra KET LUAN ve thong tin hien hanh

Format output:
## Phan tich phien ban
- Phien ban 1: [mo ta, date neu co]
- Phien ban 2: [mo ta, date neu co]

## So sanh
| Noi dung | Phien ban 1 | Phien ban 2 |
|----------|-------------|-------------|

## Ket luan
[Thong tin hien hanh, ly do]""",
            user_prompt="""[TAI LIEU]
{context}

[YEU CAU PHAN TICH]
{question}

[PHAN TICH]""",
            variables=["context", "question"]
        )

        # === ABSTENTION TEMPLATE ===
        self.templates["safe_abstention"] = PromptTemplate(
            name="safe_abstention",
            description="Template an toan - tu choi khi khong chac chan",
            system_prompt="""Ban la tro ly AI UU TIEN SU CHINH XAC hon la tra loi day du.

NGUYEN TAC VANG:
1. Chi tra loi khi DO TIN CAY >= 80%
2. Neu chi co 1 phan thong tin -> tra loi phan do, neu ro phan thieu
3. Neu KHONG CHAC CHAN -> tra loi:
   "Toi khong tim thay thong tin day du de tra loi cau hoi nay trong tai lieu.
    [Neu co thong tin lien quan: Co the lien quan: ...]
    [Goi y cau hoi khac hoac yeu cau them tai lieu]"

4. KHONG BAO GIO:
   - Doan hoac gia dinh thong tin
   - Tra loi tu "kien thuc chung" ngoai context
   - Bia so lieu, ngay thang, ten

5. Tu nhan biet gioi han va thong bao ro cho nguoi dung""",
            user_prompt="""[CONTEXT]
{context}

[CAU HOI]
{question}

[LUU Y]
- Neu khong tim thay: noi ro khong co thong tin
- Neu chi co 1 phan: tra loi phan co, neu ro phan thieu
- Khong doan mo

[TRA LOI]""",
            variables=["context", "question"]
        )

    def _load_english_templates(self):
        """Load English templates"""

        self.templates["basic_qa"] = PromptTemplate(
            name="basic_qa",
            description="Basic QA template",
            system_prompt="You are a helpful AI. Answer only from context. If not found, say so.",
            user_prompt="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            variables=["context", "question"]
        )

        self.templates["audio_qa"] = PromptTemplate(
            name="audio_qa",
            description="Audio transcript QA",
            system_prompt="You are an AI for audio transcripts. Always cite timestamps [MM:SS].",
            user_prompt="Transcript:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            variables=["context", "question"]
        )

        self.templates["factual_qa"] = PromptTemplate(
            name="factual_qa",
            description="Strict factual QA",
            system_prompt="Answer ONLY from context. No inference. If not found -> 'No information'.",
            user_prompt="[CONTEXT]\n{context}\n\n[QUESTION]\n{question}\n\n[ANSWER]",
            variables=["context", "question"]
        )

        self.templates["cot_qa"] = PromptTemplate(
            name="cot_qa",
            description="Chain of Thought QA",
            system_prompt="Reason step by step before answering.",
            user_prompt="Context:\n{context}\n\nQuestion: {question}\n\nReasoning:\n1. Relevant info:\n2. Analysis:\n3. Conclusion:",
            variables=["context", "question"]
        )

        # === ANTI-HALLUCINATION TEMPLATES (English) ===
        self.templates["strict_qa"] = PromptTemplate(
            name="strict_qa",
            description="Anti-hallucination template - citation required",
            system_prompt="""You are a STRICT AI assistant. MANDATORY rules:

1. ONLY answer from information IN the provided context
2. EVERY piece of information MUST have a citation [Source X]
3. DO NOT infer, assume, or add external information
4. If information NOT FOUND -> answer EXACTLY: "No information found in the documents to answer this question."
5. If information is INCOMPLETE -> only answer what's available, clearly state what's missing

ABSOLUTELY DO NOT:
- Make up information
- Speculate or guess
- Answer from general knowledge
- Give vague answers when uncertain""",
            user_prompt="""[REFERENCE DOCUMENTS]
{context}

[QUESTION]
{question}

[ANSWER REQUIREMENTS]
- Cite source for every piece of information: [Source: filename or section]
- If not found: clearly state "No information found in documents"
- Only answer what IS in the context

[ANSWER]""",
            variables=["context", "question"]
        )

        self.templates["citation_required"] = PromptTemplate(
            name="citation_required",
            description="Template requiring detailed citations",
            system_prompt="""You are an AI assistant for document retrieval. PRINCIPLES:

1. EVERY answer MUST have specific citations
2. Citation format: [Source: filename, page/section X]
3. If multiple sources -> list all relevant sources
4. DISTINGUISH clearly between:
   - DIRECT information from documents (with citation)
   - Inference/Conclusions (marked as "Inference:")
5. DO NOT answer without sources

Information priority:
- NEWER documents > older
- OFFICIAL documents > unofficial
- SPECIFIC information > general""",
            user_prompt="""[CONTEXT - Retrieved documents]
{context}

[QUESTION TO ANSWER]
{question}

[ANSWER FORMAT]
Answer:
[Content with citations]

References:
- [List sources used]

[BEGIN ANSWER]""",
            variables=["context", "question"]
        )

        # === CONFLICT HANDLING TEMPLATES (English) ===
        self.templates["conflict_aware"] = PromptTemplate(
            name="conflict_aware",
            description="Template handling conflicting information",
            system_prompt="""You are an AI assistant handling potentially CONFLICTING information. Rules:

1. When CONFLICTING INFO detected (same topic, different values):
   - If dates available: prefer NEWER information
   - If no dates: present BOTH and note the difference
   - Clearly cite source of each piece of info

2. Format for conflicts:
   "[Note: Conflicting information between sources]
    - According to [Source A, date]: ...
    - According to [Source B, date]: ...
    Latest information shows: ..."

3. PRIORITY order:
   - Has NEWER date > older
   - Has "updated", "new", "current" keywords > without
   - From OFFICIAL document > unofficial

4. If cannot determine which is newer -> present all and ask user to confirm""",
            user_prompt="""[DOCUMENTS - May contain information from different time periods]
{context}

[QUESTION]
{question}

[INSTRUCTIONS]
- Check for conflicting information
- If conflicts: prefer newer info and explain
- If undetermined: present all versions

[ANSWER]""",
            variables=["context", "question"]
        )

        self.templates["safe_abstention"] = PromptTemplate(
            name="safe_abstention",
            description="Safe template - refuse when uncertain",
            system_prompt="""You are an AI assistant that PRIORITIZES ACCURACY over completeness.

GOLDEN RULES:
1. Only answer when CONFIDENCE >= 80%
2. If only partial info available -> answer that part, note what's missing
3. If UNCERTAIN -> respond:
   "I cannot find sufficient information in the documents to answer this question.
    [If related info exists: May be related: ...]
    [Suggest alternative question or request more documents]"

4. NEVER:
   - Guess or assume information
   - Answer from "general knowledge" outside context
   - Make up numbers, dates, names

5. Self-aware of limitations and clearly inform user""",
            user_prompt="""[CONTEXT]
{context}

[QUESTION]
{question}

[NOTE]
- If not found: clearly state no information
- If partial: answer available part, note missing
- Do not guess

[ANSWER]""",
            variables=["context", "question"]
        )

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Lay template theo ten"""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """Liet ke tat ca templates"""
        return list(self.templates.keys())

    def add_template(self, template: PromptTemplate) -> None:
        """Them template moi"""
        self.templates[template.name] = template

    def format_prompt(
        self,
        template_name: str,
        context: str,
        question: str,
        **kwargs
    ) -> tuple:
        """
        Format prompt tu template

        Returns:
            Tuple (system_prompt, user_prompt)
        """
        template = self.get_template(template_name)
        if not template:
            template = self.get_template("basic_qa")

        variables = {"context": context, "question": question, **kwargs}
        user_prompt = template.user_prompt.format(**variables)

        return template.system_prompt, user_prompt


# Convenience function
def get_rag_prompt(
    context: str,
    question: str,
    template: str = "audio_qa",
    language: str = "vi"
) -> tuple:
    """
    Quick function de lay formatted RAG prompt

    Args:
        context: Retrieved context
        question: User question
        template: Template name
        language: "vi" or "en"

    Returns:
        Tuple (system_prompt, user_prompt)
    """
    manager = PromptTemplateManager(language=language)
    return manager.format_prompt(template, context, question)


# Test
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Prompt Templates")
    print("=" * 50)

    # Vietnamese
    manager_vi = PromptTemplateManager(language="vi")
    print(f"\nVietnamese templates: {manager_vi.list_templates()}")

    sys_prompt, user_prompt = manager_vi.format_prompt(
        "audio_qa",
        context="[00:15-00:30] Machine learning la mot nhanh cua AI.",
        question="Machine learning la gi?"
    )
    print(f"\nSystem: {sys_prompt}")
    print(f"User: {user_prompt}")

    # English
    manager_en = PromptTemplateManager(language="en")
    print(f"\nEnglish templates: {manager_en.list_templates()}")

    print("\nPrompt Templates Module ready!")
