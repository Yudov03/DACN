"""
Prompt Templates Module - Cac prompt templates toi uu cho RAG
Ho tro:
- Vietnamese prompts
- English prompts
- Multi-turn conversation prompts
- Specialized prompts (QA, Summary, Analysis)
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
