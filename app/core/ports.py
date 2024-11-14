# app/core/ports.py
from abc import ABC, abstractmethod
from typing import List, Optional
from app.core import models
import openai


class DocumentRepositoryPort(ABC):
    @abstractmethod
    def save_document(
        self,
        document: models.Document,
        content: str,
        openai_client: Optional[openai.OpenAI],
    ) -> None:
        pass

    @abstractmethod
    def get_documents(
        self,
        query: str,
        openai_client: Optional[openai.OpenAI],
        n_results: int | None = None,
    ) -> List[models.Document]:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[models.Document]:
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> models.Document:
        pass


class LlmPort(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, retrieval_context: str) -> str:
        pass
