from abc import ABC, abstractmethod
from typing import List

from app.core import models


class DocumentRepositoryPort(ABC):
    @abstractmethod
    def save_document(self, document: models.Document, content: str, openai_client) -> None:
        pass

    @abstractmethod
    def get_documents(self, query: str, openai_client, n_results: int | None = None) -> List[models.Document]:
        pass


class LlmPort(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, retrieval_context: str) -> str:
        pass

#Falta para usuarios
class DatabasePort(ABC):
    @abstractmethod
    def save_document(self, document: models.Document) -> None:
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> models.Document:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        pass