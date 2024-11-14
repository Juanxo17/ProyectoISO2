# app/usecases.py
from PyPDF2 import PdfReader
from app.core.models import Document
from app.core.ports import DocumentRepositoryPort, LlmPort
from fastapi import UploadFile
import os
from typing import List


class RAGService:
    def __init__(
        self, document_repo: DocumentRepositoryPort, openai_adapter: LlmPort
    ) -> None:
        self.document_repo = document_repo
        self.openai_adapter = openai_adapter

    def generate_answer(self, query: str) -> str:
        documents = self.document_repo.get_documents(
            query, self.openai_adapter, n_results=3
        )
        print(f"Documents: {documents}")
        context = " ".join([doc.content for doc in documents])
        return self.openai_adapter.generate_text(
            prompt=query, retrieval_context=context
        )

    def save_document(self, file: UploadFile) -> None:
        file_nombre = file.filename
        os.makedirs("Library", exist_ok=True)
        file_location = os.path.join("Library", file_nombre)
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        document = Document(
            nombre=file_nombre, path=file_location, content="", user_id="test"
        )
        pdfReader = PdfReader(file_location)
        content = ""
        for page in range(len(pdfReader.pages)):
            content += pdfReader.pages[page].extract_text() or ""
        if not content:
            raise ValueError(
                "El contenido del archivo está vacío. Verifique el archivo."
            )
        self.document_repo.save_document(document, content, self.openai_adapter)

    def get_all_documents(self) -> List[Document]:
        return self.document_repo.get_all_documents()

    def get_document(self, doc_id: str) -> Document:
        return self.document_repo.get_document(doc_id)
