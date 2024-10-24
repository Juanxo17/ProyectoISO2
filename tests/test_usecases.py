import io
import os
from unittest.mock import Mock, patch, mock_open
from app import usecases
from app.core import ports, models
from fastapi import UploadFile
import pytest
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas

class MockDocumentRepository(ports.DocumentRepositoryPort):
    def __init__(self) -> None:
        self._storage = {}

    def save_document(self, document: models.Document, content: str, openai_adapter: ports.LlmPort) -> None:
        document.content = content
        self._storage[document.doc_id] = document

    def get_documents(self, query: str, n_results: int | None = None):
        return list(self._storage.values())

    def get_all_documents(self):
        return list(self._storage.values())

    def get_document(self, doc_id: str):
        return self._storage.get(doc_id)

@pytest.fixture
def mock_file():
    # Create a valid PDF content with text using reportlab
    pdf_bytes = io.BytesIO()
    c = canvas.Canvas(pdf_bytes)
    c.drawString(100, 750, "Test PDF content")
    c.save()
    pdf_bytes.seek(0)

    file = UploadFile(filename="test.pdf", file=Mock())
    file.file.read = Mock(return_value=pdf_bytes.read())
    return file

def test_should_save_document_when_calling_rag_service_save_method(mock_file):
    # Arrange
    document_repo = MockDocumentRepository()
    llm_mock = Mock(spec=ports.LlmPort)

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_mock
    )

    # Act
    with patch("builtins.open", mock_open(read_data=mock_file.file.read())):
        rag_service.save_document(file=mock_file)

    # Assert
    documents = document_repo.get_documents(query="test")
    assert len(documents) > 0
    assert documents[0].nombre == "test.pdf"
    assert documents[0].path == os.path.join("Library", "test.pdf")
    assert documents[0].content != ""
    assert documents[0].user_id == "test"

def test_generate_answer():
    # Arrange
    document_repo = MockDocumentRepository()
    llm_mock = Mock(spec=ports.LlmPort)
    llm_mock.generate_text.return_value = "Generated answer"

    # Add some documents to the repository
    document_repo.save_document(models.Document(nombre="doc1", path="path1", content="content1", user_id="user1"), "content1", llm_mock)
    document_repo.save_document(models.Document(nombre="doc2", path="path2", content="content2", user_id="user2"), "content2", llm_mock)

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_mock
    )

    # Act
    answer = rag_service.generate_answer(query="test query")

    # Assert
    assert answer == "Generated answer"
    llm_mock.generate_text.assert_called_once_with(prompt="test query", retrieval_context="content1 content2")

def test_get_all_documents():
    # Arrange
    document_repo = MockDocumentRepository()
    llm_mock = Mock(spec=ports.LlmPort)

    # Add some documents to the repository
    document_repo.save_document(models.Document(nombre="doc1", path="path1", content="content1", user_id="user1"), "content1", llm_mock)
    document_repo.save_document(models.Document(nombre="doc2", path="path2", content="content2", user_id="user2"), "content2", llm_mock)

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_mock
    )

    # Act
    documents = rag_service.get_all_documents()

    # Assert
    assert len(documents) == 2
    assert documents[0].nombre == "doc1"
    assert documents[1].nombre == "doc2"

def test_get_document():
    # Arrange
    document_repo = MockDocumentRepository()
    llm_mock = Mock(spec=ports.LlmPort)

    # Add a document to the repository
    document = models.Document(nombre="doc1", path="path1", content="content1", user_id="user1")
    document_repo.save_document(document, "content1", llm_mock)

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_mock
    )

    # Act
    retrieved_document = rag_service.get_document(doc_id=document.doc_id)

    # Assert
    assert retrieved_document is not None
    assert retrieved_document.nombre == "doc1"
    assert retrieved_document.path == "path1"
    assert retrieved_document.content == "content1"
    assert retrieved_document.user_id == "user1"