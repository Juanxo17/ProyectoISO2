import pydantic
from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel
from app import usecases
from app.api import dependencies
from typing import Dict, List

rag_router = APIRouter()


class DocumentInput(BaseModel):
    content: str = pydantic.Field(..., min_length=1)


@rag_router.post("/generate-answer/", status_code=200)
def generate_answer(
    query: str,
    rag_service: usecases.RAGService = Depends(
        dependencies.RAGServiceSingleton.get_instance
    ),
) -> Dict[str, str]:
    return {"answer": rag_service.generate_answer(query)}


@rag_router.post("/save-document/", status_code=201)
def save_document(
    file: UploadFile = File(...),
    rag_service: usecases.RAGService = Depends(
        dependencies.RAGServiceSingleton.get_instance
    ),
) -> Dict[str, str]:
    rag_service.save_document(file)
    return {"status": "Document saved successfully"}


@rag_router.get("/get-all-documents/", status_code=200)
def get_all_documents(
    rag_service: usecases.RAGService = Depends(
        dependencies.RAGServiceSingleton.get_instance
    ),
) -> List[usecases.Document]:
    return rag_service.get_all_documents()


@rag_router.get("/get-document/{doc_id}", status_code=200)
def get_document(
    doc_id: str,
    rag_service: usecases.RAGService = Depends(
        dependencies.RAGServiceSingleton.get_instance
    ),
) -> usecases.Document:
    return rag_service.get_document(doc_id)
