import chromadb
from typing import List
from app.core import ports
from app.core import models
from app.utils.vectorization import document_in_vectors


class ChromaDBAdapter(ports.DocumentRepositoryPort):
    def __init__(self, number_of_vectorial_results: int) -> None:
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        self._number_of_vectorial_results = number_of_vectorial_results

    def save_document(self, document: models.Document, content: str, openai_client) -> None:
        embeddings_document = document_in_vectors(content, openai_client)

        self.collection.add(
            ids=[document.doc_id],
            embeddings=[embeddings_document],
            documents=[content]
        )

    def get_documents(self, query: str, n_results: int | None = None) -> List[models.Document]:
        if not n_results:
            n_results = self._number_of_vectorial_results
        results = self.collection.query(query_texts=[query], n_results=n_results)
        print(query)
        print(f"Results: {results}")
        documents = []
        for i, doc_id_list in enumerate(results['ids']):
            for doc_id in doc_id_list:
                documents.append(models.Document(id=doc_id, content=results['documents'][i][0]))
        return documents
