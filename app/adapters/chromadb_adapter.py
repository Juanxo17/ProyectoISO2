from os.path import pathsep

import chromadb
from typing import List

from sympy import Range

from app.core import ports
from app.core import models
from app.utils.vectorization import document_in_vectors
import itertools  # Para aplanar la lista
import openai


class ChromaDBAdapter(ports.DocumentRepositoryPort):
    def __init__(self, number_of_vectorial_results: int, api_key: str) -> None:
        # Crear el cliente de ChromaDB y la colección sin especificar dimensión
        openai.api_key = api_key
        self.client = chromadb.Client()
        self.collectionDocuments = self.client.create_collection("documents")  # Sin 'dimension'
        self.collectionUsers = self.client.create_collection("users")  # Sin 'dimension'
        self._number_of_vectorial_results = number_of_vectorial_results

    def save_document(self, document: models.Document, content: str, openai_client) -> None:
        # Generar embeddings divididos por fragmentos
        embeddings_document = document_in_vectors(content, openai_client)

        # Aplanar la lista de listas de embeddings
        flat_embeddings = list(itertools.chain.from_iterable(embeddings_document))

        # Verificar la dimensionalidad de los embeddings antes de agregar a la colección
        if len(flat_embeddings) != 1536:  # Cambia esto para aceptar la dimensión correcta (1536)
            raise ValueError(f"Expected embeddings of dimension 1536, but got {len(flat_embeddings)}")

        # Guardar en la colección de ChromaDB
        self.collectionDocuments.add(
            ids=[document.doc_id],
            embeddings=[flat_embeddings],  # Embedding ya está aplanado
            documents=[content],
            metadatas=[{"filename": document.nombre, "path": document.path, "user_id": document.user_id}]
        )

    def get_documents(self, query: str, n_results: int | None = None) -> List[models.Document]:
        if not n_results:
            n_results = self._number_of_vectorial_results

        # Generar el embedding de la consulta utilizando OpenAI para garantizar la misma dimensión
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response['data'][0]['embedding']

        # Consultar la colección de ChromaDB con el embedding de la consulta
        results = self.collectionDocuments.query(
            query_embeddings=[query_embedding],  # Usar embedding en lugar de texto
            n_results=n_results
        )

        print(query)
        print(f"Results: {results}")

        documents = []
        for i, doc_id_list in enumerate(results['ids']):
            for doc_id in doc_id_list:
                documents.append(models.Document(
                    id=doc_id,
                    nombre=results['metadatas'][i][0]['filename'],
                    content=results['documents'][i][0],
                    path=results['metadatas'][i][0]['path'],
                    user_id=results['metadatas'][i][0]['user_id']
                ))

        return documents

    def get_all_documents(self):
        documents = []
        results = self.collectionDocuments.get()

        for i, doc_id_list in enumerate(results['ids']):
                documents.append(models.Document(
                    id=results['ids'][i],
                    nombre=results['metadatas'][i]['filename'],
                    content=results['documents'][i],
                    path=results['metadatas'][i]['path'],
                    user_id=results['metadatas'][i]['user_id']
                ))
        return documents

    def get_document(self, doc_id: str):
        vectors = self.collectionDocuments.get(doc_id)
        return models.Document(
            id=doc_id,
            nombre="Nombre del documento",
            content=vectors['documents'][0],
            path="Ruta del documento"
        )

