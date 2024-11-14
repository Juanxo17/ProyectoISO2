import tiktoken
import openai
from typing import List, Optional


# Función para obtener embeddings de OpenAI
def get_embeddings(
    text: str, openai_client: Optional[openai.OpenAI] = None
) -> List[float]:
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return list(map(float, embedding))  # Ensure the return type is List[float]


# Función para dividir el texto en fragmentos basados en el número de tokens
def chunk_text(text: str, max_tokens: int) -> List[str]:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)

    # Dividir en fragmentos si excede el límite de tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(chunk))  # Decodificar los tokens a texto
    return chunks


# Función principal para convertir el contenido en vectores de embeddings
def document_in_vectors(
    content: str, openai_client: Optional[openai.OpenAI] = None
) -> List[List[float]]:
    chunks = chunk_text(content, 100)
    vectors = []

    # Generar embeddings para cada chunk
    for chunk in chunks:
        vector = get_embeddings(chunk, openai_client)
        vectors.append(vector)  # Añadir los embeddings a la lista de vectores

    return vectors  # Devuelve una lista de vectores (embeddings)
