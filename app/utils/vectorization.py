import tiktoken
import openai

from app.adapters.openai_adapter import OpenAIAdapter


def get_embeddings(text: str, openai_client):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


def chunk_text(text: str, max_tokens: int):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)

    # Dividir en fragmentos si excede el límite de tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk))  # Decodificar los tokens a texto
    return chunks

def document_in_vectors(content: str, openai_adapter:OpenAIAdapter):
    chunks = chunk_text(content,100)
    vectors = []
    for chunk in chunks:
        vector = get_embeddings(chunk, openai_adapter)
        vectors.append(vector)
    return vectors
