from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ===============================
# CONFIG
# ===============================

COLLECTION_NAME = "ebook_vectors"

QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333

OLLAMA_BASE_URL = "http://ollama:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# ===============================
# APP
# ===============================

app = FastAPI()

# ===============================
# CLIENTS (singleton)
# ===============================

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL
)

# ===============================
# SCHEMA
# ===============================

class Question(BaseModel):
    question: str


# ===============================
# ENDPOINT
# ===============================

@app.post("/ask")
def ask(q: Question):
    # 1. Busca no banco vetorial
    docs = vectorstore.similarity_search(q.question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    # 2. Prompt RAG
    prompt = f"""
Você deve responder a pergunta usando APENAS o contexto abaixo.
Se não souber, diga que não sabe.

Contexto:
{context}

Pergunta:
{q.question}
"""

    # 3. Chamada da LLM
    answer = llm.invoke(prompt)

    return {
        "question": q.question,
        "answer": answer
    }
