from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================
# CONFIG
# ===============================

COLLECTION_NAME = "ebook_vectors"

QDRANT_HOST = "qdrant"      # 🔥 nome do service no docker-compose
QDRANT_PORT = 6333

OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "nomic-embed-text"

# ===============================
# TEXTO DE EXEMPLO (troque pelo seu)
# ===============================

text = """
O Instagram do JCA é @jca.dev.
Ele cria conteúdo sobre Python, IA e Docker.
"""

# ===============================
# SPLIT
# ===============================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_text(text)

print(f"📄 Total de chunks: {len(texts)}")

# ===============================
# EMBEDDINGS
# ===============================

embeddings = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

# ===============================
# QDRANT CLIENT
# ===============================

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

# ===============================
# VECTOR STORE (CRIA / CONECTA)
# ===============================

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# ===============================
# INGESTÃO
# ===============================

vectorstore.add_texts(texts)

print("✅ Ingestão concluída com sucesso!")
