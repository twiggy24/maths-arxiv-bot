from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from settings import settings
from embedder import EMBED_DIM

def connect() -> QdrantClient: 
    """Create a Qdrant client from settings.
    - For local Docker: QDRANT_URL=http://localhost:6333
    - for cloud: use QdrantClient(url=..., api_key=...)
    """
    return QdrantClient(url=settings.QDRANT_URL)

def ensure_collection(client: QdrantClient) -> None:
    from embedder import EMBED_DIM  # re-read on each call in case backend/model changed
    client.recreate_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


