from pinecone import Pinecone,ServerlessSpec
import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.services.ingestion import load_file, chunk_documents
from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return _embeddings

import time

from pinecone import Pinecone, ServerlessSpec


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    # Extract the actual string from SecretStr
    pinecone_api_key = settings.pinecone_api_key.get_secret_value()

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=pinecone_api_key)

    index_name = settings.pinecone_index_name

    # Get existing indexes
    indexes = pc.list_indexes()
    names = [x["name"] for x in indexes]

    # Check whether index already exists
    if index_name in names:
        index_info = pc.describe_index(index_name)

        current_dimension = getattr(
            index_info,
            "dimension",
            None
        )

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        # Recreate index if embedding dimension changed
        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):
            print(
                f"Dimension mismatch: "
                f"existing={current_dimension}, "
                f"required={desired_dimension}"
            )

            print(f"Deleting Pinecone index: {index_name}")

            pc.delete_index(name=index_name)

            while index_name in [
                x["name"] for x in pc.list_indexes()
            ]:
                time.sleep(1)

    # Create index if it doesn't exist
    if index_name not in [
        x["name"] for x in pc.list_indexes()
    ]:
        print(
            f"Creating Pinecone index '{index_name}' "
            f"with dimension {desired_dimension}"
        )

        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until index is ready
        while True:
            index_info = pc.describe_index(index_name)

            if index_info.status["ready"]:
                break

            time.sleep(1)

    return pc.Index(index_name)


def get_vectorstore() -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace
        )
    return _vectorstore


def getretriever() :
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k} )


def add_documents(chunks):
    vectorstore = get_vectorstore()
    return vectorstore.add_documents(chunks)