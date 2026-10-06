from pathlib import Path

from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import get_embeddings, get_embedding_dimension,add_documents
from app.core.config import get_settings

settings = get_settings()
doc=load_file(Path("data/sample_kb/company_it_handbook.md"))    

print(f"Loaded document: {doc}")


chunks=chunk_documents(doc)
print(f"Created chunks: {len(chunks)}")

add_documents(chunks)
