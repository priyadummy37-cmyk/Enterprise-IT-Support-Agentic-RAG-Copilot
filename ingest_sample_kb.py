from pathlib import Path
from pathlib import Path

from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import get_embeddings, get_embedding_dimension,add_documents
from app.core.config import get_settings

settings = get_settings()

folder=Path(settings.sample_kb_dir)

for file in folder.iterdir():
    if file.suffix.lower() in {".txt", ".pdf", ".docx", ".md"}:
        doc=load_file(file)    
        print(f"Loaded document: {file.name} with {len(doc)} pages")
        chunks=chunk_documents(doc)
        print(f"Created chunks: {len(chunks)}")
        add_documents(chunks)
