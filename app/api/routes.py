from fileinput import filename
from pathlib import Path
from aiohttp import Payload
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents,supported_extensions
from app.services.audit import write_audit

settings = get_settings()
SUPPORTED = supported_extensions

router = APIRouter(prefix="/api", tags=["RAG API"])

class chatRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system.", min_length=2, max_length=3000)

@router.get("/health")
def health_check():
    settings = get_settings()
    return {"status": "ok", "message": "All required API keys are configured."}

@router.post("/chat")
def chat(payload: chatRequest):
    try:
        answer = ask(payload.question)
        write_audit(payload.question, answer["source_used"], answer.get("trace",[]))
        return {"answer": answer["answer"], "source_used": answer["source_used"], 
                "trace": answer.get("trace", []), 
                "citations": answer.get("citations", []),
                "rewrite_query": answer.get("current_query", payload.question),
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
    
@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="")
):
    # Validate admin API key
    if (
        settings.admin_api_key is None
        or x_admin_key != settings.admin_api_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )

    # Validate filename
    filename = file.filename

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    # Validate file extension
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}"
        )

    # Create upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Remove any directory component from uploaded filename
    dest = upload_dir / Path(filename).name

    # Save file
    dest.write_bytes(await file.read())

    # Load document
    docs = load_file(dest)

    # Split into chunks
    chunks = chunk_documents(docs)

    # Add chunks to vector database
    ids = add_documents(chunks)

    return {
        "message": "Document indexed",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids)
    }