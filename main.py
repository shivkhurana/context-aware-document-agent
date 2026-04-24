from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os

from src.models import DocumentUploadRequest, QueryRequest, QueryResponse
from src.aws_client import s3_client
from src.vector_store import embed_and_store_document
from src.rag_agent import qa_chain

load_dotenv()

app = FastAPI(title="Enterprise RAG API", version="1.0.0")


@app.post("/api/upload", response_model=dict)
async def upload_document(request: DocumentUploadRequest):
    """Upload a document to S3 and embed it in the vector store."""
    try:
        # Upload to S3
        success = s3_client.upload_text_document(request.filename, request.content)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload to S3")

        # Embed and store
        embed_and_store_document(request.filename, request.content)

        return {"message": f"Document {request.filename} uploaded and embedded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Answer a question using the RAG system."""
    try:
        result = qa_chain({"query": request.query})
        answer = result["result"]
        sources = [doc.metadata["source"] for doc in result["source_documents"]]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Enterprise RAG API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
