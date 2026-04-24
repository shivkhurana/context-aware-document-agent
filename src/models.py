from pydantic import BaseModel


class DocumentUploadRequest(BaseModel):
    filename: str
    content: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
