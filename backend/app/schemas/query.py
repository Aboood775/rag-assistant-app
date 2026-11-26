from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["What CPU does the Raspberry Pi 5 use?"])


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    ollama: str
    llm_model: str
    embedding_model: str
    chunks_indexed: int
