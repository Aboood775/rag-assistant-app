import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services.generation import NOT_FOUND_ANSWER

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    return HealthResponse(
        status="ok",
        ollama="up" if generator.is_available() else "down",
        llm_model=generator.model,
        embedding_model=retriever.embedding_model,
        chunks_indexed=retriever.count(),
    )


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request):
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    question = payload.question.strip()
    start = time.perf_counter()

    try:
        if not retriever.is_relevant(question):
            logger.info("Question is out of scope: %s", question)
            return QueryResponse(answer=NOT_FOUND_ANSWER, sources=[])
        docs = retriever.retrieve(question)
        groups = retriever.group_by_page(docs)
        answer = generator.generate(question, groups)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=503, detail="The language model service is not available. Is Ollama running?") from exc

    sources = list(groups.keys())
    logger.info("Answered in %.1fs with %d sources", time.perf_counter() - start, len(sources))
    return QueryResponse(answer=answer, sources=sources)
