import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vector store from %s", settings.vector_store_dir)
    app.state.retriever = RetrievalService(
        vector_store_dir=settings.vector_store_dir,
        ollama_base_url=settings.ollama_base_url,
        top_k=settings.top_k,
        fetch_k=settings.fetch_k,
        lambda_mult=settings.lambda_mult,
        max_distance=settings.max_distance,
    )
    app.state.generator = GenerationService(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    logger.info("Startup finished, LLM model: %s", settings.llm_model)
    yield
    logger.info("Shutting down")


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(query_router)
