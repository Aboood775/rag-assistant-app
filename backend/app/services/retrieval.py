import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, vector_store_dir: Path, ollama_base_url: str, top_k: int = 4,
                 fetch_k: int = 20, lambda_mult: float = 0.9, max_distance: float | None = None):
        config_path = Path(vector_store_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Vector store not found at {vector_store_dir}. Run the notebook first.")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.embedding_model = self.config["embedding_model"]
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.max_distance = max_distance if max_distance is not None else self.config.get("max_distance", 0.4)

        embedding = OllamaEmbeddings(model=self.embedding_model, base_url=ollama_base_url)
        self.vectorstore = Chroma(
            persist_directory=str(vector_store_dir),
            embedding_function=embedding,
            collection_name=self.config["collection_name"],
        )
        logger.info("Loaded vector store with %d chunks, max distance %.2f", self.count(), self.max_distance)

    def count(self) -> int:
        return self.vectorstore._collection.count()

    def is_relevant(self, question: str) -> bool:
        results = self.vectorstore.similarity_search_with_score(question, k=1)
        if not results:
            return False
        best_distance = results[0][1]
        logger.info("Best distance for question: %.3f", best_distance)
        return best_distance <= self.max_distance

    def retrieve(self, question: str) -> list[Document]:
        return self.vectorstore.max_marginal_relevance_search(
            question, k=self.top_k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
        )

    @staticmethod
    def format_source(doc: Document) -> str:
        return f"{doc.metadata['source']} (page {doc.metadata['page'] + 1})"

    def group_by_page(self, docs: list[Document]) -> dict[str, dict]:
        groups = {}
        for doc in docs:
            key = self.format_source(doc)
            text = doc.page_content
            if text.startswith("Product:"):
                text = text.split("\n", 1)[1]
            if key not in groups:
                groups[key] = {"product": doc.metadata.get("product", ""), "texts": []}
            groups[key]["texts"].append(text)
        return groups
