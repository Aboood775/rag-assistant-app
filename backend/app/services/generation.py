import logging

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "I could not find the answer in the documents."

TEMPLATE = """You are a helpful assistant that answers questions about Raspberry Pi products.
Use only the context below to answer. Do not use any outside knowledge.
If the context does not contain the answer, reply exactly: "I could not find the answer in the documents."
Copy numbers and units exactly as they are written in the context.
Keep the answer short and cite the context numbers you used, like [1] or [2].

Context:
{context}

Question: {question}

Answer:"""


class GenerationService:
    def __init__(self, model: str, base_url: str, temperature: float = 0.0, max_tokens: int = 300):
        self.model = model
        self.base_url = base_url
        llm = ChatOllama(model=model, base_url=base_url, temperature=temperature,
                         num_predict=max_tokens, seed=365)
        self.chain = PromptTemplate.from_template(TEMPLATE) | llm | StrOutputParser()

    @staticmethod
    def format_context(groups: dict[str, dict]) -> str:
        blocks = []
        for i, (source, group) in enumerate(groups.items(), start=1):
            blocks.append(f"[{i}] {source} - {group['product']}\n" + "\n".join(group["texts"]))
        return "\n\n".join(blocks)

    def generate(self, question: str, groups: dict[str, dict]) -> str:
        if not groups:
            return NOT_FOUND_ANSWER
        answer = self.chain.invoke({"context": self.format_context(groups), "question": question})
        return answer.strip()

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
