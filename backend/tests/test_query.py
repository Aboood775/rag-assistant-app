from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app


class FakeRetriever:
    embedding_model = "nomic-embed-text"

    def count(self):
        return 10

    def is_relevant(self, question):
        return "iphone" not in question.lower()

    def retrieve(self, question):
        return [Document(page_content="Product: Raspberry Pi 5\nBroadcom BCM2712 2.4GHz quad-core Cortex-A76 CPU",
                         metadata={"source": "raspberry-pi-5-product-brief.pdf", "page": 1})]

    def group_by_page(self, docs):
        return {f"{d.metadata['source']} (page {d.metadata['page'] + 1})": {"product": "Raspberry Pi 5", "texts": [d.page_content]}
                for d in docs}


class FakeGenerator:
    model = "llama3.2"

    def generate(self, question, groups):
        return "Raspberry Pi 5 uses a quad-core Cortex-A76 CPU at 2.4GHz [1]."

    def is_available(self):
        return True


app.state.retriever = FakeRetriever()
app.state.generator = FakeGenerator()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_happy_path():
    response = client.post("/query", json={"question": "What CPU does the Raspberry Pi 5 use?"})
    assert response.status_code == 200
    body = response.json()
    assert "Cortex-A76" in body["answer"]
    assert body["sources"] == ["raspberry-pi-5-product-brief.pdf (page 2)"]


def test_query_out_of_scope():
    response = client.post("/query", json={"question": "What is the battery life of the iPhone 15?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_query_missing_question_returns_422():
    response = client.post("/query", json={"text": "wrong field"})
    assert response.status_code == 422


def test_query_empty_question_returns_422():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
