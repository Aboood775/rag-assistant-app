import os

import requests
from dotenv import load_dotenv

load_dotenv()


class ApiError(Exception):
    pass


def get_base_url() -> str:
    url = os.getenv("API_BASE_URL")
    if not url:
        raise ApiError("API_BASE_URL is not set. Copy .env.example to .env and set the backend URL.")
    return url.rstrip("/")


def check_health() -> dict:
    try:
        response = requests.get(f"{get_base_url()}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ApiError("The backend is not reachable.") from exc


def ask_question(question: str) -> dict:
    try:
        response = requests.post(f"{get_base_url()}/query", json={"question": question}, timeout=180)
    except requests.Timeout as exc:
        raise ApiError("The request took too long. Please try again.") from exc
    except requests.RequestException as exc:
        raise ApiError("Could not connect to the backend. Make sure the API is running.") from exc

    if response.status_code == 422:
        raise ApiError("Please write a question between 3 and 500 characters.")
    if response.status_code != 200:
        detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type", "").startswith("application/json") else response.text
        raise ApiError(f"The API returned an error: {detail}")
    return response.json()
