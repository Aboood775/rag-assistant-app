import streamlit as st

from api_client import ApiError, ask_question, check_health

st.set_page_config(page_title="Raspberry Pi Docs Assistant", page_icon="🍓", layout="centered")

EXAMPLES = [
    "What CPU does the Raspberry Pi 5 use?",
    "What RAM options are available for the Raspberry Pi 4 Model B?",
    "What is the field of view of the Camera Module 3 Wide?",
]


def show_sources(sources):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for i, source in enumerate(sources, start=1):
                st.markdown(f"**[{i}]** {source}")


with st.sidebar:
    st.header("About")
    st.write("Ask questions about official Raspberry Pi product briefs. "
             "Answers come only from the documents, with the sources they used.")
    st.subheader("Backend status")
    try:
        health = check_health()
        st.success("API is running")
        st.write(f"LLM: `{health['llm_model']}`")
        st.write(f"Embeddings: `{health['embedding_model']}`")
        st.write(f"Chunks indexed: {health['chunks_indexed']}")
        if health["ollama"] != "up":
            st.warning("Ollama is not reachable from the API.")
    except ApiError as e:
        st.error(str(e))
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("🍓 Raspberry Pi Docs Assistant")
st.caption("RAG over official Raspberry Pi product briefs · FastAPI · Chroma · Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("Try one of these:")
    for example in EXAMPLES:
        st.markdown(f"- {example}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("Ask a question about Raspberry Pi products...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents and writing the answer..."):
                result = ask_question(question)
            st.markdown(result["answer"])
            show_sources(result["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
        except ApiError as e:
            st.error(f"Sorry, something went wrong. {e}")
