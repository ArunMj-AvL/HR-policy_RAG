import os
import streamlit as st

from pathlib import Path

from config import INDEX_DIR, TOP_K
from ingestion import VectorStore, get_embedder

from src.retrieval import PolicyRetriever
from src.answer_generator import generate_answer, FALLBACK


st.set_page_config(
    page_title="PolicyWise",
    page_icon="📘",
    layout="centered"
)

st.title("📘 PolicyWise")
st.subheader("Source-Grounded HR Policy Assistant")

st.write(
    "Ask questions about company HR policies "
    "and receive answers grounded in policy documents."
)

# Load persistent vector index
try:
    vector_store = VectorStore.load(INDEX_DIR)

    embedder_name = vector_store.meta.get("embedder", "")

    embedder_choice = (
        "hash"
        if embedder_name.startswith("hashing")
        else "minilm"
    )

    embedder = get_embedder(embedder_choice)

    if embedder.name != embedder_name:
        raise ValueError(
            "Embedding model does not match the saved index."
        )

    retriever = PolicyRetriever(
        vector_store,
        embedder
    )

except Exception as exc:
    st.error(f"Unable to load policy index: {exc}")
    st.info(
        "Run: python -m ingestion.ingest --rebuild"
    )
    st.stop()



st.divider()

st.markdown("### Ask an HR Policy Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is the hotel limit for Tier 1 cities?",
    height=100
)

if st.button("Get Answer", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching policy documents..."):

            results = retriever.retrieve(
                question,
                k=TOP_K
            )

            answer, citations = generate_answer(
                question,
                results,
                use_llm=bool(os.getenv("OPENAI_API_KEY"))
            )

        st.markdown("### Answer")
        st.write(answer)

        if citations:
            st.markdown("### Sources")

            for source, page in citations:
                st.write(f"- {source}, Page {page}")

        if answer == FALLBACK:
            st.info(FALLBACK)

        if results:
            with st.expander("Inspect retrieved evidence"):
                for i, item in enumerate(results, 1):
                    st.markdown(
                        f"**Result {i}: {item['source']}**"
                    )

                    st.caption(
                        f"Page: {item['page']} | "
                        f"Status: {item.get('status')} | "
                        f"Score: {item['score']:.4f}"
                    )

                    st.write(item["text"])
                    st.divider()

st.divider()

with st.expander("About PolicyWise"):
    st.write(
        "PolicyWise is a Retrieval-Augmented Generation (RAG) "
        "assistant that answers employee questions using "
        "policy documents and provides source references."
    )

