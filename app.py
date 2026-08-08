"""Simple Streamlit UI for the Mini Movie RAG system.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from src.rag_graph import RAGPipeline
from src.utils import load_config

load_dotenv()

# Pull API keys from Streamlit secrets into the environment if present
# (set via the app's `.streamlit/secrets.toml`).
for key in ("GROQ_API_KEY",):
    if key in st.secrets:
        os.environ.setdefault(key, st.secrets[key])

st.set_page_config(page_title="Mini Movie RAG", layout="centered")

MAX_TOP_K = 10

SAMPLE_QUESTIONS = [
    "What is the plot of 4D Man?",
    "In Poltergeist II: The Other Side, who is the main villain?",
    "What is the Bengal tiger in Life of Pi called?",
    "In Billy Elliot, what sport does his father want him to learn?",
    "What kind of dinosaur is Patchi in Walking with Dinosaurs?",
    "Who is the imprisoned author in Quills?",
    "How does Count Dracula get revived in Dracula Has Risen from the Grave?",
    "Where does Robert McCall work in The Equalizer?",
    "What do the Boxtrolls wear?",
    "What does young David see land in the sandpit in Invaders from Mars?",
    "What element do the stranded aliens need in The Lost Skeleton of Cadavra?",
    "Who is the second son of Henry Frankenstein in The Ghost of Frankenstein?",
    "What begins to haunt Dr. Peter Proud in The Reincarnation of Peter Proud?",
    "What is the album the band promotes in This Is Spinal Tap?",
    "What is the name of the abused dog in Shiloh?",
    "Which movie features a Bengal tiger named Richard Parker?",
    "A boy named Eggs is raised by underground trolls who wear cardboard boxes — which movie is this?",
    "Which film tells the story of an 11-year-old boy who secretly takes ballet lessons?",
    "Which movie is about a retired CIA operative who works in a hardware store?",
    "Which movie follows a family haunted by an evil spirit in Cuesta Verde?",
]


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(load_config())


def main() -> None:
    st.title("Mini Movie RAG")
    st.caption("Ask questions about movie plots from a 500-movie subset.")

    pipeline = get_pipeline()

    if pipeline.retriever.count == 0:
        st.info("No vector index found yet. Build it once from the dataset.")
        if st.button("Build index"):
            with st.spinner("Building the index (chunk + embed + store)..."):
                pipeline.build_index(force=False)
            st.success(f"Index ready: {pipeline.retriever.count:,} chunks stored.")
            st.rerun()
        return

    st.sidebar.header("Settings")
    use_llm = st.sidebar.checkbox("Use LLM", value=True)
    top_k = st.sidebar.slider("top-k chunks", 1, MAX_TOP_K, 3)
    st.sidebar.caption(
        "Disable 'Use LLM' to see retrieval-only output (no API key needed)."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sample questions**")
    with st.sidebar:
        for i, sample in enumerate(SAMPLE_QUESTIONS):
            if st.button(sample, key=f"sample_{i}", use_container_width=True):
                st.session_state.question_input = sample
                st.rerun()

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the plot of the film 4d man?",
        key="question_input",
    )
    st.markdown("---")

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Thinking..."):
            if use_llm:
                result = pipeline.answer(question, top_k=top_k)
            else:
                chunks = pipeline.retrieve(question, top_k=top_k)
                result = {
                    "answer": "(LLM skipped)",
                    "contexts": [c["chunk_text"][:400] for c in chunks],
                    "reasoning": (
                        f"Retrieved {len(chunks)} chunks without generating an answer."
                    ),
                }

        st.markdown("---")

        if result.get("clarifying_question"):
            st.subheader("Clarifying question")
            st.info(result["clarifying_question"])
        else:
            st.subheader("Answer")
            st.markdown(result.get("answer") or "_No answer._")

        if result.get("contexts"):
            st.subheader("Retrieved context")
            for i, snippet in enumerate(result["contexts"], start=1):
                st.markdown(f"**Context {i}**")
                st.code(snippet, language=None)
        else:
            st.subheader("Retrieved context")
            st.markdown("_No retrieved context._")

        st.subheader("Reasoning")
        st.write(result.get("reasoning", ""))

        with st.expander("Full JSON"):
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
