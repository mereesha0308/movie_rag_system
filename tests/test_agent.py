"""Tests for the LangGraph agent (uses a mocked LLM - no API key needed)."""

from __future__ import annotations

import json

import numpy as np
from langchain_core.messages import AIMessage

from src.agent import MovieAgent
from src.embeddings import Embedder
from src.retriever import Retriever


class FakeEmbedder(Embedder):
    """Content-based fake embeddings: HAL->row1, boxer->row2, scientist->row3."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            low = t.lower()
            if "hal" in low or "2001" in low:
                rows.append(1)
            elif "boxer" in low or "champ" in low:
                rows.append(2)
            elif "scientist" in low or "amplifier" in low or "4d" in low:
                rows.append(3)
            else:
                rows.append(0)
        return np.eye(self._dim)[rows].astype("float32")


class FakeModel:
    """Replays a canned list of AIMessages (tool call, final JSON, repairs...)."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def invoke(self, messages, config=None):
        msg = self._responses[self.calls]
        self.calls += 1
        return msg


def tool_call_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve_movie_plots",
                "args": {"query": "mock query"},
                "id": "call_1",
            }
        ],
    )


def answer_tool_call(args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "produce_answer", "args": args, "id": "call_ans"}],
    )


def make_config(tmp_path):
    return {
        "data": {"raw_path": str(tmp_path / "raw.csv"), "subset_size": 10, "random_seed": 1},
        "preprocessing": {
            "drop_duplicates": True,
            "drop_missing": True,
            "drop_unknown": True,
            "unknown_director": "Unknown",
            "unknown_genre": "unknown",
            "unknown_cast": "Unknown",
        },
        "chunking": {"chunk_size": 300, "overlap": 0},
        "retriever": {
            "vector_store": "chroma",
            "store_path": str(tmp_path / "chroma_store"),
            "collection_name": "test_movie_chunks",
            "top_k": 3,
        },
        "llm": {"model": "fake", "temperature": 0.2, "max_tokens": 300},
        "agent": {"max_iterations": 3},
        "rag": {"system_prompt": "You are a movie assistant. Use the tool to retrieve context."},
    }


def test_agent_loop_tool_then_answer(tmp_path):
    config = make_config(tmp_path)
    retriever = Retriever(config)
    records = [
        {
            "chunk_id": "0-0",
            "chunk_text": "A spaceship is controlled by an artificial intelligence named HAL.",
            "Title": "2001",
            "Release Year": 1968,
        },
        {
            "chunk_id": "1-0",
            "chunk_text": "A young boxer trains hard to win a championship.",
            "Title": "Champ",
            "Release Year": 1979,
        },
    ]
    retriever.add_chunks(records, FakeEmbedder().encode([r["chunk_text"] for r in records]))

    final = {
        "answer": "HAL 9000 is the AI in 2001: A Space Odyssey.",
        "reasoning": "The retrieved chunk mentions HAL.",
    }
    responses = [
        tool_call_message(),
        AIMessage(content="DONE"),
        answer_tool_call(final),
    ]
    agent = MovieAgent(config, retriever, FakeEmbedder(), model=FakeModel(responses))

    result = agent.run("Which movie has an AI called HAL?")

    assert agent.model.calls == 3, "tool call + DONE + answer generation"
    assert result["answer"].startswith("HAL 9000")
    assert 1 <= len(result["contexts"]) <= 2
    assert any("HAL" in c for c in result["contexts"]), "context should support the answer"
    assert "The retrieved chunk" in result["reasoning"]


def test_agent_clarifying_question(tmp_path):
    config = make_config(tmp_path)
    retriever = Retriever(config)
    records = [
        {
            "chunk_id": "0-0",
            "chunk_text": "A long plot about a detective solving crimes in a city.",
            "Title": "Detective",
            "Release Year": 1990,
        }
    ]
    retriever.add_chunks(records, FakeEmbedder().encode([r["chunk_text"] for r in records]))

    responses = [AIMessage(content="CLARIFY: Which movie are you asking about? Could you give me more details?")]
    agent = MovieAgent(config, retriever, FakeEmbedder(), model=FakeModel(responses))

    result = agent.run("Tell me about that movie")

    assert agent.model.calls == 1, "vague query should not trigger retrieval or answer calls"
    assert result["answer"] == ""
    assert result.get("clarifying_question")
    assert result["clarifying_question"].startswith("Which movie")


def test_agent_recovers_from_misclassified_vague_query(tmp_path):
    config = make_config(tmp_path)
    retriever = Retriever(config)
    records = [
        {
            "chunk_id": "0-0",
            "chunk_text": "Two slacker roommates make silly jokes and pranks all day.",
            "Title": "Comedy Bros",
            "Release Year": 2005,
        }
    ]
    retriever.add_chunks(records, FakeEmbedder().encode([r["chunk_text"] for r in records]))

    # The model wrongly asks for clarification even though the query has
    # searchable content ("comedy"). The agent must recover and answer anyway.
    responses = [
        AIMessage(content="CLARIFY: Which specific movie are you interested in?"),
        answer_tool_call(
            {"answer": "Comedy Bros is a comedy about two slacker roommates.",
             "reasoning": "The query mentioned comedy and the retrieved chunk is a comedy."}
        ),
    ]
    agent = MovieAgent(config, retriever, FakeEmbedder(), model=FakeModel(responses))

    result = agent.run("something about comedy")

    assert agent.model.calls == 2, "clarify + recovery answer generation"
    assert result["answer"].startswith("Comedy Bros")
    assert not result.get("clarifying_question")
    assert any("roommate" in c.lower() for c in result["contexts"])


def test_agent_recovers_from_empty_answer_args(tmp_path):
    config = make_config(tmp_path)
    retriever = Retriever(config)
    records = [
        {
            "chunk_id": "0-0",
            "chunk_text": "A scientist develops an electronic amplifier for a fourth dimension.",
            "Title": "4D Man",
            "Release Year": 1959,
        }
    ]
    retriever.add_chunks(records, FakeEmbedder().encode([r["chunk_text"] for r in records]))

    repaired = {
        "answer": "The plot is about a scientist who builds an amplifier.",
        "reasoning": "I found the scientist in the retrieved chunk.",
    }
    responses = [
        tool_call_message(),
        AIMessage(content="DONE"),
        answer_tool_call({}),
        answer_tool_call(repaired),
    ]
    agent = MovieAgent(config, retriever, FakeEmbedder(), model=FakeModel(responses))

    result = agent.run("What is the plot of 4d man?")

    assert agent.model.calls == 4, "empty answer args should trigger one retry"
    assert result["answer"].startswith("The plot is about a scientist")
    assert "Could not parse" not in result["reasoning"]
    assert result["reasoning"] == "I found the scientist in the retrieved chunk."
