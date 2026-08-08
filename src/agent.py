"""Single-agent RAG system built with LangGraph.

The agent is a ReAct-style loop: it can call one tool
(`retrieve_movie_plots`), receive the results, and then either call the tool
again, ask a clarifying question, or emit a final structured answer.
"""

from __future__ import annotations

import re
import sys
import time
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.embeddings import Embedder
from src.retriever import Retriever, _content_tokens

ANSWER_SYSTEM_PROMPT = """You are a movie plot assistant. Answer the user's question \
using ONLY the retrieved movie plot chunks.

- "answer": 1-3 concise, natural sentences. Always give the complete answer - never \
truncate or cut it off mid-sentence.
- "reasoning": A cohesive, natural paragraph explaining the logical steps taken to form the answer. \
Describe the connection between the user's query and the specific plot points retrieved, explaining \
how the evidence supports the conclusion. Do not use numbered lists, bullet points, or mention tool/function names.
- "clarifying_question": leave empty unless you genuinely need more details from the user.
- If no retrieved chunk is relevant, set "answer" to: "I do not have that information \
in my sources."
- Never invent facts that are not in the context."""


class AgentState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    chunks: list[dict]


class MovieAgent:
    """LangGraph agent that retrieves movie chunks and answers grounded questions."""

    def __init__(
        self,
        config: dict,
        retriever: Retriever,
        embedder: Embedder,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.embedder = embedder
        self.top_k = config["retriever"].get("top_k", 3)
        self.snippet_length = config["agent"].get("context_snippet_length", 200)
        self.system_prompt = config["rag"]["system_prompt"]
        self.max_iterations = config["agent"].get("max_iterations", 3)

        # Model and graph are built lazily so the index can be built without
        # an API key. Use `self.model` to trigger construction.
        self._injected_model = model
        self._model: Any | None = None
        self._graph: Any | None = None

    @property
    def model(self) -> Any:
        return self._get_model()

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._injected_model or self._build_model()
        return self._model

    def _produce_answer_tool(self) -> StructuredTool:
        def _produce(
            answer: str,
            reasoning: str,
            clarifying_question: str = "",
        ) -> str:
            """Emit the final structured answer about a movie.
            Args:
                answer: 1-2 complete sentences answering from the context.
                reasoning: plain-language explanation without tool names.
                clarifying_question: optional question when more detail is needed.
            """
            return ""

        return StructuredTool.from_function(
            func=_produce,
            name="produce_answer",
            description="Emit the final structured answer about a movie.",
        )

    def _structured_answer_model(self) -> Any:
        """Model forced to emit the `produce_answer` tool call.

        Groq's server-side validation returns the call args as valid JSON, so
        structured output is guaranteed - no fragile JSON parsing needed.
        """
        model = self._get_model()
        if self._injected_model is not None:
            return model
        return model.bind_tools(
            [self._produce_answer_tool()], tool_choice="produce_answer"
        )

    @staticmethod
    def _extract_tool_args(response: Any) -> dict:
        calls = getattr(response, "tool_calls", None)
        if calls:
            return dict(calls[0].get("args", {}) or {})
        return {}

    def _get_graph(self) -> Any:
        if self._graph is None:
            bound_model = self._get_model().bind_tools([self._retrieve_tool()])
            self._graph = self._build_graph(bound_model)
        return self._graph

    # ------------------------------------------------------------ model setup
    def _build_model(self) -> Any:
        llm = self.config["llm"]
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=llm["model"],
            temperature=llm.get("temperature", 0.2),
            max_tokens=llm.get("max_tokens", 300),
        )

    # ---------------------------------------------------------------- retrieval
    def _search(self, query: str, top_k: int | None = None) -> list[dict]:
        query_emb = self.embedder.encode([query])[0]
        return self.retriever.retrieve(
            query_emb, top_k=top_k or self.top_k, query_text=query
        )

    def _search_merged(self, primary: str, secondary: str, top_k: int) -> list[dict]:
        """Search with both the original user question and the model's query.

        The model's rephrasing can hurt retrieval (e.g. "4d man film plot"
        misses the "4D Man" chunk), so the original question leads and the
        model's query only adds extra candidates.
        """
        if not primary and not secondary:
            return []
        seen: set[str] = set()
        merged: list[dict] = []
        for q in (primary, secondary):
            if not q:
                continue
            for c in self._search(q, top_k=top_k):
                cid = c.get("chunk_id")
                if cid in seen:
                    continue
                seen.add(cid)
                merged.append(c)
        return merged[:top_k]

    @staticmethod
    def _format_chunks(chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant movie chunks were found."
        return "\n\n".join(
            f"[{i + 1}] {c.get('Title', 'Unknown')} ({c.get('Release Year', '?')}, "
            f"{c.get('Origin/Ethnicity', '?')})\n{c['chunk_text']}"
            for i, c in enumerate(chunks)
        )

    def _retrieve_tool(self) -> StructuredTool:
        def _func(query: str) -> str:
            return self._format_chunks(self._search(query))

        return StructuredTool.from_function(
            func=_func,
            name="retrieve_movie_plots",
            description=(
                "Search movie plot summaries for a query and return the most "
                "relevant snippets with movie titles. Use this to find context "
                "for answering the user's question."
            ),
        )

    # ------------------------------------------------------------------- graph
    def _build_graph(self, bound_model):
        def agent_node(state: AgentState):
            # Tool-call generation is occasionally flaky (Groq may reject a
            # malformed function call with a 400), so retry a few times. If it
            # keeps failing, degrade to DONE so the query still completes.
            response = None
            for attempt in range(3):
                try:
                    response = bound_model.invoke(state["messages"])
                    break
                except Exception as exc:  # noqa: BLE001 - transient API errors
                    print(f"[agent] tool-call attempt {attempt + 1} failed: {exc}", file=sys.stderr)
                    time.sleep(0.5 * (attempt + 1))
            if response is None:
                print("[agent] tool-call generation failed after 3 attempts; continuing", file=sys.stderr)
                response = AIMessage(content="DONE")
            return {"messages": [response]}

        def tools_node(state: AgentState):
            last = state["messages"][-1]
            tool_messages: list[BaseMessage] = []
            chunks: list[dict] = []
            for tool_call in last.tool_calls:
                query = tool_call["args"].get("query", "")
                chunks = self._search_merged(state["question"], query, self.top_k)
                tool_messages.append(
                    ToolMessage(content=self._format_chunks(chunks), tool_call_id=tool_call["id"])
                )
            return {"messages": tool_messages, "chunks": chunks}

        def router(state: AgentState) -> Literal["tools", "end"]:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return "end"

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ------------------------------------------------------------------- run
    def run(self, query: str) -> dict:
        """Run the agent and return a structured {answer, contexts, reasoning} dict."""
        initial_state: AgentState = {
            "question": query,
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query),
            ],
            "chunks": [],
        }
        result = self._get_graph().invoke(
            initial_state,
            config={"recursion_limit": max(4, (self.max_iterations + 1) * 2 + 1)},
        )
        return self._finalize(query, result)

    @staticmethod
    def _last_agent_message(result: AgentState) -> AIMessage | None:
        return next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
        )

    def _generate_answer(self, question: str, chunks: list[dict]) -> dict:
        """Produce the structured answer via a forced `produce_answer` tool call.

        The answer step is separate from the tool-decide step, so there is no
        JSON/tool-call mixing, and Groq returns the fields as validated JSON.
        """
        user_content = (
            f"User question:\n{question}\n\n"
            f"Retrieved chunks:\n{self._format_chunks(chunks)}"
        )
        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        bound = self._structured_answer_model()
        for _ in range(2):
            try:
                resp = bound.invoke(messages)
            except Exception:  # noqa: BLE001
                continue
            args = self._extract_tool_args(resp)
            if args:
                return args
        return {}

    def _select_contexts(self, answer: str, chunks: list[dict]) -> list[dict]:
        """Pick the chunks that best support the *answer*.

        The model's own index guesses are unreliable, so we embed the answer
        and retrieve the closest chunks - but only from the chunks that were
        actually retrieved for the user's question. This guarantees the shown
        context backs the answer without pulling in unrelated movies.
        """
        if not chunks:
            return []
        if "do not have that information" in answer.lower():
            return []
        retrieved_ids = {c.get("chunk_id") for c in chunks}
        try:
            if answer:
                matched = [
                    c
                    for c in self._search(answer, top_k=len(chunks) + 2)
                    if c.get("chunk_id") in retrieved_ids
                ]
                if matched:
                    return matched[:2]
        except Exception:  # noqa: BLE001
            pass
        return chunks[:1]

    def _finalize(self, query: str, state: AgentState) -> dict:
        chunks = state.get("chunks", [])
        last = self._last_agent_message(state)
        last_content = self._clean_text(getattr(last, "content", "")) if last else ""

        if last_content.upper().startswith("CLARIFY:") and self._query_is_vague(query):
            clarifying = last_content.split(":", 1)[1].strip()
            return {
                "answer": "",
                "contexts": [],
                "reasoning": (
                    "The question was too vague to search for, so I asked the "
                    "user for more details."
                ),
                "clarifying_question": clarifying,
            }

        # Safety net: the model sometimes asks for clarification even when the
        # query has searchable content (e.g. "something about comedy"). In that
        # case recover by running retrieval ourselves instead of dead-ending.
        if not chunks:
            chunks = self._search_merged(query, query, self.top_k)

        parsed = self._generate_answer(query, chunks)
        answer = self._clean_text(parsed.get("answer"))

        relevant = self._select_contexts(answer, chunks)
        contexts = [
            {
                "title": c.get("Title"),
                "release_year": c.get("Release Year"),
                "origin": c.get("Origin/Ethnicity"),
                "genre": c.get("Genre"),
                "snippet": self._clean_text(self._trim_snippet(c["chunk_text"], self.snippet_length)),
                "similarity": round(float(c.get("similarity", 0.0)), 4),
            }
            for c in relevant
        ]

        result: dict[str, Any] = {
            "answer": answer,
            "contexts": [c["snippet"] for c in contexts],
            "reasoning": self._clean_text(
                parsed.get(
                    "reasoning",
                    "I searched the movie plot summaries and used the most relevant "
                    "retrieved chunks to form this answer.",
                )
            ),
        }
        if parsed.get("clarifying_question") and not answer:
            result["clarifying_question"] = self._clean_text(parsed["clarifying_question"])
        return result

    @staticmethod
    def _query_is_vague(query: str) -> bool:
        """True only when the query has no searchable content (no title, genre,
        character, setting, plot element or theme). E.g. "tell me about a movie"
        -> vague; "something about comedy" -> NOT vague."""
        return not bool(_content_tokens(query))

    @staticmethod
    def _trim_snippet(text: str, limit: int) -> str:
        """Trim to `limit` chars at a word boundary (no mid-word cuts)."""
        text = text.strip()
        if len(text) <= limit:
            return text
        cut = text[:limit]
        idx = cut.rfind(" ")
        if idx > limit // 2:
            cut = cut[:idx]
        return cut.strip()

    @staticmethod
    def _clean_text(text: Any) -> str:
        """Make a string safe for clean JSON: no double quotes, no line breaks."""
        if text is None:
            return ""
        cleaned = str(text).replace('"', "'")
        cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
