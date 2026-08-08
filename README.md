# Mini RAG System — Movie Plots

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions
about movie plots from a subset of the
[Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
dataset.

## Live demo

Try the deployed app: **https://movierag.streamlit.app/**

The system works entirely from the command line, but a Streamlit web UI is
included purely for user ease — anyone can ask questions about movie plots
directly in the browser without touching any code.

1. Loads the dataset and samples a **subset (500 movies)**.
2. Preprocesses it (drop duplicates, missing values, `Unknown`/`unknown` records).
3. **Chunks** long plot texts into ~**300-word** sentence-aware chunks.
4. **Embeds** each chunk with `sentence-transformers/BAAI/bge-small-en-v1.5`.
5. Stores chunks in a **Chroma vector store** (cosine similarity, top-k retrieval
   with a similarity threshold).
6. A **single LangGraph agent** (Groq LLM) retrieves top-k chunks with a
   `retrieve_movie_plots` tool, decides whether to ask a clarifying question,
   and answers grounded on the retrieved context.
7. Outputs **structured JSON**: `answer`, `contexts`, `reasoning`
   (+ `clarifying_question` when the query is vague).

### Example output

```json
{
  "answer": "Billy Elliot is a 2000 British film about an 11-year-old boy who secretly takes ballet lessons. He lives with his widowed father Jackie and older brother Tony, both coal miners out on strike; his father sends him to the gym for boxing, but Billy secretly practices ballet instead.",
  "contexts": [
    "Billy Elliot, an 11-year-old from the fictional Everington in County Durham, England, loves to dance and has hopes of becoming a professional ballet dancer. Billy lives with his widowed father, Jackie, and older brother, Tony, both coal miners out on strike"
  ],
  "reasoning": "I searched the plot summaries for a story about an 11-year-old boy secretly taking ballet lessons and matched Billy Elliot."
}
```

## Project structure

```
movie-rag-system/
├── config/config.yaml      # all pipeline settings (models, sizes, thresholds)
├── data/                   # dataset + Chroma store (created at runtime)
├── src/
│   ├── data_loader.py      # load the CSV
│   ├── preprocessor.py     # cleaning (duplicates / missing / unknown)
│   ├── chunker.py          # 300-word sentence-aware chunking
│   ├── embeddings.py       # sentence-transformers encoder
│   ├── retriever.py        # Chroma vector store wrapper
│   ├── agent.py            # single LangGraph agent (retrieve tool, clarification)
│   └── rag_graph.py        # end-to-end pipeline orchestration
├── notebooks/              # EDA + preprocessing notebooks
├── tests/                  # pytest unit tests
├── main.py                 # CLI
└── app.py                  # Streamlit web UI
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Groq API key (Groq is the only LLM provider)
cp .env.example .env              # then fill in GROQ_API_KEY

# On Windows:  $env:GROQ_API_KEY="gsk_..."
# On macOS/Linux: export GROQ_API_KEY="gsk_..."
```

> **Dataset** (optional once, ~80 MB): download the
> [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
> dataset (`wiki_movie_plots_deduped.csv`) from Kaggle and save it **exactly** at:
>
> ```
> movie-rag-system/data/wiki_movie_plots_deduped.csv
> ```
>
> You only need this file if the 500-movie subset
> (`data/subset.csv`, already committed to the repo) is missing or you want to
> regenerate it with a different `data.subset_size` / `data.random_seed`.
> `build-index` preprocesses it automatically (drops duplicates, missing values
> and `Unknown`/`unknown` records) before sampling the subset. The file is
> **gitignored** (`.gitignore` line 15) because of its size, so it never enters
> version control — it lives only on your machine.

> **No API key?** Use `--no-llm` to demo retrieval-only mode (no LLM call).

## Usage

### 1. Build the index (subset → chunk → embed → store)

```bash
python main.py build-index
```

### 2. Ask a question (full RAG)

```bash
python main.py query "which movie is about an 11-year-old boy who secretly takes ballet lessons?"
```

### 3. Retrieval only (no LLM / no API key)

```bash
python main.py query "a monster attack" --no-llm
```

### 4. Streamlit UI

```bash
streamlit run app.py
```

A simple web UI with a question box, `top-k` slider, and an optional
retrieval-only mode. Open the URL Streamlit prints (default `http://localhost:8501`).

> **Already deployed** — use the live version at
> **https://movierag.streamlit.app/** (hosted on Streamlit Community Cloud).

### Sample questions

Try these against the fixed 500-movie subset (all verified to retrieve the
right movie in the top-3 chunks). They mix **fact extraction** (specific
details pulled from a plot) with **content-based questions** (no title given,
so they test semantic retrieval rather than the title boost):

*Fact extraction (title given):*

1. What is the plot of 4D Man?
2. In Poltergeist II: The Other Side, who is the main villain?
3. What is the Bengal tiger in Life of Pi called?
4. In Billy Elliot, what sport does his father want him to learn?
5. What kind of dinosaur is Patchi in Walking with Dinosaurs?
6. Who is the imprisoned author in Quills?
7. How does Count Dracula get revived in Dracula Has Risen from the Grave?
8. Where does Robert McCall work in The Equalizer?
9. What do the Boxtrolls wear?
10. What does young David see land in the sandpit in Invaders from Mars?
11. What element do the stranded aliens need in The Lost Skeleton of Cadavra?
12. Who is the second son of Henry Frankenstein in The Ghost of Frankenstein?
13. What begins to haunt Dr. Peter Proud in The Reincarnation of Peter Proud?
14. What is the album the band promotes in This Is Spinal Tap?
15. What is the name of the abused dog in Shiloh?

*Content-based (no title mentioned):*

16. Which movie features a Bengal tiger named Richard Parker?
17. A boy named Eggs is raised by underground trolls who wear cardboard boxes — which movie is this?
18. Which film tells the story of an 11-year-old boy who secretly takes ballet lessons?
19. Which movie is about a retired CIA operative who works in a hardware store?
20. Which movie follows a family haunted by an evil spirit in Cuesta Verde?

Questions about movies outside the subset answer honestly with *"I do not have
that information in my sources."*, and vague questions (e.g. *"tell me about a
movie"*) return a `clarifying_question` instead of guessing.

### Configuration

Everything is tunable in `config/config.yaml`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `data.subset_size` | 500 | number of movies randomly sampled |
| `data.random_seed` | 42 | deterministic subset |
| `chunking.chunk_size` | 300 | words per chunk |
| `retriever.top_k` | 3 | chunks retrieved per query |
| `retriever.similarity_threshold` | 0.3 | minimum cosine similarity; weaker chunks are dropped |
| `retriever.title_boost` | 0.4 | keyword boost for chunks whose Title matches the query |
| `embeddings.model` | BAAI/bge-small-en-v1.5 | embedding model |
| `llm.model` | llama-3.3-70b-versatile | agent's LLM (Groq) |
| `agent.max_iterations` | 3 | max agent/tool loop rounds |

## Choosing the chunk size (data-driven)

The 300-word window is not pulled from the task example — it is the measured
sweet spot for this dataset + embedding model:

- Plot-length analysis of the fixed `data/subset.csv` (500 movies):
  **median 348 words, mean 402, 75th percentile 615, max 2245**.
- A ~300-word window therefore splits a typical movie into **~2 tight,
  topically-focused chunks** (951 total) instead of one diluted mega-chunk.
- We benchmarked retrieval on the 20 sample questions below for several sizes:

  | chunk size | chunks | correct movie in top-3 |
  | --- | --- | --- |
  | 300 | 951 | **20/20** |
  | 350 | 862 | 19/20 |
  | 400 | 789 | 18/20 |

  Bigger chunks over-dilute the small open-source encoder
  (`BAAI/bge-small-en-v1.5`), hurting content-based queries (no title in the
  question). The agent still gets the whole story because top-3 chunks cover
  the full plot of a typical movie. Re-run the benchmark after changing
  `chunking.chunk_size` in `config/config.yaml`.

## Tests

```bash
python -m pytest tests/ -q
```

## How the pipeline works (end-to-end)

1. **Load & subset** — `DataLoader` reads the raw CSV, `Preprocessor` cleans it,
   and `RAGPipeline` samples 500 rows.
2. **Chunk** — `Chunker` splits each plot into ~300-word chunks at sentence
   boundaries (oversized sentences are hard-split).
3. **Embed** — `Embedder` encodes every chunk with `BAAI/bge-small-en-v1.5`
   (384-dim, L2-normalised).
4. **Store & retrieve** — `Retriever` stores the vectors in a Chroma collection
   (`hnsw:space=cosine`) on disk under `data/chroma_store/`, ranks chunks by
   cosine similarity, and drops any chunk below `retriever.similarity_threshold`.
   Retrieval is **hybrid**: pure vector similarity is combined with a keyword
   boost on the chunk's `Title` metadata.

### Why hybrid retrieval?

Dense vector embeddings map sentences into a continuous semantic space, which excels at capturing conceptual meanings (e.g., matching `"an 11-year-old boy secretly takes ballet"` to *Billy Elliot*). However, dense models frequently suffer from the "vocabulary mismatch" problem and struggle with exact keyword matching or retrieving specific proper nouns (e.g., queries containing exact titles like `"Poltergeist II"`).

To resolve this, the system implements a dense-sparse hybrid retrieval strategy. Chunks are retrieved via vector cosine similarity, and then a deterministic metadata-based keyword boost (`retriever.title_boost`, default `0.4`) is applied to the movie `Title`. This hybrid mechanism ensures that specific entity-driven queries are reliably surfaced even if the underlying embedding model ranks them lower, providing production-grade retrieval accuracy without the high API costs or network latency associated with commercial embedding models. Set `title_boost` to `0` in `config/config.yaml` to fall back to pure vector retrieval.
5. **Agent** — a single LangGraph agent (ReAct loop) is bound to the
   `retrieve_movie_plots` tool:
   - The agent calls the tool to fetch top-k chunks for the user's question.
   - If the question is **vague**, it responds with a `clarifying_question`
     instead of guessing.
   - If no retrieved chunk is relevant, it says
     *"I do not have that information in my sources."*
   - Otherwise it emits a JSON `{answer, reasoning}` grounded only on the context.
### 6. Output

`main.py` prints the structured JSON; `contexts` is always built from the
*actual* retrieved chunks (never hallucinated by the model).

```bash
python main.py query "which movie is about an 11-year-old boy who secretly takes ballet lessons?"
```

```json
{
  "answer": "Billy Elliot is a 2000 British film about an 11-year-old boy who secretly takes ballet lessons. He lives with his widowed father Jackie and older brother Tony, both coal miners out on strike; his father sends him to the gym for boxing, but Billy secretly practices ballet instead.",
  "contexts": [
    "Billy Elliot, an 11-year-old from the fictional Everington in County Durham, England, loves to dance and has hopes of becoming a professional ballet dancer. Billy lives with his widowed father, Jackie, and older brother, Tony, both coal miners out on strike"
  ],
  "reasoning": "I searched the plot summaries for a story about an 11-year-old boy secretly taking ballet lessons and matched Billy Elliot."
}
```
