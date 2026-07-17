# RAG Agent for Visigothic Hispania Historiography

A multi-phase Retrieval-Augmented Generation (RAG) agent built with **LangGraph** that answers academic questions about the historiography of Visigothic Hispania. It retrieves paragraphs from a PostgreSQL + pgvector corpus of academic sources and generates cited, APA7-formatted answers in Spanish through three progressive phases: initial answer → contextualization → iterative gap-based refinement.

## How it works

The agent runs a linear LangGraph pipeline:

```
prepare_question → rag_main_question → generate_initial_answer
    → generate_sub_questions → rag_sub_questions → rerank → process_resources
    → generate_contextualized_answer → verify_and_refine → END
```

**Preparation.** The question is analyzed to extract keywords (Ollama), NER entities (spaCy, multilingual), Latin terms (wordlist lookup), publication-year ranges (Ollama, JSON mode), and author names matched against the database and a biography file.

**Phase 1 — Initial answer.** A tiered semantic search retrieves the top-K paragraphs (Tier 1: persons/authors, Tier 2: Latin terms, Tier 3: places, Tier 4: pure semantic), applies keyword/NER/Latin score boosts, and generates a direct answer using only those paragraphs.

**Phase 2 — Contextualization.** Sub-questions are generated to find framing context (never new facts). Extra paragraphs, glossary definitions, author biographies, and historiographical currents are retrieved, and the answer is re-written with added framing — only where it helps.

**Phase 3 — Refinement.** Up to `MAX_ITERATIONS` cycles: an LLM identifies concrete missing data (gaps), new fragments are retrieved per gap, and the answer is refined. Stops when no gaps remain or no new fragments are found.

## Requirements

- Python 3.10+
- PostgreSQL with the **pgvector** extension
- [Ollama](https://ollama.com) running with the `qwen3:8b` model pulled
- An OpenAI API key

```bash
pip install spacy langdetect python-dotenv langgraph langchain-core psycopg2-binary openai ollama

python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download fr_core_news_sm
```

Missing spaCy models degrade gracefully (a warning is printed and a regex fallback is used).

## Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434
PG_DBNAME=your_db
PG_USER=your_user
PG_PASS=your_password
PG_HOST=localhost
PG_PORT=5432
```

Tunable constants at the top of the script: `TOP_K`, `TOP_K_RERANK`, `MIN_LENGTH`, `MAX_ITERATIONS`, `FRAGMENTS_PER_GAP`, boost weights (`SEMANTIC_WEIGHT`, `KEYWORDS_WEIGHT`, `BOOST_PER_LATIN_TERM`, `BOOST_PER_PERSON`, `BOOST_PER_PLACE`, `MAX_NER_BOOST`), and the model names (`GENERATION_MODEL`, `CLASSIFICATION_MODEL`, `EMBEDDING_MODEL`, `OLLAMA_MODEL`).

## Required data files

| File | Purpose |
|---|---|
| `biografias.json` | Author name → biography, used for author detection and context |
| `glosario_traducido.json` | Multilingual glossary: term → `{lang: {termino, definicion}}` |
| `corrientes.json` | Historiographical current → description |
| `latin_words.txt` | One Latin word per line (Latin-term boost); optional |
| `lemario.txt` | Spanish lemma list to exclude Spanish words from Latin detection |
| `new_prompts2.json` | Input questions: list of strings or of `{"pregunta": "..."}` objects |

## Database schema

Table `chunks_w_context` with at least:

```sql
texto_completo TEXT,      -- chunk text (min length filter applied)
contexto       TEXT,
titulo_obra    TEXT,
autor          TEXT,
año            TEXT,      -- publication year (numeric string)
pagina         TEXT,
embedding      VECTOR     -- text-embedding-3-large (3072 dims)
```

## Usage

Batch mode (reads `prompts.json`):

```bash
python baddo_agent.py
```

Programmatic:

```python
from baddo_agent import run_agent, show_result, dump_state_to_json

state = run_agent("¿Qué efectos tuvo la conversión de Recaredo?")
show_result(state)
dump_state_to_json(state, "result.json")
```

## Output

- `eval/results.json` — full per-question state dump: answers from every phase, paragraphs used, gaps per refinement cycle, resources used.
- `results.json` — consolidated record of all questions with search configuration, per-phase answers and contexts, refinement history, and agent statistics.

## Notes

- Retrieval filters and boosts are cumulative: structured SQL filters (years, DB authors) apply to Tiers 1–3; year filters also apply to Tier 4.
- Near-duplicate paragraphs from the same author are deduplicated (Jaccard similarity > 0.9).
- The gap identifier is instructed to stay strictly on-topic; already-attempted gaps are excluded from later cycles.
- Data-contract strings remain in Spanish on purpose (DB columns, resource file keys, LLM-returned JSON keys like `rangos`/`esta_completa`); renaming them would break the pipeline.
