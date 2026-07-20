# SEC Filing RAG

Financial question answering over SEC filings for BU493 Machine Learning in
Finance. The project downloads a fixed filing set, cleans and chunks it, builds
FAISS or BM25 indexes, asks Claude to answer evaluation questions, and scores
both answer quality and retrieval quality.

The default companies are `META`, `AMZN`, `AAPL`, `NFLX`, and `GOOG`.

## Setup

Requirements:

- Python 3.11+
- an SEC identity for downloads;
- an OpenAI API key for FAISS and semantic answer scoring;
- an Anthropic API key for answer generation and optional LLM judging.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure the root `.env`:

```dotenv
SEC_IDENTITY="Your Name your.email@example.com"
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
```

Every entry point loads this shared file without overriding variables already
present in the shell. The keys are required only under these conditions:

| Setting | Used for | Not required when |
| --- | --- | --- |
| `SEC_IDENTITY` | SEC EDGAR downloads through EdgarTools | compatible filing datasets already exist and `--skip-build` is used |
| `OPENAI_API_KEY` | chunk/query embeddings for FAISS; answer/reference embeddings during default scoring | using BM25 and scoring with `--no-embeddings` |
| `ANTHROPIC_API_KEY` | Claude answer generation; optional `--llm-judge` | only building data/indexes or running non-LLM scoring |

OpenAI and Anthropic clients use a 60-second timeout and up to three SDK
retries. API calls can incur usage charges. Keep `.env` private; it is ignored
by Git.

## Dataset defaults

Each ticker contains exactly:

- the latest unamended 10-K;
- the latest unamended 10-Q; and
- the single latest unamended 8-K.

Selection is always frozen at **July 14, 2026**. Filings after that date are
excluded. `--num-8k N` can explicitly request more historical 8-K filings, but
the default is always one.

```powershell
# Build every default ticker and the combined FAANG analysis dataset.
python -m src.data.build_dataset

# Build one ticker only.
python -m src.data.build_dataset --ticker AMZN
```

Ticker builds are transactional: all stages are generated and validated in a
staging directory before the ticker's live artifacts are replaced. A failed
build leaves the previous valid ticker dataset in place. After a successful
replacement, obsolete files are removed only from that ticker's generated raw,
human-readable, processed, index, and summary locations.
Compatible cached datasets receive the same manifest-scoped pruning before
evaluation, without contacting EDGAR.

Running `src.data.build_dataset` without ticker flags also creates a combined
`data/processed/faang/` chunk dataset and aggregate descriptive reports.
`src.run_tests.run_eval`, by contrast, validates or builds each requested ticker
independently and does not build the combined dataset. Evaluation retrieval is
ticker-scoped, so the combined dataset is unnecessary for RAG correctness; it
exists for aggregate inspection and reporting.

See [DATAREADME.md](DATAREADME.md) for schemas and generated data details.

## Retrieval and evaluation

```powershell
# Default: milestone questions, completed answers only, FAISS, k=5.
python -m src.run_tests.run_eval --skip-build

# Recommended comparison run.
python -m src.run_tests.run_eval --retriever both --skip-build

# BM25 avoids OpenAI embedding calls during retrieval.
python -m src.run_tests.run_eval --retriever bm25 --skip-build
```

The default evaluation file is
`eval_sets/faang_eval_set_milestone.csv`. It contains the planned 60 questions,
but only rows with nonblank reference answers are run by default. This prevents
unfinished questions from silently entering scored results. Use
`--include-incomplete` to retrieve and generate for all selected rows; answer
metrics remain unavailable when the reference answer is blank.

The runner validates every exact `source_doc_id` against the ticker's chunk
database before API calls. Multi-filing questions use a marker such as
`AMZN_MULTI`; when known, their exact contributing documents belong in the
pipe-separated `source_doc_ids` column. A `_MULTI` marker without exact IDs is
allowed, but document Recall@k and reciprocal rank are unavailable for that
row. Missing `source_chunk_ids` are handled the same way for chunk metrics.

### Why the indexes differ

- **FAISS** stores dense OpenAI embedding vectors and searches by semantic
  similarity. It can match related wording, but building and querying it needs
  OpenAI calls.
- **BM25** stores sparse token statistics and ranks literal term overlap. It is
  local and inexpensive but less capable of matching paraphrases.

Only the requested index is built. FAISS is the default, so a default run does
not spend time and storage building BM25 when it will not be queried. Use
`--retriever both` to build and evaluate both.

Each index has a manifest covering its schema, retriever settings, chunk CSV
SHA-256, row order, and embedding model/dimension where applicable. Missing,
stale, or corrupt requested indexes are rebuilt automatically. FAISS and BM25
share one chunk metadata snapshot instead of duplicating it.

### Prompt and citations

Claude receives only the retrieved chunk text plus a system prompt that tells
it to treat excerpts as untrusted data, use no unsupported outside facts,
preserve dates/units/periods, label calculations, disclose conflicts or missing
evidence, and cite supporting excerpts as `[Chunk N]`. It normally returns a
short answer followed by a `Sources:` line.

These labels are answer-level citations to the numbered context supplied in
that request. The result CSV separately records the corresponding stable
`retrieved_chunk_ids`, document IDs, scores, sections, and filing dates. Passing
chunks to the model supplies evidence; requiring and preserving citation labels
makes the model's claimed support inspectable. The pipeline does not currently
compute a separate groundedness or faithfulness score.

### Checkpoint and resume behavior

The default output is `outputs/eval_results/eval_results.csv`. Each attempted
`(qa_id, retriever)` record is flushed immediately to:

- `eval_results.csv.checkpoint.jsonl` - durable last-write-wins records;
- `eval_results.csv.manifest.json` - run configuration and fingerprints.

The final CSV is replaced atomically from the checkpoint. Retrieval and answer
generation have separate statuses and errors, so retrieved evidence is retained
when Claude fails. If a process is interrupted, already completed records are
also materialized to the CSV.

```powershell
python -m src.run_tests.run_eval --retriever both --skip-build --resume
```

`--resume` continues only an exact match of the evaluation CSV and selected
rows, prompt, model, `k`, retrievers, chunk data, and index fingerprints. It
skips successful rows and retries missing or failed rows. A mismatch stops with
an instruction to start a fresh run or use a different output path; incompatible
runs are never appended together.

Legacy result files and indexes are intentionally unsupported after this
clean-break migration. Rerun evaluation/index construction to recreate them.

## Scoring

```powershell
# Retrieval metrics + word overlap + OpenAI semantic similarity.
python -m src.run_tests.score_eval

# Fully local scoring.
python -m src.run_tests.score_eval --no-embeddings

# Add optional Claude 1-5 factual-alignment judging.
python -m src.run_tests.score_eval --llm-judge
```

The scorer validates the v2 result schema and single run fingerprint. It keeps
all rows, including failures, and reports metric denominators plus retrieval and
generation success rates. Retrieval metrics are computed whenever retrieval
succeeded, even when answer generation failed. Answer metrics require both a
successful generation and a nonblank reference answer.

| Metric | Meaning | API |
| --- | --- | --- |
| `retrieval_success` | Fraction of rows whose retriever returned context | None |
| `generation_success` | Fraction of rows with a Claude answer | None |
| `document_recall_at_k` | Fraction of exact gold documents found in top-k | None |
| `document_reciprocal_rank` | Reciprocal rank of the first gold document | None |
| `chunk_recall_at_k` | Fraction of exact gold chunks found in top-k | None |
| `chunk_reciprocal_rank` | Reciprocal rank of the first gold chunk | None |
| `word_overlap_f1` | Frequency-aware unigram precision/recall F1 | None |
| `cosine_sim` | Embedding similarity between answer and reference | OpenAI |
| `llm_score` | Strict Claude factual-alignment score from 1 to 5 | Anthropic |

Unavailable gold labels, failed stages, and incomplete references produce blank
metric cells rather than fabricated zeros. LLM judge failures are recorded per
row instead of aborting the entire scoring run.

Charts are generated when enough data is available:

- `evaluation_coverage.png`;
- `filing_timeline.png`;
- `retriever_comparison.png` (includes stage success rates);
- `per_question_delta.png` for paired FAISS/BM25 runs.

## Project structure

```text
eval_sets/
  faang_eval_set_milestone.csv
src/
  config.py
  data/                 download, clean, chunk, transactional build, reports
  ingest_data/          FAISS/BM25 builders and index manifests
  LLM_response/         retrieval, prompt, durable evaluation
  run_tests/            evaluation and scoring CLIs
tests/                  offline unit and fault-injection tests
data/                   generated filing and index artifacts
outputs/                generated summaries, results, and charts
```
