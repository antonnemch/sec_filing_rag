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

# Optional; number of chunks retrieved for each question.
RAG_TOP_K=5
```

Every entry point loads this shared file without overriding variables already
present in the shell. The keys are required only under these conditions:

| Setting | Used for | Not required when |
| --- | --- | --- |
| `SEC_IDENTITY` | SEC EDGAR downloads through EdgarTools | compatible filing datasets already exist and `--skip-build` is used |
| `OPENAI_API_KEY` | chunk/query embeddings for FAISS; answer/reference embeddings during default scoring | using BM25 and scoring with `--no-embeddings` |
| `ANTHROPIC_API_KEY` | Claude answer generation; optional `--llm-judge` | only building data/indexes or running non-LLM scoring |
| `RAG_TOP_K` | Default number of chunks retrieved per evaluation question | omitted to use the built-in default of `5`; `--k` overrides it |

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
`src.evaluation.run_eval`, by contrast, validates or builds each requested ticker
independently and does not build the combined dataset. Evaluation retrieval is
ticker-scoped, so the combined dataset is unnecessary for RAG correctness; it
exists for aggregate inspection and reporting.

See [DATAREADME.md](DATAREADME.md) for schemas and generated data details.

## Retrieval and evaluation

```powershell
# Default: complete 60-question set, FAISS, k=5, new timestamped run directory.
python -m src.evaluation.run_eval --skip-build

# Recommended named comparison run (easy to score or resume later).
python -m src.evaluation.run_eval --retriever both --skip-build --run-name baseline-k5

# BM25 avoids OpenAI embedding calls during retrieval.
python -m src.evaluation.run_eval --retriever bm25 --skip-build
```

Retrieval depth uses this precedence: an explicit `--k` argument, then
`RAG_TOP_K` from the shell or root `.env`, then the built-in default of `5`.
For example, `RAG_TOP_K=10` changes the default for subsequent runs, while
`python -m src.evaluation.run_eval --k 3` still retrieves three chunks for that
run. Both values must be integers of at least `1`.

The default evaluation file is `eval_sets/faang_eval_set_complete.csv`. It
contains 60 questions with completed reference answers, so a default run selects
all 60. The completed-answer filter remains active for custom evaluation files;
use `--include-incomplete` when a custom file intentionally contains blank
references, in which case answer metrics for those rows remain unavailable.

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

Retrieval also uses filing metadata. If a question explicitly names a 10-K,
10-Q, or 8-K, candidates are restricted to the named form or forms. Comparison
questions reserve context for every named form. BM25 applies these values only
as filters; they are not repeated in its indexed text. FAISS embeds each chunk
together with its company, ticker, filing type, filing date, and normalized SEC
section heading so semantic search can distinguish a filing cover page from a
substantive item disclosure.

The cleaner recognizes split 8-K headings such as `Item 5.07.` followed by its
title, demotes repeated table-of-contents item headings, and omits page-number-
only sections. Chunk schema and dense-document format versions make datasets
and FAISS indexes built before this metadata repair stale. Run the normal
pipeline without `--skip-build` once to transactionally rebuild them; subsequent
runs reuse the validated artifacts.

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

Every fresh evaluation is isolated under
`outputs/eval_results/runs/<run-name>/`. Without `--run-name`, the runner creates
a unique UTC timestamp such as `20260720T190504_123456Z`. A run contains:

- `eval_results.csv` - materialized results;
- `eval_results.csv.checkpoint.jsonl` - durable last-write-wins records;
- `eval_results.csv.manifest.json` - run configuration and fingerprints;
- `eval_results_scored.csv` - metrics created by the scorer;
- `figures/` - that run's input, coverage, and performance charts.

Fresh runs refuse to overwrite an existing result, checkpoint, manifest, or
named run directory. A custom `--output` remains available; its checkpoint,
manifest, scored output, and figures are stored beside that CSV.

The final CSV is replaced atomically from the checkpoint. Retrieval and answer
generation have separate statuses and errors, so retrieved evidence is retained
when Claude fails. If a process is interrupted, already completed records are
also materialized to the CSV.

```powershell
python -m src.evaluation.run_eval --retriever both --skip-build --run-name baseline-k5 --resume
```

`--resume` continues only an exact match of the evaluation CSV and selected
rows, prompt, model, `k`, retrievers, chunk data, and index fingerprints. It
skips successful rows and retries missing or failed rows. A mismatch stops with
an instruction to start a fresh run or use a different output path; incompatible
runs are never appended together. Because an automatically generated timestamp
cannot identify an earlier run, `--resume` requires either `--run-name` or
`--output`.

Legacy result schemas and index formats remain intentionally unsupported after
the clean-break migration. Rebuild those artifacts with the current pipeline.

## Scoring

```powershell
# Score the named run used above.
python -m src.evaluation.score_eval --run-name baseline-k5

# With no input option, score the newest run.
python -m src.evaluation.score_eval

# Fully local scoring or optional Claude judging.
python -m src.evaluation.score_eval --run-name baseline-k5 --no-embeddings
python -m src.evaluation.score_eval --run-name baseline-k5 --llm-judge
```

Scoring also refuses to replace an existing scored CSV or performance figure.
Use `--overwrite` only when intentionally rescoring the same run, or provide an
`--output` in another directory to isolate a different scoring variant and its
figures.

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

Charts are written to the selected run's `figures/` directory. Input/run charts
are created during evaluation, and performance charts are created during scoring.
A chart is skipped cleanly when its required metric or pairing is unavailable:

- `01_evaluation_coverage.png`: completed, incomplete, and multi-document QA rows;
- `02_filing_timeline.png`: selected 10-K, 10-Q, and 8-K filings against the cutoff;
- `03_overall_retriever_metrics.png`: every available metric by retriever;
- `04_metric_availability_and_outcomes.png`: eligible denominators and failure counts;
- `05_category_performance.png`: every available metric by category and retriever;
- `06_category_retriever_deltas.png`: paired FAISS-minus-BM25 category differences;
- `07_ticker_performance.png`: every available metric by company and retriever;
- `08_answerability_performance.png`: performance on answerable/unanswerable rows;
- `09_metric_distributions.png`: row-level score distributions by retriever;
- `10_per_question_deltas.png`: paired FAISS-minus-BM25 question differences;
- `11_metric_correlations.png`: within-retriever agreement between metrics.

## Project structure

```text
eval_sets/
  faang_eval_set_complete.csv       default 60-question evaluation set
  faang_eval_set_milestone.csv      earlier milestone snapshot
src/
  config.py
  data/                 download, clean, chunk, transactional build, reports
  ingest_data/          FAISS/BM25 builders and index manifests
  LLM_response/         retrieval, prompt, durable evaluation
  evaluation/           evaluation and scoring CLIs
tests/                  offline unit and fault-injection tests
data/                   generated filing and index artifacts
outputs/                generated summaries and evaluation results
  eval_results/runs/<run-name>/
    eval_results.csv    isolated results, checkpoint, manifest, and scored CSV
    figures/            figures belonging only to this run
```
