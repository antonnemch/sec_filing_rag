# SEC Filing RAG

BU493 Machine Learning in Finance — financial question answering over SEC filings.

The pipeline downloads SEC filings, cleans and chunks them, builds retrieval indexes (FAISS dense embeddings or BM25 sparse keyword), and runs a RAG evaluation comparing both retrieval methods using an LLM (Claude) to answer questions from `eval_sets/faang_eval_set_dummy.csv`.

The default dataset covers the FAANG tickers `META`, `AMZN`, `AAPL`, `NFLX`, and `GOOG`.

---

## Prerequisites

- Python 3.11+
- An SEC identity (free — required by EDGAR for automated access)
- An OpenAI API key (for embeddings)
- An Anthropic API key (for LLM answers)

---

## Setup

**1. Create and activate a virtual environment**

```bash
cd sec_filing_rag
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your SEC identity**

EDGAR requires every automated client to identify a responsible person.

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
SEC_IDENTITY="Your Name your.email@example.com"
```

**4. Set API keys**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Running the full evaluation

All commands below must be run from the `sec_filing_rag/` directory.

### Step 1 — Build filing datasets

Downloads, cleans, and chunks SEC filings for each ticker into metadata-rich word chunks and data summaries. Each ticker takes ~1–2 minutes. You only need to run this once; the data is saved to `data/`.

```bash
# Build all 5 FAANG tickers (META, AMZN, AAPL, NFLX, GOOG)
for ticker in META AMZN AAPL NFLX GOOG; do
    python -m src.data.build_dataset --ticker $ticker --num-8k 5
done
```

Or build a single ticker:

```bash
python -m src.data.build_dataset --ticker AMZN --num-8k 5
```

This produces for each ticker:
- `data/raw/{ticker}/` — downloaded HTML/text filings
- `data/processed/{ticker}/{ticker}_filing_chunks.csv` — chunked text ready for retrieval
- `outputs/data_summary/{ticker}_dataset_summary.json` — summary stats

### Step 2 — Run the RAG evaluation

Runs the full pipeline: build retrieval indexes → retrieve chunks → call Claude → save answers.

```bash
# Run both FAISS and BM25 retrievers (recommended — enables comparison)
python -m src.run_tests.run_eval --retriever both --skip-build

# Run only one retriever
python -m src.run_tests.run_eval --retriever faiss --skip-build
python -m src.run_tests.run_eval --retriever bm25  --skip-build
```

Results are saved to `outputs/eval_results/eval_results.csv` with one row per question per retriever (120 rows total when using `--retriever both`).

If the run is interrupted, resume from where it left off:

```bash
python -m src.run_tests.run_eval --retriever both --skip-build --resume
```

### Step 3 — Score and compare

Embeds all LLM answers and ground truth answers with OpenAI, computes cosine similarity between them, and prints a side-by-side FAISS vs BM25 comparison.

```bash
python -m src.run_tests.score_eval
```

To skip the embedding API call and use word-overlap (F1) only:

```bash
python -m src.run_tests.score_eval --no-embeddings
```

To add LLM-as-judge scores (1–5 per answer, uses claude-haiku):

```bash
python -m src.run_tests.score_eval --llm-judge
```

Scored results are saved to `outputs/eval_results/eval_results_scored.csv`.

---

## Project structure

```
sec_filing_rag/
├── eval_sets/
│   └── faang_eval_set_dummy.csv   # 60 questions across META/AMZN/AAPL/NFLX/GOOG
├── src/
│   ├── data/                      # Download → clean → chunk pipeline
│   │   ├── download_filings.py
│   │   ├── clean_filings.py
│   │   ├── chunk_filings.py
│   │   ├── build_dataset.py       # Runs all three stages
│   │   └── utils.py
│   ├── injest_data/               # Index builders
│   │   ├── embeddings.py          # FAISS (OpenAI text-embedding-3-small)
│   │   └── bm25.py                # BM25 (rank-bm25)
│   ├── LLM_response/              # RAG pipeline
│   │   ├── retrieve_context.py    # Wraps FAISS and BM25 retrieval
│   │   ├── LLM.py                 # Claude answer generation
│   │   ├── ground_truth.py        # Loads the eval CSV
│   │   └── batch_eval.py          # Eval loop — retrieve → LLM → save
│   └── run_tests/
│       ├── run_eval.py            # Entry point: build indexes + run eval
│       └── score_eval.py          # Scoring + FAISS vs BM25 comparison
├── data/                          # Generated — not committed
├── outputs/                       # Generated — not committed
├── .env.example                   # Copy to .env and fill in SEC identity
└── requirements.txt
```

---

## Metrics

| Metric | How it works | API needed |
|--------|-------------|------------|
| `cosine_sim` | OpenAI embedding of LLM answer vs ground truth — measures semantic agreement | OpenAI |
| `word_overlap_f1` | ROUGE-1 F1 word overlap — fast sanity check | None |
| `llm_score` | Claude rates each answer 1–5 vs ground truth | Anthropic |

**Note:** The ground truth answers in `faang_eval_set_dummy.csv` are dummy placeholders. `cosine_sim` and `llm_score` are more meaningful than `word_overlap_f1` for this reason.

---

## Individual pipeline stages

Each stage can be run independently:

```bash
python -m src.data.download_filings --ticker AMZN --num-8k 5
python -m src.data.clean_filings    --ticker AMZN
python -m src.data.chunk_filings    --ticker AMZN --chunk-size 400 --chunk-overlap 75
```

See [DATAREADME.md](DATAREADME.md) for data schemas and pipeline details.
