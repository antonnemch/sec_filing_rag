# SEC Filing Data Pipeline

## Selection policy

The default dataset covers `META`, `AMZN`, `AAPL`, `NFLX`, and `GOOG`. For each
ticker it selects the latest unamended 10-K, latest unamended 10-Q, and exactly
one latest unamended 8-K whose filing date is no later than **2026-07-14**.
`--num-8k` changes only the requested 8-K count and must be at least one.

SEC EDGAR is the authoritative source. EdgarTools supplies company lookup,
metadata, documents, caching, and SEC-compatible rate limiting. Downloads need
`SEC_IDENTITY="Name email@example.com"` in the root `.env` or shell.

## Commands

```powershell
# Default five-ticker build plus combined FAANG outputs.
python -m src.data.build_dataset

# One ticker only.
python -m src.data.build_dataset --ticker AMZN

# Explicit set.
python -m src.data.build_dataset --tickers META AMZN
```

Defaults are a 400-word maximum chunk size and 75-word within-section overlap.
Validation rejects nonpositive chunk sizes, negative overlap, overlap greater
than or equal to chunk size, duplicate tickers, and `--num-8k` below one.

Individual stages remain available:

```powershell
python -m src.data.download_filings --ticker AMZN
python -m src.data.clean_filings --ticker AMZN
python -m src.data.chunk_filings --ticker AMZN --chunk-size 400 --chunk-overlap 75
```

The complete ticker build is transactional. It performs download, cleaning,
chunking, summary generation, and validation under `.cache/dataset_builds/`.
Only after all checks pass are the exact ticker directories and summary files
swapped into place. Commit failures roll back to the prior artifacts. A
successful replacement removes obsolete files from that ticker's scoped
generated locations, including superseded indexes.
The evaluation preflight also applies manifest-scoped pruning to compatible
cached ticker datasets, so unreferenced files do not accumulate between builds.

Independent cleaning and chunking commands are manifest-driven but are not a
complete transactional ticker rebuild.

## Generated artifacts

For ticker `AMZN`:

```text
data/raw/amzn/
  filing_metadata.json
  <date>_<form>_<accession>.<html|txt>
data/human_readable/amzn/
  <date>_<form>_<accession>.<md|txt>
data/processed/amzn/
  cleaning_manifest.json
  cleaned_filings/*.txt
  amzn_filing_chunks.csv
  amzn_filing_chunks.jsonl
  index_chunks.pkl
  embeddings.faiss                 # when FAISS is requested
  faiss_index_manifest.json        # when FAISS is requested
  bm25_index.pkl                   # when BM25 is requested
  bm25_index_manifest.json         # when BM25 is requested
outputs/data_summary/
  amzn_filing_inventory.csv
  amzn_cleaning_summary.csv
  amzn_dataset_summary.json
```

The default multi-ticker data build additionally writes combined FAANG chunks
and aggregate filing, cleaning, missingness, outlier, section, and descriptive
reports under `data/processed/faang/` and `outputs/data_summary/`.

The RAG evaluation runner does not need or create the combined FAANG chunks. It
retrieves within one company's index for each question. Combined outputs are
for aggregate analysis and communication.

## Data contracts

### Filing manifest

`filing_metadata.json` has one object per selected filing:

| Field | Meaning |
| --- | --- |
| `company`, `ticker`, `cik` | Filing entity identifiers |
| `filing_type`, `filing_date` | SEC form and frozen-policy date |
| `accession_number` | Unique SEC filing identifier |
| `source_url` | SEC primary-document URL |
| `local_raw_path`, `raw_format` | Project-relative raw file and `html`/`text` format |
| `local_markdown_path`, `human_readable_format` | Readable copy and format |
| `human_readable_warning` | Markdown fallback warning, if any |

The cleaning manifest carries these fields forward and adds
`local_cleaned_path`, raw/cleaned character lengths, and warnings.

### Chunk rows

CSV and JSONL contain identical ordered rows:

| Field | Meaning |
| --- | --- |
| `company`, `ticker`, `cik` | Entity metadata |
| `filing_type`, `filing_date`, `accession_number` | Source document identity |
| `source_url`, `source_file` | SEC URL and cleaned local source |
| `chunk_id` | Deterministic ticker/accession/index ID |
| `chunk_index` | Zero-based position within the filing |
| `section_heading` | Detected heading or `unknown` |
| `text`, `word_count` | Retrieval text and size |

Chunks never cross detected section boundaries. Overlap occurs only between
adjacent chunks in the same section. Blank text, duplicate/missing chunk IDs,
unexpected filing counts, dates after the cutoff, and chunks exceeding the
configured size fail staged validation.

### Summary freshness

Each ticker summary records the chunk CSV SHA-256 and row count. Evaluation
refreshes missing or stale inventory, cleaning, and dataset summaries directly
from current manifests and chunks without contacting EDGAR. This keeps
presentation outputs synchronized with the database used for retrieval.

### Index freshness

FAISS and BM25 have different index files because dense vector similarity and
sparse token-ranking require different structures. They share one ordered chunk
metadata snapshot. Each retriever manifest records the source CSV SHA-256, row
count/order fingerprint, schema version, and retriever settings. FAISS also
records the OpenAI embedding model and vector dimension. Loading rejects stale,
tampered, or structurally inconsistent indexes; the evaluation runner rebuilds
the requested index automatically.

## Storage and limitations

Raw, human-readable, processed, cache, and evaluation-result directories are
ignored by Git. Lightweight summary metadata and charts may be retained for
inspection. HTML is preferred; plain text is a fallback. Tables are flattened,
heading detection is conservative, and uncertain disclosure text is retained
rather than aggressively deleted. The dataset is a frozen latest-filing
snapshot, not a historical panel, and separate exhibits are not downloaded.
