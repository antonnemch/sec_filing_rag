# SEC Filing Data

## Purpose

This project uses public SEC EDGAR filings to build the document dataset for a
financial question-answering system. The current milestone covers data
collection, conservative text cleaning, and word-based chunking only. It does
not implement keyword search, embeddings, a vector database, LLM calls, or RAG.

Amazon.com, Inc. (`AMZN`) is the default company because it is a well-known
public filer with substantial annual, quarterly, and current-report
disclosures. The scripts are reusable for another ticker through `--ticker`.

The pipeline collects:

- the most recent unamended Form 10-K;
- the most recent unamended Form 10-Q; and
- a configurable number of recent unamended Form 8-K filings (five by default).

SEC EDGAR is the authoritative source for these primary disclosure documents.
The open-source `edgartools` package handles company lookup, filing metadata,
SEC URLs, document retrieval, and SEC-compatible rate limiting.

## Setup

Use Python 3.11 or newer in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

SEC automated access requires an identity containing a real name and contact
email. Copy the example file and edit the untracked copy:

```powershell
Copy-Item .env.example .env
```

```text
SEC_IDENTITY="Your Name your.email@example.com"
```

The code also accepts the standard `EDGAR_IDENTITY` environment variable as a
fallback. The identity is passed to `edgartools`, is never printed, and must not
be committed. The library's default rate limit remains in effect.

## Build the Dataset

Run the complete default Amazon pipeline:

```powershell
python -m src.data.build_dataset --ticker AMZN --num-8k 5 --chunk-size 400 --chunk-overlap 75
```

Each stage can also run independently:

```powershell
python -m src.data.download_filings --ticker AMZN --num-8k 5
python -m src.data.clean_filings --ticker AMZN
python -m src.data.chunk_filings --ticker AMZN --chunk-size 400 --chunk-overlap 75
```

The cleaning and chunking stages read their preceding manifest rather than
scanning directories. As a result, stale files from an older run cannot enter a
new dataset silently.

Argument rules:

- `--num-8k` must be at least 1.
- `--chunk-size` must be at least 1.
- `--chunk-overlap` must be non-negative and smaller than the chunk size.

## Generated Files

For `AMZN`, a successful run creates:

```text
data/
  raw/amzn/
    filing_metadata.json
    <filing-date>_<form>_<accession>.html
  raw/.edgar_cache/
  processed/amzn/
    cleaning_manifest.json
    cleaned_filings/
      <filing-date>_<form>_<accession>.txt
    amzn_filing_chunks.csv
    amzn_filing_chunks.jsonl
outputs/data_summary/
  amzn_filing_inventory.csv
  amzn_cleaning_summary.csv
  amzn_dataset_summary.json
```

HTML is preferred as the raw format because it preserves more of the source
document structure. If a filing has no usable HTML, the downloader stores the
plain text returned by `edgartools`. The library's internal response cache is
kept below ignored `data/raw/.edgar_cache/` by default rather than in a user's
home directory.

`data/raw/` and `data/processed/` are ignored by Git. SEC documents can be
large, are reproducible from their accession numbers, and may change the size
of the repository substantially. The lightweight summary files contain
metadata and counts rather than filing text, so they may be retained for
inspection or a course presentation.

## Metadata

### Filing manifest

`filing_metadata.json` contains one object per selected filing:

| Field | Meaning |
| --- | --- |
| `company` | SEC filer name |
| `ticker` | Normalized ticker |
| `cik` | Zero-padded SEC Central Index Key when available |
| `filing_type` | Exact form type: `10-K`, `10-Q`, or `8-K` |
| `filing_date` | SEC filing date |
| `accession_number` | Unique SEC accession number |
| `source_url` | SEC URL for the primary filing document |
| `local_raw_path` | Project-relative raw file path |
| `raw_format` | `html` or fallback `text` |

The cleaning manifest carries these fields forward and adds the cleaned file
path, raw and cleaned character lengths, and a list of warnings.

### Chunk dataset

Both chunk formats contain the same rows:

| Field | Meaning |
| --- | --- |
| `company`, `ticker`, `cik` | Filing entity identifiers |
| `filing_type`, `filing_date` | Filing classification and date |
| `accession_number` | Filing-level stable identifier |
| `source_url` | Original SEC filing URL |
| `source_file` | Project-relative cleaned source file |
| `chunk_id` | Deterministic ticker/accession/index identifier |
| `chunk_index` | Zero-based index within the filing |
| `section_heading` | Most recently detected heading, or `unknown` |
| `text` | Chunk text |
| `word_count` | Whitespace-delimited word count |

Chunks do not cross detected section boundaries. Overlap applies only between
adjacent chunks in the same section.

## Regeneration

To regenerate from current SEC EDGAR data:

1. Install the requirements.
2. Configure `.env` with a valid SEC identity.
3. Run the complete build command.

The downloader selects filings dynamically, so dates and accession numbers may
change when Amazon files a newer report. Existing generated files are
overwritten when their names match; current manifests determine which files are
used by later stages.

## Current Limitations

- HTML structures vary across filing years and companies.
- Heading detection uses HTML heading tags and conservative SEC `Item`
  patterns; some headings will remain `unknown`.
- Tables are flattened into readable text and may remain noisy.
- The cleaner intentionally retains uncertain or repetitive disclosure text
  rather than risk deleting financial information.
- The dataset covers only the latest selected filings, not a historical panel.
- Recent 8-K filings may describe unrelated events and may rely on exhibits
  that are not downloaded separately in this milestone.
