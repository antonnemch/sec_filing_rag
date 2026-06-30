# SEC Filing RAG

BU493 Machine Learning in Finance group project for financial question
answering over SEC filings.

The current milestone implements only the reproducible data pipeline:

1. download recent SEC filings with `edgartools`;
2. clean filing HTML conservatively; and
3. create metadata-rich, word-based chunks.

Amazon (`AMZN`) is the default company. Keyword search, embeddings, vector
databases, LLM calls, and RAG are intentionally out of scope for this stage.

See [DATAREADME.md](DATAREADME.md) for setup, data schemas, generated files,
limitations, and regeneration instructions.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with your SEC identity, then run:
python -m src.data.build_dataset --ticker AMZN --num-8k 5 --chunk-size 400 --chunk-overlap 75
```
