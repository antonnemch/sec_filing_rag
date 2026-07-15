param(
    [int]$Num8K = 1,
    [int]$ChunkSize = 400,
    [int]$ChunkOverlap = 75
)

$ErrorActionPreference = "Stop"

python -m src.data.build_dataset `
    --tickers META AMZN AAPL NFLX GOOG `
    --num-8k $Num8K `
    --chunk-size $ChunkSize `
    --chunk-overlap $ChunkOverlap
