# research/

Exploratory Data Analysis (EDA) scripts and investigation utilities.

These scripts were used during development to understand the dataset
and make design decisions. They are **not** part of the production pipeline
but are preserved here for reproducibility and future reference.

---

## Scripts

| Script | Purpose |
|---|---|
| `explore_schema.py` | Recursively profiles the `candidates.jsonl.gz` schema — field types, nesting depth, value distributions, examples. Outputs `outputs/schema_report.md`. |
| `dataset_memory_report.py` | Estimates uncompressed dataset size by sampling, calculates per-field memory footprint, and produces a pandas `DataFrame` memory report. Useful for capacity planning. |

---

## Running

```bash
# Explore the dataset schema
python research/explore_schema.py

# Get memory footprint report
python research/dataset_memory_report.py
```

Both scripts expect `data/raw/candidates.jsonl.gz` to be present.
