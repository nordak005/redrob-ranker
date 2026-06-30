# configs/

Project configuration files.

---

## Files

| File | Purpose |
|---|---|
| `submission_metadata.yaml` | Stage 3 competition submission metadata — team info, reproduce command, compute environment, methodology summary, and honesty declarations. Fill in `TODO` fields before submitting. |

---

## Usage

The `submission_metadata.yaml` must be submitted alongside your code as part of the
competition portal upload. Verify all `TODO` placeholders are filled before the
submission deadline.

```bash
# Validate the submission CSV
python scripts/validate_submission.py outputs/final_submission.csv
```
