import os
import tempfile
import pytest
from src.utils import load_candidates, load_json, save_csv
from src.submission_validator import SubmissionValidator


def test_load_json_and_candidates():
    # Test loading sample_candidates
    sample_path = "data/sample/sample_candidates.json"
    if os.path.exists(sample_path):
        data = load_json(sample_path)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "candidate_id" in data[0]


def test_submission_validator_valid():
    # Test that the sample submission conforms to validator specifications
    sample_sub = "data/sample/sample_submission.csv"
    if os.path.exists(sample_sub):
        validator = SubmissionValidator(sample_sub)
        is_valid = validator.validate()
        assert is_valid, f"Sample submission validation failed: {validator.get_errors()}"


def test_submission_validator_invalid_format():
    # Create an invalid CSV to verify error detection
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_csv = os.path.join(tmpdir, "invalid.csv")
        
        # Test 1: Empty CSV
        with open(invalid_csv, "w", encoding="utf-8") as f:
            pass
        validator = SubmissionValidator(invalid_csv)
        assert not validator.validate()
        assert any("empty" in e.lower() for e in validator.get_errors())

        # Test 2: Missing header columns
        with open(invalid_csv, "w", encoding="utf-8", newline="") as f:
            f.write("candidate_id,rank,score\n")  # Missing reasoning
        validator = SubmissionValidator(invalid_csv)
        assert not validator.validate()
        assert any("header" in e.lower() or "columns" in e.lower() for e in validator.get_errors())


def test_submission_validator_duplicate_ranks():
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_csv = os.path.join(tmpdir, "dupe_ranks.csv")
        
        # Write rows with duplicate ranks (e.g. two rank 1s)
        rows = [["candidate_id", "rank", "score", "reasoning"]]
        for i in range(1, 101):
            rank = 1 if i == 2 else i  # Duplicate rank 1
            rows.append([f"CAND_{i:07d}", str(rank), str(100 - i), "Reason"])
            
        with open(invalid_csv, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(rows)
            
        validator = SubmissionValidator(invalid_csv)
        assert not validator.validate()
        assert any("duplicate rank" in e.lower() or "missing" in e.lower() for e in validator.get_errors())


def test_submission_validator_score_ordering():
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_csv = os.path.join(tmpdir, "bad_order.csv")
        
        # Write rows where score increases
        rows = [["candidate_id", "rank", "score", "reasoning"]]
        for i in range(1, 101):
            # Make score at rank 2 higher than rank 1
            score = 100.0 if i == 2 else (100.0 - i)
            rows.append([f"CAND_{i:07d}", str(i), f"{score:.2f}", "Reason"])
            
        with open(invalid_csv, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(rows)
            
        validator = SubmissionValidator(invalid_csv)
        assert not validator.validate()
        assert any("non-increasing" in e.lower() for e in validator.get_errors())


def test_submission_validator_tie_break():
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_csv = os.path.join(tmpdir, "bad_tie.csv")
        
        # Write rows with tied scores but descending candidate IDs
        rows = [
            ["candidate_id", "rank", "score", "reasoning"],
            ["CAND_0000002", "1", "1.0", "Reason"],
            ["CAND_0000001", "2", "1.0", "Reason"],  # Ties on score, but CAND_0000002 > CAND_0000001 (violates ascending ID tie-break!)
        ]
        # Fill rest of 100 rows
        for i in range(3, 101):
            rows.append([f"CAND_{i:07d}", str(i), f"{1.0 - i/100:.2f}", "Reason"])
            
        with open(invalid_csv, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(rows)
            
        validator = SubmissionValidator(invalid_csv)
        assert not validator.validate()
        assert any("tie-break" in e.lower() for e in validator.get_errors())
