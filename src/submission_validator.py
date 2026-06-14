import csv
import os
import re
from typing import Any, Dict, List, Tuple


class SubmissionValidator:
    """
    Validates submission CSV format and constraints according to challenge rules.
    """

    REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
    CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
    EXPECTED_ROWS = 100

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.errors: List[str] = []

    def validate(self) -> bool:
        """
        Runs all validation checks on the CSV file.
        Returns True if valid, False otherwise.
        """
        self.errors.clear()

        # Check file extension
        _, ext = os.path.splitext(self.csv_path)
        if ext.lower() != ".csv":
            self.errors.append("Filename must use a .csv extension.")
            return False

        if not os.path.exists(self.csv_path):
            self.errors.append(f"File not found at: {self.csv_path}")
            return False

        # Try to parse the CSV
        data_rows: List[List[str]] = []
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    self.errors.append("Row 1 must be the header row; file is empty.")
                    return False

                if header != self.REQUIRED_HEADER:
                    self.errors.append(
                        f"Row 1 (header) must be exactly: {','.join(self.REQUIRED_HEADER)}. "
                        f"Found: {','.join(header)}"
                    )

                for row in reader:
                    # Skip empty rows if any
                    if any(cell.strip() for cell in row):
                        data_rows.append(row)
        except UnicodeDecodeError:
            self.errors.append("File must be UTF-8 encoded.")
            return False
        except Exception as e:
            self.errors.append(f"Error reading file: {e}")
            return False

        # Verify row count
        if len(data_rows) != self.EXPECTED_ROWS:
            self.errors.append(
                f"There must be exactly {self.EXPECTED_ROWS} data rows (excluding header). "
                f"Found {len(data_rows)} rows."
            )

        seen_ids = set()
        seen_ranks = set()
        rank_score_pairs: List[Tuple[int, float, str]] = []

        # Validate rows individually
        for idx, cells in enumerate(data_rows, start=2):
            if len(cells) != len(self.REQUIRED_HEADER):
                self.errors.append(
                    f"Row {idx}: expected {len(self.REQUIRED_HEADER)} columns, got {len(cells)}."
                )
                continue

            row = dict(zip(self.REQUIRED_HEADER, cells))
            cid = row["candidate_id"].strip()
            rank_s = row["rank"].strip()
            score_s = row["score"].strip()

            # Verify candidate_id format
            if not cid:
                self.errors.append(f"Row {idx}: candidate_id is required.")
            elif not self.CANDIDATE_ID_PATTERN.match(cid):
                self.errors.append(f"Row {idx}: candidate_id '{cid}' is malformed.")
            elif cid in seen_ids:
                self.errors.append(f"Row {idx}: duplicate candidate_id '{cid}'.")
            else:
                seen_ids.add(cid)

            # Verify rank
            try:
                rank = int(rank_s)
                if str(rank) != rank_s:
                    raise ValueError
                if not 1 <= rank <= 100:
                    self.errors.append(f"Row {idx}: rank '{rank}' must be between 1 and 100.")
                elif rank in seen_ranks:
                    self.errors.append(f"Row {idx}: duplicate rank {rank}.")
                else:
                    seen_ranks.add(rank)
            except ValueError:
                self.errors.append(f"Row {idx}: rank '{rank_s}' must be an integer (1-100).")
                rank = None

            # Verify score
            try:
                score = float(score_s)
            except ValueError:
                self.errors.append(f"Row {idx}: score '{score_s}' must be a float.")
                score = None

            if rank is not None and score is not None:
                rank_score_pairs.append((rank, score, cid))

        # Check for missing ranks
        missing_ranks = set(range(1, 101)) - seen_ranks
        if missing_ranks:
            self.errors.append(f"Missing ranks: {sorted(list(missing_ranks))}")

        # Check score ordering (non-increasing by rank)
        rank_score_pairs.sort(key=lambda x: x[0])
        for i in range(len(rank_score_pairs) - 1):
            r1, s1, c1 = rank_score_pairs[i]
            r2, s2, c2 = rank_score_pairs[i + 1]
            if s1 < s2:
                self.errors.append(
                    f"Score must be non-increasing by rank: rank {r1} ({s1}) < rank {r2} ({s2})."
                )
            elif s1 == s2 and c1 > c2:
                self.errors.append(
                    f"Equal scores at ranks {r1} and {r2}: tie-break requires candidate_id ascending "
                    f"('{c1}' > '{c2}')."
                )

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        return self.errors
