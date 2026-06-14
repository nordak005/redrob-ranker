import gzip
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple


def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Loads candidate_schema.json.
    """
    if not os.path.exists(schema_path):
        print(f"[-] ERROR: Schema file not found at {schema_path}", file=sys.stderr)
        sys.exit(1)
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_fields(
    data: Any, schema_node: Dict[str, Any], path: str = ""
) -> List[str]:
    """
    Recursively validates presence of required fields defined in schema_node.
    Returns a list of error messages for missing fields.
    """
    errors = []
    if not isinstance(schema_node, dict):
        return errors

    node_type = schema_node.get("type")

    # If it is an object, check its required properties
    if node_type == "object" and isinstance(data, dict):
        required_fields = schema_node.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"{path}.{field}" if path else field)
            else:
                # Recursively check sub-properties
                properties = schema_node.get("properties", {})
                if field in properties:
                    errors.extend(
                        check_required_fields(
                            data[field],
                            properties[field],
                            f"{path}.{field}" if path else field,
                        )
                    )

    # If it is an array, check each item against the items schema
    elif node_type == "array" and isinstance(data, list):
        items_schema = schema_node.get("items")
        if items_schema:
            for idx, item in enumerate(data):
                errors.extend(
                    check_required_fields(
                        item, items_schema, f"{path}[{idx}]"
                    )
                )

    return errors


def validate_candidates(
    data_path: str, schema: Dict[str, Any]
) -> Tuple[int, int, Dict[str, int], List[Dict[str, Any]]]:
    """
    Validates candidates.jsonl.gz against schema.
    Returns: (total_count, malformed_count, missing_field_counts, sample_records)
    """
    total_count = 0
    malformed_count = 0
    missing_field_counts: Dict[str, int] = {}
    sample_records: List[Dict[str, Any]] = []

    if not os.path.exists(data_path):
        print(f"[-] ERROR: Data file not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Validating candidate pool from {data_path}...")
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_count += 1

            # Parse JSON
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                malformed_count += 1
                if malformed_count <= 5:
                    print(
                        f"[-] Malformed JSON on line {line_num}: {e}",
                        file=sys.stderr,
                    )
                continue

            # Keep first 5 records as samples
            if len(sample_records) < 5:
                sample_records.append(candidate)

            # Validate against schema (missing fields)
            missing = check_required_fields(candidate, schema)
            for m_field in missing:
                # Standardize path by replacing array indexes (e.g. career_history[0].title -> career_history.title)
                clean_path = ""
                in_bracket = False
                for char in m_field:
                    if char == "[":
                        in_bracket = True
                    elif char == "]":
                        in_bracket = False
                    elif not in_bracket:
                        clean_path += char
                missing_field_counts[clean_path] = (
                    missing_field_counts.get(clean_path, 0) + 1
                )

    return (
        total_count,
        malformed_count,
        missing_field_counts,
        sample_records,
    )


def main() -> None:
    schema_file = "data/raw/candidate_schema.json"
    data_file = "data/raw/candidates.jsonl.gz"

    schema = load_schema(schema_file)

    # File size report
    file_size_bytes = os.path.getsize(data_file)
    file_size_mb = file_size_bytes / (1024 * 1024)

    (
        total_count,
        malformed_count,
        missing_fields,
        samples,
    ) = validate_candidates(data_file, schema)

    print("\n" + "=" * 40)
    print("        DATA VALIDATION REPORT        ")
    print("=" * 40)
    print(f"File Path:          {data_file}")
    print(f"File Size:          {file_size_mb:.2f} MB")
    print(f"Total Records:      {total_count}")
    print(f"Malformed Records:  {malformed_count}")
    print("-" * 40)

    if missing_fields:
        print("Missing Required Fields (summary):")
        for field, count in missing_fields.items():
            pct = (count / total_count) * 100
            print(f"  - {field}: missing in {count} records ({pct:.2f}%)")
    else:
        print("[+] All records conform to the schema's required fields.")

    print("-" * 40)
    print("Sample Records (first 2 shown for brevity):")
    for idx, sample in enumerate(samples[:2], start=1):
        print(f"\n--- Sample Candidate #{idx} (ID: {sample.get('candidate_id')}) ---")
        print(json.dumps(sample, indent=2)[:800] + "\n... [truncated]")

    print("=" * 40)

    if malformed_count > 0:
        print("\n[-] Validation failed: Malformed JSON detected.")
        sys.exit(1)
    else:
        print("\n[+] Validation succeeded: All candidate records are valid JSON.")
        sys.exit(0)


if __name__ == "__main__":
    main()
