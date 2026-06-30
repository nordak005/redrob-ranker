import gzip
import json
import os
import sys
import pandas as pd
from typing import List, Dict, Any, Tuple


def estimate_uncompressed_size(filepath: str, sample_lines: int = 1000) -> Tuple[float, int]:
    """
    Estimates the total uncompressed size of candidates.jsonl.gz by sampling.
    Returns: (estimated_uncompressed_mb, total_lines)
    """
    total_uncompressed_bytes = 0
    sampled_lines = 0
    
    # First count lines and sample some bytes
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if sampled_lines < sample_lines:
                    total_uncompressed_bytes += len(line.encode("utf-8"))
                    sampled_lines += 1
    
    # We already know total records is 100,000
    total_lines = 100000
    avg_line_size = total_uncompressed_bytes / sampled_lines if sampled_lines > 0 else 0
    estimated_total_bytes = avg_line_size * total_lines
    estimated_total_mb = estimated_total_bytes / (1024 * 1024)
    
    return estimated_total_mb, total_lines


def profile_pandas_memory(filepath: str, sample_size: int = 1000) -> Tuple[float, float]:
    """
    Estimates pandas dataframe memory usage for 100k candidates by loading a sample.
    Returns: (estimated_shallow_mb, estimated_deep_mb)
    """
    records = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= sample_size:
                    break
                    
    df = pd.DataFrame(records)
    
    # Memory usage in bytes
    shallow_bytes = df.memory_usage(index=True, deep=False).sum()
    deep_bytes = df.memory_usage(index=True, deep=True).sum()
    
    # Scale up to 100k candidates
    scale_factor = 100000 / sample_size
    estimated_shallow_mb = (shallow_bytes * scale_factor) / (1024 * 1024)
    estimated_deep_mb = (deep_bytes * scale_factor) / (1024 * 1024)
    
    return estimated_shallow_mb, estimated_deep_mb


def calculate_embedding_memory(num_candidates: int, dimensions: List[int]) -> Dict[int, float]:
    """
    Calculates memory required to hold float32 embeddings in RAM.
    """
    memory_dict = {}
    for dim in dimensions:
        # float32 = 4 bytes
        bytes_needed = num_candidates * dim * 4
        mb_needed = bytes_needed / (1024 * 1024)
        memory_dict[dim] = mb_needed
    return memory_dict


def main() -> None:
    data_file = "data/raw/candidates.jsonl.gz"
    if not os.path.exists(data_file):
        print(f"[-] ERROR: Data file not found at {data_file}", file=sys.stderr)
        sys.exit(1)

    import json  # Import locally for JSON loading
    
    # 1. File Size Profile
    compressed_size_bytes = os.path.getsize(data_file)
    compressed_size_mb = compressed_size_bytes / (1024 * 1024)
    
    print("[*] Sampling candidate pool for memory profiling...")
    est_uncompressed_mb, total_records = estimate_uncompressed_size(data_file, 1000)
    
    # 2. Pandas DataFrame Profile
    est_shallow_mb, est_deep_mb = profile_pandas_memory(data_file, 1000)
    
    # 3. Embedding Memory Profile
    dims = [384, 768, 1024, 1536]
    emb_memory = calculate_embedding_memory(total_records, dims)
    
    # System memory constraints
    limit_gb = 16.0
    limit_mb = limit_gb * 1024
    
    print("\n" + "=" * 45)
    print("      DATASET MEMORY ESTIMATION REPORT      ")
    print("=" * 45)
    print(f"Total Candidates:             {total_records:,}")
    print(f"Compressed File Size (.gz):   {compressed_size_mb:.2f} MB")
    print(f"Est. Uncompressed JSONL Size: {est_uncompressed_mb:.2f} MB")
    print("-" * 45)
    print("Estimated Pandas DataFrame Memory (100k):")
    print(f"  - Shallow Reference memory: {est_shallow_mb:.2f} MB")
    print(f"  - Deep (Object values) memory: {est_deep_mb:.2f} MB")
    print("-" * 45)
    print("Float32 Embedding Matrix Memory (100k):")
    for dim, mb in emb_memory.items():
        pct = (mb / limit_mb) * 100
        print(f"  - {dim} dims: {mb:.2f} MB ({pct:.2f}% of {limit_gb} GB RAM limit)")
    print("-" * 45)
    
    # RAM consumption warning checks
    expected_usage_mb = est_deep_mb + emb_memory[768] + 500  # DataFrame + Embedding + Python overhead
    expected_usage_gb = expected_usage_mb / 1024
    print(f"Expected Memory Footprint:    {expected_usage_gb:.2f} GB")
    print(f"RAM limit:                    {limit_gb:.2f} GB")
    
    if expected_usage_gb > limit_gb:
        print("[!] WARNING: Expected memory footprint exceeds the 16 GB limit!")
    else:
        print("[+] PASS: Estimated memory footprint fits comfortably within 16 GB RAM.")
    print("=" * 45)


if __name__ == "__main__":
    from typing import Tuple  # Ensure Tuple import is active
    main()
