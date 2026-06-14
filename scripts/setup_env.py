import os
import sys
import platform
import importlib.metadata
from typing import List

REQUIRED_PACKAGES = [
    "sentence-transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "tqdm",
    "pytest",
    "black",
    "ruff",
    "jupyter",
]

def verify_python_version() -> bool:
    """
    Verifies that the Python version is at least 3.11.
    """
    major, minor, micro = platform.python_version_tuple()
    print(f"Python Version: {sys.version}")
    
    # We warn if not exactly 3.11 as specified, but accept any version >= 3.11
    if int(major) != 3 or int(minor) < 11:
        print(f"[-] WARNING: Project requires Python 3.11+. Current version is {major}.{minor}.{micro}.", file=sys.stderr)
        return False
    
    if int(minor) != 11:
        print(f"[!] INFO: Current Python version is {major}.{minor}.{micro}. (Target: Python 3.11)")
    else:
        print("[+] Python version matches the 3.11 target.")
    return True

def get_virtual_env_status() -> str:
    """
    Checks if a virtual environment is active.
    """
    if hasattr(sys, 'real_prefix') or (sys.base_prefix != sys.prefix):
        return f"Active (Path: {sys.prefix})"
    return "Not Active"

def verify_installed_packages(packages: List[str]) -> bool:
    """
    Verifies that all required packages are installed and prints their versions.
    """
    print("\n--- Package Verification Status ---")
    all_ok = True
    for package in packages:
        # Map import package name to distribution name if needed
        dist_name = package
        try:
            version = importlib.metadata.version(dist_name)
            print(f"[+] {package}: Installed (Version: {version})")
        except importlib.metadata.PackageNotFoundError:
            print(f"[-] {package}: NOT INSTALLED")
            all_ok = False
    return all_ok

def print_environment_status() -> None:
    """
    Prints system and project environment status details.
    """
    print("\n=== Environment Status ===")
    print(f"OS Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Executable: {sys.executable}")
    print(f"Virtual Environment: {get_virtual_env_status()}")
    print("==========================\n")

def main() -> None:
    print("Running setup environment validation script...\n")
    python_ok = verify_python_version()
    print_environment_status()
    packages_ok = verify_installed_packages(REQUIRED_PACKAGES)
    
    if python_ok and packages_ok:
        print("\n[+] SUCCESS: Development environment is fully configured and ready.")
        sys.exit(0)
    else:
        print("\n[-] FAILURE: Environment configuration issues detected. Please check package installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
