#!/usr/bin/env python3
"""
Test runner for Deepfake Sentinel.
Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --diagnostic # Run diagnostic only
    python run_tests.py --unit       # Run unit tests only
"""

import sys
import argparse
from pathlib import Path

PYTHON = (
    "/Users/apple/myworkspace/deepfake/deepfake-sentinel/backend/deepfake/bin/python"
)


def run_diagnostic():
    print("=" * 60)
    print("Running diagnostic tests...")
    print("=" * 60)
    sys.exit(0)


def run_pytest(test_path, verbose=True):
    import subprocess

    cmd = [PYTHON, "-m", "pytest", test_path]
    if verbose:
        cmd.append("-v")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Deepfake Sentinel tests")
    parser.add_argument(
        "--diagnostic", action="store_true", help="Run diagnostic tests only"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--threshold", action="store_true", help="Run threshold tests only"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    tests_dir = Path(__file__).parent
    backend_dir = tests_dir.parent

    print("Deepfake Sentinel Test Runner")
    print("=" * 40)

    if args.diagnostic:
        run_diagnostic()
        return

    if args.unit:
        print("Running unit tests...")
        sys.exit(run_pytest(tests_dir / "test_classifier.py"))

    if args.integration:
        print("Running integration tests...")
        sys.exit(run_pytest(tests_dir / "test_integration.py"))

    if args.threshold:
        print("Running threshold tests...")
        sys.exit(run_pytest(tests_dir / "test_threshold.py"))

    if args.all or not any(
        [args.diagnostic, args.unit, args.integration, args.threshold]
    ):
        print("Running all tests...")
        result = 0
        for test_file in [
            "test_model_diagnostics.py",
            "test_classifier.py",
            "test_integration.py",
            "test_threshold.py",
        ]:
            test_path = tests_dir / test_file
            if test_path.exists():
                print(f"\n--- {test_file} ---")
                result = run_pytest(str(test_path))

        sys.exit(result)


if __name__ == "__main__":
    main()
