#!/usr/bin/env python
"""
Test Runner - Run all tests
============================

Simple test runner that executes all test suites.

Usage:
    python tests/run_tests.py           # Run all tests
    python tests/run_tests.py unit      # Run only unit tests
    python tests/run_tests.py integration  # Run only integration tests
    python tests/run_tests.py e2e       # Run only e2e tests
    python tests/run_tests.py quick     # Run quick tests (unit only)

Or use pytest:
    pytest tests/ -v                    # All tests
    pytest tests/test_unit.py -v        # Unit tests only
    pytest tests/test_integration.py -v # Integration tests only
    pytest tests/test_e2e.py -v         # E2E tests only
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unit_tests():
    """Run unit tests"""
    from test_unit import run_all_unit_tests
    return run_all_unit_tests()


def run_integration_tests():
    """Run integration tests"""
    from test_integration import run_all_integration_tests
    return run_all_integration_tests()


def run_e2e_tests():
    """Run end-to-end tests"""
    from test_e2e import run_all_e2e_tests
    return run_all_e2e_tests()


def run_all_tests():
    """Run all test suites"""
    print("=" * 70)
    print("FULL TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # Unit tests
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    results["unit"] = run_unit_tests()

    # Integration tests
    print("\n" + "=" * 70)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 70)
    results["integration"] = run_integration_tests()

    # E2E tests
    print("\n" + "=" * 70)
    print("RUNNING E2E TESTS")
    print("=" * 70)
    results["e2e"] = run_e2e_tests()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    all_passed = True
    for suite, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {suite.upper()}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n SUCCESS: All test suites passed!")
    else:
        print("\n FAILURE: Some tests failed!")

    return all_passed


def main():
    """Main entry point"""
    args = sys.argv[1:]

    if not args or args[0] == "all":
        success = run_all_tests()
    elif args[0] == "unit" or args[0] == "quick":
        success = run_unit_tests()
    elif args[0] == "integration":
        success = run_integration_tests()
    elif args[0] == "e2e":
        success = run_e2e_tests()
    elif args[0] in ["-h", "--help", "help"]:
        print(__doc__)
        return 0
    else:
        print(f"Unknown test suite: {args[0]}")
        print("Use: all, unit, integration, e2e, quick")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
