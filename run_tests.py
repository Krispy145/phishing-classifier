#!/usr/bin/env python3
"""
Comprehensive Test Runner

This script runs all tests with detailed reporting and generates structured
output in multiple formats for analysis and CI/CD integration.

Usage:
    python run_tests.py [--format FORMAT] [--category CATEGORY] [--verbose]
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import shutil


def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def create_test_directories():
    """Create test result directories."""
    dirs = [
        "test_results/html",
        "test_results/json", 
        "test_results/xml",
        "test_results/coverage/html",
        "test_results/coverage",
        "test_results/logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def run_all_tests(verbose=False):
    """Run all tests with comprehensive reporting."""
    print("🧪 Running comprehensive test suite...")
    print("=" * 60)
    
    # Base pytest command
    cmd = "python3 -m pytest"
    
    if verbose:
        cmd += " -v"
    
    # Run tests
    print(f"Command: {cmd}")
    returncode, stdout, stderr = run_command(cmd, capture_output=False)
    
    return returncode, stdout, stderr


def run_category_tests(category, verbose=False):
    """Run tests for a specific category."""
    print(f"🧪 Running {category} tests...")
    print("=" * 60)
    
    cmd = f"python3 -m pytest tests/test_features.py::Test{category}FeatureExtractor"
    
    if verbose:
        cmd += " -v"
    
    returncode, stdout, stderr = run_command(cmd, capture_output=False)
    
    return returncode, stdout, stderr


def run_unit_tests(verbose=False):
    """Run only unit tests."""
    print("🧪 Running unit tests...")
    print("=" * 60)
    
    cmd = "python3 -m pytest -m unit"
    
    if verbose:
        cmd += " -v"
    
    returncode, stdout, stderr = run_command(cmd, capture_output=False)
    
    return returncode, stdout, stderr


def run_integration_tests(verbose=False):
    """Run only integration tests."""
    print("🧪 Running integration tests...")
    print("=" * 60)
    
    cmd = "python3 -m pytest -m integration"
    
    if verbose:
        cmd += " -v"
    
    returncode, stdout, stderr = run_command(cmd, capture_output=False)
    
    return returncode, stdout, stderr


def generate_test_summary():
    """Generate a comprehensive test summary."""
    print("\n📊 Generating test summary...")
    
    # Read JSON report
    json_path = Path("test_results/json/report.json")
    if json_path.exists():
        with open(json_path, 'r') as f:
            report = json.load(f)
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "summary": report.get("summary", {}),
            "test_results": report.get("tests", []),
            "coverage": {}
        }
        
        # Read coverage report
        coverage_path = Path("test_results/coverage/coverage.json")
        if coverage_path.exists():
            with open(coverage_path, 'r') as f:
                coverage = json.load(f)
            summary["coverage"] = coverage.get("totals", {})
        
        # Write summary
        summary_path = Path("test_results/test_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Test summary saved to: {summary_path}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("📈 TEST SUMMARY")
        print("=" * 60)
        
        if "summary" in summary:
            s = summary["summary"]
            print(f"Total tests: {s.get('total', 'N/A')}")
            print(f"Passed: {s.get('passed', 'N/A')}")
            print(f"Failed: {s.get('failed', 'N/A')}")
            print(f"Errors: {s.get('error', 'N/A')}")
            print(f"Duration: {s.get('duration', 'N/A')}s")
        
        if "coverage" in summary and summary["coverage"]:
            c = summary["coverage"]
            print(f"\nCoverage: {c.get('percent_covered', 'N/A')}%")
            print(f"Lines covered: {c.get('covered_lines', 'N/A')}/{c.get('num_statements', 'N/A')}")
        
        print("=" * 60)


def generate_markdown_report():
    """Generate a markdown test report."""
    print("\n📝 Generating markdown report...")
    
    # Read JSON report
    json_path = Path("test_results/json/report.json")
    if not json_path.exists():
        print("❌ No JSON report found")
        return
    
    with open(json_path, 'r') as f:
        report = json.load(f)
    
    # Generate markdown
    md_content = f"""# Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {report.get('summary', {}).get('total', 'N/A')} |
| Passed | {report.get('summary', {}).get('passed', 'N/A')} |
| Failed | {report.get('summary', {}).get('failed', 'N/A')} |
| Errors | {report.get('summary', {}).get('error', 'N/A')} |
| Duration | {report.get('summary', {}).get('duration', 'N/A')}s |

## Test Results

"""
    
    # Add test details
    for test in report.get('tests', []):
        status = "✅ PASS" if test.get('outcome') == 'passed' else "❌ FAIL"
        md_content += f"- {status} `{test.get('nodeid', 'Unknown')}`\n"
        if test.get('outcome') != 'passed':
            md_content += f"  - Error: {test.get('call', {}).get('longrepr', 'Unknown error')}\n"
    
    # Write markdown report
    md_path = Path("test_results/test_report.md")
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"✓ Markdown report saved to: {md_path}")


def cleanup_old_results():
    """Clean up old test results."""
    print("🧹 Cleaning up old test results...")
    
    test_results_dir = Path("test_results")
    if test_results_dir.exists():
        # Keep only the latest results
        for item in test_results_dir.iterdir():
            if item.is_file() and item.suffix in ['.html', '.json', '.xml']:
                # Keep the files, they'll be overwritten
                pass
            elif item.is_dir():
                # Clean up subdirectories
                shutil.rmtree(item, ignore_errors=True)
                item.mkdir(parents=True, exist_ok=True)
    
    print("✓ Cleanup complete")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--format", choices=["all", "unit", "integration", "category"], 
                       default="all", help="Test format to run")
    parser.add_argument("--category", choices=["URL", "Domain", "Content", "Pipeline"], 
                       help="Category for category-specific tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Clean up old results before running")
    
    args = parser.parse_args()
    
    print("🚀 Phishing Classifier - Test Runner")
    print("=" * 60)
    
    # Setup
    if args.cleanup:
        cleanup_old_results()
    
    create_test_directories()
    
    # Run tests based on format
    if args.format == "all":
        returncode, stdout, stderr = run_all_tests(args.verbose)
    elif args.format == "unit":
        returncode, stdout, stderr = run_unit_tests(args.verbose)
    elif args.format == "integration":
        returncode, stdout, stderr = run_integration_tests(args.verbose)
    elif args.format == "category" and args.category:
        returncode, stdout, stderr = run_category_tests(args.category, args.verbose)
    else:
        print("❌ Invalid arguments")
        sys.exit(1)
    
    # Generate reports
    generate_test_summary()
    generate_markdown_report()
    
    # Print results
    print(f"\n📁 Test results saved to:")
    print(f"  - HTML Report: test_results/html/report.html")
    print(f"  - JSON Report: test_results/json/report.json")
    print(f"  - XML Report: test_results/xml/junit.xml")
    print(f"  - Coverage: test_results/coverage/html/index.html")
    print(f"  - Summary: test_results/test_summary.json")
    print(f"  - Markdown: test_results/test_report.md")
    
    # Exit with appropriate code
    if returncode == 0:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Tests failed with return code: {returncode}")
        sys.exit(returncode)


if __name__ == "__main__":
    main()
