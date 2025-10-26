#!/usr/bin/env python
"""
Main test runner for data processing module tests.

This script runs all tests for the data processing pipeline including:
- Data fetching and parsing tests
- Feature engineering tests
- Feature selection tests
- Integration and validation tests

Usage:
    # Run all tests
    python main_test.py

    # Run with verbose output
    python main_test.py -v

    # Run specific test file
    python main_test.py -f test_data_parser.py

    # Run with coverage report
    python main_test.py --coverage

    # Run fast tests only (exclude slow integration tests)
    python main_test.py --fast

    # Generate HTML report
    python main_test.py --html
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


class DataTestRunner:
    """Test runner for data processing module."""
    
    def __init__(self, test_dir: Optional[Path] = None):
        """
        Initialize test runner.
        
        Args:
            test_dir: Directory containing tests (default: current file's directory)
        """
        if test_dir is None:
            test_dir = Path(__file__).parent
        self.test_dir = test_dir
        self.project_root = test_dir.parent.parent.parent.parent
        
    def run_tests(
        self,
        verbose: bool = False,
        coverage: bool = False,
        html_report: bool = False,
        test_file: Optional[str] = None,
        fast: bool = False,
        markers: Optional[str] = None,
        extra_args: Optional[List[str]] = None
    ) -> int:
        """
        Run tests with specified options.
        
        Args:
            verbose: Enable verbose output
            coverage: Generate coverage report
            html_report: Generate HTML test report
            test_file: Run specific test file only
            fast: Run only fast tests (skip slow integration tests)
            markers: Pytest markers to filter tests
            extra_args: Additional pytest arguments
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Build pytest command
        cmd = ['pytest']
        
        # Add test directory or specific file
        if test_file:
            test_path = self.test_dir / test_file
            if not test_path.exists():
                print(f"Error: Test file '{test_file}' not found in {self.test_dir}")
                return 1
            cmd.append(str(test_path))
        else:
            cmd.append(str(self.test_dir))
        
        # Verbose output
        if verbose:
            cmd.append('-v')
        else:
            cmd.append('-q')
        
        # Show test summary
        cmd.append('-ra')
        
        # Show local variables on failure
        if verbose:
            cmd.append('--showlocals')
        
        # Coverage options
        if coverage:
            data_module = self.test_dir.parent
            cmd.extend([
                '--cov=' + str(data_module),
                '--cov-report=term-missing',
                '--cov-report=html:htmlcov',
            ])
        
        # HTML report
        if html_report:
            cmd.extend([
                '--html=test_report.html',
                '--self-contained-html'
            ])
        
        # Fast tests only
        if fast:
            cmd.extend(['-m', 'not slow'])
        
        # Custom markers
        if markers:
            cmd.extend(['-m', markers])
        
        # Fail fast on first error (optional)
        # cmd.append('-x')
        
        # Show warnings
        cmd.append('-W')
        cmd.append('default')
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        # Print command
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        
        # Run tests
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))
            return result.returncode
        except FileNotFoundError:
            print("Error: pytest not found. Install it with: pip install pytest pytest-cov pytest-html")
            return 1
        except KeyboardInterrupt:
            print("\nTests interrupted by user")
            return 130


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description='Run data processing module tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all tests
  %(prog)s -v                        # Verbose output
  %(prog)s -f test_data_parser.py    # Run specific test file
  %(prog)s --coverage                # Generate coverage report
  %(prog)s --fast                    # Run fast tests only
  %(prog)s -m "not slow"             # Custom pytest markers
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose test output'
    )
    
    parser.add_argument(
        '-f', '--file',
        dest='test_file',
        help='Run specific test file (e.g., test_data_parser.py)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate code coverage report'
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        dest='html_report',
        help='Generate HTML test report'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run only fast tests (skip slow integration tests)'
    )
    
    parser.add_argument(
        '-m', '--markers',
        help='Run tests matching given mark expression (e.g., "not slow")'
    )
    
    parser.add_argument(
        '--list-tests',
        action='store_true',
        help='List all available tests without running them'
    )
    
    parser.add_argument(
        'pytest_args',
        nargs='*',
        help='Additional arguments to pass to pytest'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = DataTestRunner()
    
    # List tests if requested
    if args.list_tests:
        cmd = ['pytest', '--collect-only', '-q', str(runner.test_dir)]
        subprocess.run(cmd, cwd=str(runner.project_root))
        return 0
    
    # Run tests
    exit_code = runner.run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        html_report=args.html_report,
        test_file=args.test_file,
        fast=args.fast,
        markers=args.markers,
        extra_args=args.pytest_args
    )
    
    # Print summary
    print("-" * 80)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests failed with exit code {exit_code}")
    
    if args.coverage:
        print(f"\nCoverage report saved to: {runner.project_root}/htmlcov/index.html")
    
    if args.html_report:
        print(f"HTML test report saved to: {runner.project_root}/test_report.html")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

