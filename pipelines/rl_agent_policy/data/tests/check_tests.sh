#!/bin/bash
# Quick test checker script

echo "==================================="
echo "Data Processing Tests - Quick Check"
echo "==================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main_test.py" ]; then
    echo "Error: Please run this script from the tests directory"
    exit 1
fi

# Count test files
echo "📁 Test files:"
ls -1 test_*.py | wc -l | xargs echo "   Found test files:"
echo ""

# List test files with line counts
echo "📊 Test file statistics:"
for file in test_*.py; do
    lines=$(wc -l < "$file")
    printf "   %-30s %5d lines\n" "$file" "$lines"
done
echo ""

# Count total lines
total_lines=$(cat test_*.py conftest.py | wc -l)
echo "   Total test code: $total_lines lines"
echo ""

# Check for pytest
echo "🔍 Checking dependencies:"
if command -v pytest &> /dev/null; then
    pytest_version=$(pytest --version | head -n1)
    echo "   ✓ $pytest_version"
else
    echo "   ✗ pytest not found (install with: pip install pytest)"
fi

if python -c "import pandas" 2>/dev/null; then
    echo "   ✓ pandas installed"
else
    echo "   ✗ pandas not found"
fi

if python -c "import numpy" 2>/dev/null; then
    echo "   ✓ numpy installed"
else
    echo "   ✗ numpy not found"
fi

if python -c "import sklearn" 2>/dev/null; then
    echo "   ✓ scikit-learn installed"
else
    echo "   ✗ scikit-learn not found"
fi
echo ""

# Try to collect tests (without running)
echo "📝 Collecting tests (without running):"
cd ../../../.. && python -m pytest pipelines/rl_agent_policy/data/tests -v
echo ""

echo "==================================="
echo "Quick check complete!"
echo ""
echo "To run tests:"
echo "  python main_test.py"
echo ""
echo "To run with coverage:"
echo "  python main_test.py --coverage"
echo ""
echo "To see all options:"
echo "  python main_test.py --help"
echo "==================================="

