#!/bin/bash

# OTEL ML Demo - Test Script
# Run comprehensive test suite with various options

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
VERBOSE=false
COVERAGE=false
COVERAGE_FORMAT="term"
PARALLEL=false
FAIL_FAST=false
OUTPUT_FILE=""
TEST_PATTERN=""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [TEST_TYPE]"
    echo ""
    echo "Run test suite for OTEL ML Demo project"
    echo ""
    echo "Test Types:"
    echo "  all                      Run all tests (default)"
    echo "  unit                     Run unit tests only"
    echo "  integration              Run integration tests only"
    echo "  regression               Run non-regression tests only"
    echo "  data                     Run data loader tests only"
    echo "  model                    Run model tests only"
    echo "  pipeline                 Run pipeline integration tests"
    echo "  fast                     Run quick subset of tests"
    echo ""
    echo "Options:"
    echo "  -v, --verbose            Verbose test output"
    echo "  -c, --coverage           Generate coverage report"
    echo "  -f, --coverage-format F  Coverage format: term, html, xml (default: term)"
    echo "  -p, --parallel           Run tests in parallel (if pytest-xdist installed)"
    echo "  -x, --fail-fast          Stop on first failure"
    echo "  -o, --output FILE        Save test results to file"
    echo "  -k, --pattern PATTERN    Run tests matching pattern"
    echo "  -q, --quiet              Minimal output"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Coverage Formats:"
    echo "  term                     Terminal output"
    echo "  html                     HTML report (creates htmlcov/)"
    echo "  xml                      XML report (coverage.xml)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run all tests"
    echo "  $0 unit -v               # Verbose unit tests"
    echo "  $0 --coverage --coverage-format html  # HTML coverage report"
    echo "  $0 -k \"test_train\" -v    # Run tests matching pattern"
    echo "  $0 fast -x               # Quick tests, stop on first failure"
    echo "  $0 integration -o results.xml  # Save integration test results"
}

# Function to log messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

log_coverage() {
    echo -e "${MAGENTA}[COVERAGE]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    if [ ! -d "venv" ]; then
        log_error "Virtual environment not found."
        log_error "Please run: python3 -m venv venv && source venv/bin/activate && pip install -e .[test]"
        exit 1
    fi

    # Check if pytest is installed
    if ! source venv/bin/activate && python -c "import pytest" &>/dev/null; then
        log_error "pytest not installed."
        log_error "Please run: source venv/bin/activate && pip install -e .[test]"
        exit 1
    fi

    # Check for coverage if requested
    if [ "$COVERAGE" = true ]; then
        if ! source venv/bin/activate && python -c "import pytest_cov" &>/dev/null; then
            log_warning "pytest-cov not installed. Coverage disabled."
            COVERAGE=false
        fi
    fi

    # Check for parallel execution
    if [ "$PARALLEL" = true ]; then
        if ! source venv/bin/activate && python -c "import xdist" &>/dev/null; then
            log_warning "pytest-xdist not installed. Parallel execution disabled."
            PARALLEL=false
        fi
    fi
}

# Function to build pytest command
build_pytest_cmd() {
    local cmd="source venv/bin/activate && python -m pytest"

    # Add test path based on type
    case $TEST_TYPE in
        all)
            cmd="$cmd tests/"
            ;;
        unit)
            cmd="$cmd tests/test_data_loader.py tests/test_model.py"
            ;;
        integration)
            cmd="$cmd tests/test_integration.py"
            ;;
        regression)
            cmd="$cmd tests/test_regression.py"
            ;;
        data)
            cmd="$cmd tests/test_data_loader.py"
            ;;
        model)
            cmd="$cmd tests/test_model.py"
            ;;
        pipeline)
            cmd="$cmd tests/test_integration.py::TestPipelineIntegration"
            ;;
        fast)
            cmd="$cmd tests/test_data_loader.py::TestSDSSDataLoader::test_init tests/test_model.py::TestStellarClassifier::test_init"
            ;;
        *)
            log_error "Unknown test type: $TEST_TYPE"
            exit 1
            ;;
    esac

    # Add verbosity
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    elif [ "$QUIET" = true ]; then
        cmd="$cmd -q"
    fi

    # Add coverage
    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=src --cov-report=$COVERAGE_FORMAT"
        if [ "$COVERAGE_FORMAT" = "html" ]; then
            cmd="$cmd --cov-report=html:htmlcov"
        elif [ "$COVERAGE_FORMAT" = "xml" ]; then
            cmd="$cmd --cov-report=xml:coverage.xml"
        fi
    fi

    # Add parallel execution
    if [ "$PARALLEL" = true ]; then
        cmd="$cmd -n auto"
    fi

    # Add fail fast
    if [ "$FAIL_FAST" = true ]; then
        cmd="$cmd -x"
    fi

    # Add test pattern
    if [ -n "$TEST_PATTERN" ]; then
        cmd="$cmd -k '$TEST_PATTERN'"
    fi

    # Add output file
    if [ -n "$OUTPUT_FILE" ]; then
        cmd="$cmd --junitxml='$OUTPUT_FILE'"
    fi

    # Add traceback format
    cmd="$cmd --tb=short"

    echo "$cmd"
}

# Function to show test summary
show_summary() {
    local exit_code=$1
    local duration=$2

    echo ""
    echo "========================= TEST SUMMARY ========================="
    echo "Test Type: $TEST_TYPE"
    echo "Duration: ${duration}s"

    if [ "$COVERAGE" = true ]; then
        echo "Coverage: Enabled ($COVERAGE_FORMAT format)"
        if [ "$COVERAGE_FORMAT" = "html" ]; then
            echo "HTML Report: htmlcov/index.html"
        elif [ "$COVERAGE_FORMAT" = "xml" ]; then
            echo "XML Report: coverage.xml"
        fi
    fi

    if [ -n "$OUTPUT_FILE" ]; then
        echo "Results saved to: $OUTPUT_FILE"
    fi

    if [ $exit_code -eq 0 ]; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed (exit code: $exit_code)"
    fi
    echo "============================================================="
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -f|--coverage-format)
            COVERAGE_FORMAT="$2"
            if [[ ! "$COVERAGE_FORMAT" =~ ^(term|html|xml)$ ]]; then
                log_error "Invalid coverage format: $COVERAGE_FORMAT"
                exit 1
            fi
            shift 2
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -x|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -k|--pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            VERBOSE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        all|unit|integration|regression|data|model|pipeline|fast)
            TEST_TYPE="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Print configuration
echo ""
log_info "=== OTEL ML Demo - Test Configuration ==="
log_info "Test Type: $TEST_TYPE"
log_info "Verbose: $VERBOSE"
log_info "Coverage: $COVERAGE"
if [ "$COVERAGE" = true ]; then
    log_info "Coverage Format: $COVERAGE_FORMAT"
fi
log_info "Parallel: $PARALLEL"
log_info "Fail Fast: $FAIL_FAST"
if [ -n "$TEST_PATTERN" ]; then
    log_info "Pattern: $TEST_PATTERN"
fi
if [ -n "$OUTPUT_FILE" ]; then
    log_info "Output File: $OUTPUT_FILE"
fi
echo ""

# Check dependencies
check_dependencies

# Create output directory if needed
if [ -n "$OUTPUT_FILE" ]; then
    mkdir -p "$(dirname "$OUTPUT_FILE")"
fi

# Build and run pytest command
log_test "Starting tests..."
START_TIME=$(date +%s)

PYTEST_CMD=$(build_pytest_cmd)

if [ "$VERBOSE" = true ]; then
    log_info "Running: $PYTEST_CMD"
fi

# Run tests
set +e  # Don't exit on pytest failure
eval "$PYTEST_CMD"
EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Show coverage info if enabled
if [ "$COVERAGE" = true ] && [ $EXIT_CODE -eq 0 ]; then
    echo ""
    if [ "$COVERAGE_FORMAT" = "html" ]; then
        log_coverage "HTML coverage report generated: htmlcov/index.html"
        if command -v xdg-open &> /dev/null; then
            read -p "Open coverage report in browser? (y/N): " open_browser
            if [[ $open_browser =~ ^[Yy]$ ]]; then
                xdg-open htmlcov/index.html
            fi
        fi
    elif [ "$COVERAGE_FORMAT" = "xml" ]; then
        log_coverage "XML coverage report generated: coverage.xml"
    fi
fi

# Show summary
show_summary $EXIT_CODE $DURATION

# Exit with the same code as pytest
exit $EXIT_CODE