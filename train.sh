#!/bin/bash

# OTEL ML Demo - Training Script
# Train SDSS stellar classification model

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
N_ESTIMATORS=100
TEST_SIZE=0.2
RANDOM_STATE=42
LOG_LEVEL="INFO"
DATA_DIR="data"
MODEL_DIR="models"
LOG_FILE=""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train SDSS stellar classification model"
    echo ""
    echo "Options:"
    echo "  -n, --n-estimators NUM    Number of trees in Random Forest (default: 100)"
    echo "  -t, --test-size FLOAT     Test set fraction 0.0-1.0 (default: 0.2)"
    echo "  -r, --random-state NUM    Random seed for reproducibility (default: 42)"
    echo "  -l, --log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  -d, --data-dir PATH       Data directory (default: data)"
    echo "  -m, --model-dir PATH      Model output directory (default: models)"
    echo "  -f, --log-file FILE       Log file name (optional)"
    echo "  -q, --quick               Quick training with 10 estimators"
    echo "  -v, --verbose             Verbose output (DEBUG level)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Train with default parameters"
    echo "  $0 --quick                # Quick training (10 estimators)"
    echo "  $0 -n 200 -t 0.3         # 200 estimators, 30% test split"
    echo "  $0 --verbose -f train.log # Verbose logging to file"
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n-estimators)
            N_ESTIMATORS="$2"
            shift 2
            ;;
        -t|--test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        -r|--random-state)
            RANDOM_STATE="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -f|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -q|--quick)
            N_ESTIMATORS=10
            log_info "Quick training mode: using 10 estimators"
            shift
            ;;
        -v|--verbose)
            LOG_LEVEL="DEBUG"
            log_info "Verbose mode enabled"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate parameters
if ! [[ "$N_ESTIMATORS" =~ ^[0-9]+$ ]] || [ "$N_ESTIMATORS" -lt 1 ]; then
    log_error "Invalid n-estimators: must be a positive integer"
    exit 1
fi

if ! [[ "$TEST_SIZE" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$TEST_SIZE < 0" | bc -l) )) || (( $(echo "$TEST_SIZE > 1" | bc -l) )); then
    log_error "Invalid test-size: must be between 0.0 and 1.0"
    exit 1
fi

if ! [[ "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]]; then
    log_error "Invalid log-level: must be DEBUG, INFO, WARNING, or ERROR"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log_error "Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

# Check if source files exist
if [ ! -f "src/otel_ml_demo/pipeline.py" ]; then
    log_error "Pipeline source not found. Please ensure you're in the project root directory."
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$MODEL_DIR" "logs"

# Print configuration
echo ""
log_info "=== OTEL ML Demo - Training Configuration ==="
log_info "N_ESTIMATORS: $N_ESTIMATORS"
log_info "TEST_SIZE: $TEST_SIZE"
log_info "RANDOM_STATE: $RANDOM_STATE"
log_info "LOG_LEVEL: $LOG_LEVEL"
log_info "DATA_DIR: $DATA_DIR"
log_info "MODEL_DIR: $MODEL_DIR"
if [ -n "$LOG_FILE" ]; then
    log_info "LOG_FILE: logs/$LOG_FILE"
fi
echo ""

# Build command
CMD="source venv/bin/activate && python -m src.otel_ml_demo.pipeline"
CMD="$CMD --data-dir '$DATA_DIR'"
CMD="$CMD --model-dir '$MODEL_DIR'"
CMD="$CMD --n-estimators $N_ESTIMATORS"
CMD="$CMD --test-size $TEST_SIZE"
CMD="$CMD --random-state $RANDOM_STATE"
CMD="$CMD --log-level $LOG_LEVEL"

if [ -n "$LOG_FILE" ]; then
    CMD="$CMD --log-file '$LOG_FILE'"
fi

# Run training
log_info "Starting model training..."
START_TIME=$(date +%s)

if eval "$CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    log_success "Training completed successfully!"
    log_success "Duration: ${DURATION}s"

    # Show model info if training succeeded
    if [ -f "$MODEL_DIR/stellar_classifier.joblib" ]; then
        echo ""
        log_info "=== Model Information ==="
        source venv/bin/activate && python -m src.otel_ml_demo.inference --model-dir "$MODEL_DIR" --model-info
    fi

    echo ""
    log_success "Model saved to: $MODEL_DIR/"
    log_success "Ready for inference! Try: ./infer.sh"

else
    log_error "Training failed!"
    exit 1
fi