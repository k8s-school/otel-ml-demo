#!/bin/bash

# OTEL ML Demo - Inference Script
# Make predictions with trained SDSS stellar classification model

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
MODEL_DIR="models"
LOG_LEVEL="INFO"
OUTPUT_FILE=""
MODE="interactive"

# Sample observations for testing
GALAXY_EXAMPLE='{"objid": 1237648720693755918, "u": 23.87882, "g": 22.2753, "r": 20.39501, "i": 19.16573, "z": 18.79371, "alpha": 135.689, "delta": 32.494, "redshift": 0.634794}'
STAR_EXAMPLE='{"objid": 1237648668670849481, "u": 19.43718, "g": 17.58028, "r": 16.49747, "i": 15.97711, "z": 15.54461, "alpha": 345.282, "delta": 21.183, "redshift": 0.000010}'
QSO_EXAMPLE='{"objid": 1237648720693755922, "u": 21.45678, "g": 19.87654, "r": 18.23456, "i": 17.56789, "z": 16.89012, "alpha": 234.567, "delta": 12.345, "redshift": 1.234567}'

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [MODE]"
    echo ""
    echo "Make predictions with trained SDSS stellar classification model"
    echo ""
    echo "Modes:"
    echo "  interactive               Interactive mode with menu (default)"
    echo "  info                     Show model information"
    echo "  examples                 Run all example predictions"
    echo "  galaxy                   Predict galaxy example"
    echo "  star                     Predict star example"
    echo "  qso                      Predict quasar example"
    echo "  custom JSON              Predict custom observation"
    echo "  file PATH                Predict from JSON file"
    echo ""
    echo "Options:"
    echo "  -m, --model-dir PATH     Model directory (default: models)"
    echo "  -o, --output FILE        Output file (optional)"
    echo "  -l, --log-level LEVEL    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    echo "  -q, --quiet              Quiet mode (ERROR level logging)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Interactive mode"
    echo "  $0 info                  # Show model information"
    echo "  $0 examples              # Run all examples"
    echo "  $0 galaxy                # Predict galaxy example"
    echo "  $0 '{\"u\": 20, ...}'     # Custom prediction"
    echo "  $0 file data.json        # Predict from file"
    echo "  $0 star -o result.json   # Save result to file"
}

# Function to log messages
log_info() {
    if [ "$LOG_LEVEL" != "ERROR" ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
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

log_prediction() {
    echo -e "${CYAN}[PREDICTION]${NC} $1"
}

# Function to run inference
run_inference() {
    local input="$1"
    local description="$2"

    if [ -n "$description" ]; then
        log_info "Predicting: $description"
    fi

    local cmd="source venv/bin/activate && python -m src.otel_ml_demo.inference --model-dir '$MODEL_DIR' --log-level '$LOG_LEVEL'"

    if [ -n "$OUTPUT_FILE" ]; then
        cmd="$cmd --output '$OUTPUT_FILE'"
    fi

    if [ -f "$input" ]; then
        cmd="$cmd --input '$input'"
    else
        cmd="$cmd --input '$input'"
    fi

    eval "$cmd"
}

# Function to show model info
show_model_info() {
    log_info "=== Model Information ==="
    source venv/bin/activate && python -m src.otel_ml_demo.inference --model-dir "$MODEL_DIR" --model-info --log-level ERROR
}

# Function to show interactive menu
interactive_mode() {
    while true; do
        echo ""
        echo -e "${CYAN}=== OTEL ML Demo - Inference Menu ===${NC}"
        echo "1) Show model information"
        echo "2) Predict galaxy example"
        echo "3) Predict star example"
        echo "4) Predict quasar example"
        echo "5) Run all examples"
        echo "6) Custom prediction (enter JSON)"
        echo "7) Predict from file"
        echo "8) Show usage examples"
        echo "9) Exit"
        echo ""
        read -p "Choose an option (1-9): " choice

        case $choice in
            1)
                show_model_info
                ;;
            2)
                log_prediction "Galaxy example"
                run_inference "$GALAXY_EXAMPLE" "Galaxy (objid: 1237648720693755918)"
                ;;
            3)
                log_prediction "Star example"
                run_inference "$STAR_EXAMPLE" "Star (objid: 1237648668670849481)"
                ;;
            4)
                log_prediction "Quasar example"
                run_inference "$QSO_EXAMPLE" "Quasar (objid: 1237648720693755922)"
                ;;
            5)
                log_info "Running all examples..."
                run_inference "$GALAXY_EXAMPLE" "Galaxy example"
                echo ""
                run_inference "$STAR_EXAMPLE" "Star example"
                echo ""
                run_inference "$QSO_EXAMPLE" "Quasar example"
                ;;
            6)
                echo ""
                echo "Enter JSON observation (or press Enter for example):"
                echo "Format: {\"objid\": 123, \"u\": 23.9, \"g\": 22.3, \"r\": 20.4, \"i\": 19.2, \"z\": 18.8, \"alpha\": 135.7, \"delta\": 32.5, \"redshift\": 0.63}"
                read -p "> " custom_json
                if [ -z "$custom_json" ]; then
                    custom_json="$GALAXY_EXAMPLE"
                    log_info "Using galaxy example"
                fi
                run_inference "$custom_json" "Custom observation"
                ;;
            7)
                read -p "Enter JSON file path: " file_path
                if [ -f "$file_path" ]; then
                    run_inference "$file_path" "File: $file_path"
                else
                    log_error "File not found: $file_path"
                fi
                ;;
            8)
                source venv/bin/activate && python -m src.otel_ml_demo.inference --example
                ;;
            9)
                log_info "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid option: $choice"
                ;;
        esac
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -q|--quiet)
            LOG_LEVEL="ERROR"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ "$MODE" = "interactive" ]; then
                MODE="$1"
            else
                log_error "Too many arguments"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log_error "Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_DIR/stellar_classifier.joblib" ]; then
    log_error "Trained model not found in $MODEL_DIR/"
    log_error "Please run training first: ./train.sh"
    exit 1
fi

# Handle different modes
case $MODE in
    interactive)
        interactive_mode
        ;;
    info)
        show_model_info
        ;;
    examples)
        log_info "=== Running All Examples ==="
        echo ""
        log_prediction "Galaxy Example"
        run_inference "$GALAXY_EXAMPLE" "Galaxy (bright, high redshift)"
        echo ""
        log_prediction "Star Example"
        run_inference "$STAR_EXAMPLE" "Star (bright, very low redshift)"
        echo ""
        log_prediction "Quasar Example"
        run_inference "$QSO_EXAMPLE" "Quasar (moderate brightness, high redshift)"
        ;;
    galaxy)
        log_prediction "Galaxy Example"
        run_inference "$GALAXY_EXAMPLE" "Galaxy (objid: 1237648720693755918)"
        ;;
    star)
        log_prediction "Star Example"
        run_inference "$STAR_EXAMPLE" "Star (objid: 1237648668670849481)"
        ;;
    qso|quasar)
        log_prediction "Quasar Example"
        run_inference "$QSO_EXAMPLE" "Quasar (objid: 1237648720693755922)"
        ;;
    file)
        if [ $# -eq 0 ]; then
            log_error "File path required for 'file' mode"
            exit 1
        fi
        file_path="$1"
        if [ -f "$file_path" ]; then
            run_inference "$file_path" "File: $file_path"
        else
            log_error "File not found: $file_path"
            exit 1
        fi
        ;;
    *)
        # Treat as custom JSON input
        if [[ "$MODE" == "{"* ]]; then
            log_prediction "Custom Observation"
            run_inference "$MODE" "Custom JSON input"
        else
            log_error "Unknown mode: $MODE"
            usage
            exit 1
        fi
        ;;
esac