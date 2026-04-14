# OTEL ML Demo - SDSS Stellar Classification

A Python machine learning project for stellar object classification using the Sloan Digital Sky Survey (SDSS) dataset. This project demonstrates a complete ML pipeline from data loading to model inference, designed to be extended with OpenTelemetry instrumentation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-48%20passing-green.svg)](#testing)

## 🌟 Features

- **Complete ML Pipeline**: End-to-end stellar classification from raw data to predictions
- **Three-class Classification**: Distinguishes between galaxies, stars, and quasars
- **Robust Data Processing**: Handles missing values and irrelevant features automatically
- **Model Persistence**: Save and load trained models with preprocessing metadata
- **Flexible Inference**: Single observation or batch prediction capabilities
- **Command Line Interface**: Easy-to-use CLI for training and inference
- **Comprehensive Testing**: Unit, integration, and non-regression tests
- **Extensible Architecture**: Ready for OpenTelemetry instrumentation

## 📊 Dataset

The project uses SDSS (Sloan Digital Sky Survey) stellar classification data containing:

- **16 sample observations** (expandable to full SDSS dataset)
- **8 photometric features**: u, g, r, i, z bands + coordinates + redshift
- **3 target classes**: GALAXY, STAR, QSO (quasar)
- **Automatic preprocessing**: Removes observational metadata, preserves object IDs

### Feature Description

| Feature | Description | Unit |
|---------|-------------|------|
| `u`, `g`, `r`, `i`, `z` | Five-band photometry (ultraviolet to infrared) | magnitude |
| `alpha`, `delta` | Right ascension and declination coordinates | degrees |
| `redshift` | Cosmological redshift | dimensionless |
| `objid` | Unique object identifier | integer |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/k8s-school/otel-ml-demo.git
cd otel-ml-demo

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Basic Usage

#### 1. Train a Model

```bash
# Using convenience script (recommended)
./train.sh

# Quick training with fewer estimators
./train.sh --quick

# Custom parameters
./train.sh --n-estimators 200 --test-size 0.3 --verbose

# Direct Python module usage
python -m src.otel_ml_demo.pipeline --n-estimators 100 --test-size 0.3 --log-level INFO
```

#### 2. Make Predictions

```bash
# Interactive inference menu (recommended)
./infer.sh

# Quick examples
./infer.sh examples           # Run all example predictions
./infer.sh galaxy             # Predict galaxy example
./infer.sh star               # Predict star example
./infer.sh qso                # Predict quasar example

# Show model information
./infer.sh info

# Custom JSON prediction
./infer.sh '{"objid": 123, "u": 23.9, "g": 22.3, "r": 20.4, "i": 19.2, "z": 18.8, "alpha": 135.7, "delta": 32.5, "redshift": 0.63}'

# Predict from file
./infer.sh file my_data.json

# Direct Python module usage
python -m src.otel_ml_demo.inference --input '{"objid": 1237648720693755918, "u": 23.87882, "g": 22.2753, "r": 20.39501, "i": 19.16573, "z": 18.79371, "alpha": 135.689, "delta": 32.494, "redshift": 0.634794}'
```

#### 3. Run Tests

```bash
# Run all tests with coverage
./test.sh --coverage

# Run specific test categories
./test.sh unit                # Unit tests only
./test.sh integration         # Integration tests only
./test.sh data                # Data loader tests only

# Quick test run
./test.sh fast

# Verbose output with HTML coverage report
./test.sh --verbose --coverage --coverage-format html

# Direct pytest usage
pytest tests/ -v --cov=src --cov-report=html
```

#### 4. Example Output

```json
{
  "predicted_class": "GALAXY",
  "predicted_class_encoded": 0,
  "confidence": 0.9,
  "class_probabilities": {
    "GALAXY": 0.9,
    "QSO": 0.1,
    "STAR": 0.0
  },
  "objid": 1237648720693755918,
  "input_features": {
    "u": 23.87882,
    "g": 22.2753,
    "r": 20.39501,
    "i": 19.16573,
    "z": 18.79371,
    "alpha": 135.689,
    "delta": 32.494,
    "redshift": 0.634794
  }
}
```

## 🏗️ Project Structure

```
otel-ml-demo/
├── src/otel_ml_demo/           # Main package
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── model.py               # ML model implementation
│   ├── pipeline.py            # Training pipeline
│   └── inference.py           # Inference and prediction
├── tests/                     # Comprehensive test suite
│   ├── conftest.py           # Test fixtures
│   ├── test_data_loader.py   # Data loading tests
│   ├── test_model.py         # Model tests
│   ├── test_integration.py   # Integration tests
│   └── test_regression.py    # Non-regression tests
├── data/                      # Dataset storage
│   └── sample_sdss.csv       # Sample SDSS data
├── models/                    # Trained model artifacts
│   ├── stellar_classifier.joblib
│   └── data_preprocessing.joblib
├── logs/                      # Log files
├── train.sh                   # Training convenience script
├── infer.sh                   # Inference convenience script
├── test.sh                    # Testing convenience script
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## 🚀 Convenience Scripts

The project includes three powerful convenience scripts for common operations:

### `train.sh` - Model Training

```bash
./train.sh [OPTIONS]

# Examples
./train.sh                      # Default training
./train.sh --quick               # Fast training (10 estimators)
./train.sh -n 200 -t 0.3         # 200 estimators, 30% test split
./train.sh --verbose -f train.log # Verbose with log file
```

**Key Features:**
- ✅ Colored output with progress indicators
- ✅ Parameter validation and error handling
- ✅ Automatic model info display after training
- ✅ Duration tracking and performance metrics

### `infer.sh` - Model Inference

```bash
./infer.sh [MODE] [OPTIONS]

# Examples
./infer.sh                      # Interactive menu
./infer.sh info                 # Model information
./infer.sh examples             # Run all examples
./infer.sh galaxy               # Galaxy prediction
./infer.sh '{"objid": 123, ...}' # Custom JSON
```

**Key Features:**
- ✅ Interactive menu with 9 options
- ✅ Pre-loaded example observations (galaxy, star, quasar)
- ✅ Support for JSON input and file input
- ✅ Colored output with prediction highlighting

### `test.sh` - Test Suite

```bash
./test.sh [TEST_TYPE] [OPTIONS]

# Examples
./test.sh                       # All tests
./test.sh unit --verbose        # Unit tests with verbose output
./test.sh --coverage --coverage-format html # HTML coverage report
./test.sh integration -x        # Integration tests, stop on failure
```

**Key Features:**
- ✅ Multiple test categories (unit, integration, regression)
- ✅ Coverage reporting (term, HTML, XML)
- ✅ Parallel execution support
- ✅ Detailed test summaries with timing

## 🔧 API Reference

### Training Pipeline

```python
from src.otel_ml_demo.pipeline import train_model

results = train_model(
    data_dir="data",
    model_dir="models",
    test_size=0.2,
    n_estimators=100,
    random_state=42
)
```

### Inference

```python
from src.otel_ml_demo.inference import StellarInference

# Initialize and load model
inference = StellarInference(model_dir="models")
inference.load_model()

# Single prediction
observation = {
    "u": 23.9, "g": 22.3, "r": 20.4, "i": 19.2, "z": 18.8,
    "alpha": 135.7, "delta": 32.5, "redshift": 0.63
}
result = inference.predict(observation)

# Batch predictions
observations = [observation1, observation2, ...]
results = inference.batch_predict(observations)

# Model information
info = inference.get_model_info()
```

### Data Loading

```python
from src.otel_ml_demo.data_loader import SDSSDataLoader

# Load and preprocess data
loader = SDSSDataLoader(data_dir="data")
df = loader.load_raw_data()
X, y, objids = loader.preprocess_data(df)

# Get class names
class_names = loader.get_class_names()  # ['GALAXY', 'QSO', 'STAR']
```

## 📈 Model Performance

Based on the sample dataset (16 observations):

- **Training Accuracy**: 100% (perfect fit on small dataset)
- **Test Accuracy**: Variable (depends on random split)
- **Model Type**: Random Forest Classifier
- **Key Features**: z-band (31.1%), u-band (16.1%), r-band (15.5%)

### Feature Importance

The model identifies the most discriminative features for stellar classification:

1. **z-band magnitude** (31.1%) - Infrared photometry
2. **u-band magnitude** (16.1%) - Ultraviolet photometry
3. **r-band magnitude** (15.5%) - Red photometry
4. **i-band magnitude** (12.4%) - Near-infrared photometry
5. **redshift** (12.3%) - Cosmological distance indicator

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_data_loader.py -v     # Unit tests
pytest tests/test_integration.py -v    # Integration tests
pytest tests/test_regression.py -v     # Non-regression tests
```

### Test Categories

- **Unit Tests** (30 tests): Individual component functionality
- **Integration Tests** (8 tests): End-to-end pipeline validation
- **Non-regression Tests** (10 tests): Consistent behavior verification

## 🔄 Development Workflow

1. **Data Preparation**: Load and preprocess SDSS data
2. **Model Training**: Train Random Forest classifier
3. **Model Evaluation**: Assess performance with metrics
4. **Model Persistence**: Save trained model and metadata
5. **Inference**: Load model and make predictions
6. **Testing**: Run comprehensive test suite

## 📝 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `"data"` | Data directory path |
| `--model-dir` | `"models"` | Model output directory |
| `--test-size` | `0.2` | Test set fraction (0.0-1.0) |
| `--n-estimators` | `100` | Number of trees in Random Forest |
| `--random-state` | `42` | Random seed for reproducibility |
| `--log-level` | `"INFO"` | Logging level |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-dir` | `"models"` | Model directory path |
| `--input` | - | JSON input (file or string) |
| `--output` | - | Output file (optional) |
| `--example` | - | Show example usage |
| `--model-info` | - | Show model information |

## 🚧 Future Enhancements

- [ ] **OpenTelemetry Integration**: Add distributed tracing and metrics
- [ ] **Full SDSS Dataset**: Scale to complete dataset (100k+ observations)
- [ ] **Advanced Models**: XGBoost, Neural Networks, Ensemble methods
- [ ] **Feature Engineering**: Color indices, magnitude ratios
- [ ] **Model Serving**: REST API with FastAPI
- [ ] **Monitoring**: MLflow integration for experiment tracking
- [ ] **Containerization**: Docker deployment
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SDSS Collaboration** for providing astronomical data
- **Scikit-learn** for machine learning algorithms
- **Python Community** for excellent tooling and libraries

## 📚 References

- [SDSS Data Release Documentation](https://www.sdss4.org/dr17/)
- [Stellar Classification with Machine Learning](https://arxiv.org/abs/astro-ph/0606544)
- [Random Forest Algorithm](https://doi.org/10.1023/A:1010933404324)

---

**Ready for OpenTelemetry integration!** 🎯