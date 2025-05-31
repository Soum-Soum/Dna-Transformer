# 🧬 DNA Transformer

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.51.3+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*A state-of-the-art transformer architecture for genomic sequence classification and analysis*

</div>

## 🌟 Features

- **🔬 Advanced DNA Analysis**: Modern transformer architectures (BERT & ModernBERT) optimized for genomic data
- **⚡ High Performance**: GPU-accelerated training with CUDA support and optimized data loading
- **📊 Rich Visualizations**: t-SNE embeddings, training metrics, and error analysis plots
- **🎯 Flexible Training**: Support for various sequence lengths, overlapping ratios, and model configurations
- **💾 Smart Caching**: Automatic result caching for faster repeated analyses
- **🔧 Easy CLI**: User-friendly command-line interface for all operations

## 🚀 Quick Start

### 1. Environment Setup

#### Using UV (Recommended)
```bash
# Install UV if you haven't already
pipx install uv

# Clone the repository
git clone <your-repo-url>
cd Dna-Transformer

# Create and activate virtual environment with UV
uv venv --python 3.13
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv sync
```

### 2. Data Setup with DVC

#### Install DVC
```bash
pipx install 'dvc[s3]' --pip-args="boto3<1.35 botocore<1.35 s3fs<2024.1 aiobotocore<2.15"
```

#### Download Data
```bash
# Pull data from DVC remote
dvc pull data.dvc
dvc pull output.dvc

# Or initialize DVC if starting fresh
dvc init --no-scm
dvc remote add -d myremote s3://your-bucket/data
dvc pull
```

### 3. Data Preprocessing

Transform your HapMap data into the required format:

```bash
# Convert HapMap file to smaller parquet chunks
python cli.py hapmap-to-parquet \
    --hapmap-path data/DIVRICE_3k10M/DIVRICE_3k10M_ch1.hapmap \
    --output-dir data/output \
    --chunk-size 5000

# Process chunks into individual SNP files
python cli.py hapmap-to-snp-per-individual \
    --metadata-path data/DIVRICE_3k10M/DIVRICE_3k10M_metadata.tsv \
    --base-dir data/output \
    --max-workers 10
```

## 🎯 Usage

### Training a Model

```bash
python train.py \
    --base-dir data/output \
    --output-dir output \
    --run-name "my-dna-model" \
    --sequence-length 150 \
    --batch-size 256 \
    --epochs 20 \
    --learning-rate 5e-5 \
    --model-type modern_bert \
    --model-dim 256
```

### Making Predictions

```bash
python predict.py \
    --base-dir data/output \
    --checkpoint-dir output/my-dna-model/checkpoints/checkpoint-xxx \
    --sequence-length 150 \
    --batch-size 256 \
    --overlapping-ratio 0.5
```

### Training a Custom Tokenizer

```bash
python train_tokenizer.py \
    --base-dir data/output \
    --vocab-size 1000 \
    --output-dir tokenizers/my-tokenizer
```

## 🏗️ Architecture

### Model Types
- **BERT**: Classic bidirectional encoder for sequence classification
- **ModernBERT**: Enhanced architecture with improved efficiency and performance

### Key Components
- **Activation Shaping**: Novel pruning technique for model optimization
- **DNA-specific Tokenization**: Custom tokenizers designed for genomic sequences
- **Multi-label Classification**: Support for complex genomic label hierarchies

## 📊 Model Configuration

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `sequence_length` | Length of DNA sequences | 150 | 50-1000 |
| `batch_size` | Training batch size | 256 | 16-512 |
| `model_dim` | Model hidden dimension | 128 | 64-1024 |
| `overlapping_ratio` | Sequence overlap for prediction | 0.5 | 0.0-1.0 |
| `learning_rate` | Optimizer learning rate | 5e-5 | 1e-6-1e-3 |

## 📁 Project Structure

```
Dna-Transformer/
├── 📄 README.md                    # You are here!
├── 🐍 train.py                     # Model training script
├── 🔮 predict.py                   # Prediction script
├── 🎯 cli.py                       # Data processing CLI
├── 📦 pyproject.toml               # Project dependencies
├── 📊 data/                        # Raw and processed data
│   ├── DIVRICE_3k10M/             # Raw HapMap data
│   └── output/                    # Processed datasets
├── 🤖 adn/                         # Core library
│   ├── 🧠 models/                  # Model architectures
│   ├── 📈 data/                    # Data processing
│   ├── 🎨 plots.py                 # Visualization utilities
│   └── 🔧 cli/                     # CLI commands
└── 📈 output/                      # Training outputs
```

## 🔧 Advanced Usage

### Custom Model Training

```python
from adn.models.transformers.modern_bert import DnaModernBertForSequenceClassification
from adn.data.data import load_datasets
from adn.utils.paths_utils import PathHelper

# Load your data
path_helper = PathHelper("data/output")
train_ds, eval_ds = load_datasets(
    path_helper=path_helper,
    sequence_length=150,
    train_eval_split=0.1,
    mode=DatasetMode.RANDOM_FIXED_LEN
)

# Initialize model
model = DnaModernBertForSequenceClassification(config)

# Train with custom parameters...
```

### Analysis and Visualization

```python
from adn.prediction_results import OnDiskPredictionResults
from adn.plots import plot_tsne

# Analyze prediction results
results = OnDiskPredictionResults("output/my-model/predictions")
centroids = results.compute_centroids()
distances = results.compute_distances()
errors = results.compute_error()

# Create t-SNE visualization
plot_tsne(results_df, centroids, output_dir="plots")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [🤗 Transformers](https://huggingface.co/transformers/)
- Powered by [PyTorch](https://pytorch.org/)
- Data management with [DVC](https://dvc.org/)
- Fast dependency management with [UV](https://astral.sh/uv/)

---

<div align="center">
<strong>Made with ❤️ for genomic research</strong>
</div>