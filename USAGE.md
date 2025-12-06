# Usage Guide

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run all steps for all regions
python main.py

# Run for specific regions
python main.py --regions israel_palestine russia_ukraine

# Skip data fetching (use existing data)
python main.py --skip-fetch

# Skip training (use existing models)
python main.py --skip-train
```

### 3. View Results

Results are saved in the `results/` directory:
- `{region}/metrics.txt` - Performance metrics
- `{region}/confusion_matrix.png` - Confusion matrix
- `{region}/roc_curve.png` - ROC curve
- `{region}/timeline.png` - Detection timeline
- `{region}/error_distribution.png` - Error distribution

## Step-by-Step Usage

### Step 1: Fetch GDELT Data

```python
from src.data_collection.fetch_gdelt import GDELTFetcher
from src.utils.config_loader import load_config

config = load_config()
fetcher = GDELTFetcher()

# Fetch data for all regions
data = fetcher.fetch_all_regions(config)
```

### Step 2: Create Ground Truth Labels

```python
from src.data_collection.create_ground_truth import GroundTruthCreator

creator = GroundTruthCreator()
labels = creator.create_all_labels(config)
```

### Step 3: Preprocess Data

```python
from src.preprocessing.preprocess import GDELTPreprocessor

preprocessor = GDELTPreprocessor()
processed = preprocessor.process_all_regions(config)
```

### Step 4: Train Model

```python
from src.features.feature_engineering import FeatureEngineer
from src.models.train import train_region_model

engineer = FeatureEngineer()

# Prepare training data
train_seq, train_labels, train_dates = engineer.prepare_training_data(
    df=processed_df,
    labels=labels_df,
    sequence_length=30,
    use_normal_only=True
)

# Train model
model, trainer = train_region_model(
    region_name='israel_palestine',
    config=config,
    train_sequences=train_seq
)
```

### Step 5: Evaluate

```python
from src.models.lstm_autoencoder import AnomalyDetector
from src.evaluation.evaluate import ConflictEvaluator

# Create detector
detector = AnomalyDetector(model)
detector.compute_threshold(train_seq)

# Predict
scores, predictions = detector.predict(test_seq)

# Evaluate
evaluator = ConflictEvaluator()
metrics = evaluator.compute_metrics(test_labels, predictions, scores)
```

## Configuration

Edit `configs/config.yaml` to customize:

### Regions

```yaml
regions:
  your_region:
    name: "Your Region"
    countries: ["ABC", "XYZ"]
    actor_keywords: ["KEYWORD1", "KEYWORD2"]
    date_range:
      start: "2020-01-01"
      end: "2024-12-31"
```

### Model Parameters

```yaml
model:
  lstm:
    sequence_length: 30
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2
    bidirectional: true

  training:
    batch_size: 32
    epochs: 100
    learning_rate: 0.001
    early_stopping_patience: 10

  anomaly_detection:
    threshold_method: "percentile"  # or "std"
    threshold_percentile: 95
```

## Jupyter Notebook Analysis

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The notebook includes:
- Time series visualization
- Feature distribution analysis
- Correlation analysis
- Event code distribution
- Summary statistics

## Adding New Regions

1. Edit `configs/config.yaml`:

```yaml
regions:
  new_region:
    name: "New Region"
    countries: ["AAA", "BBB"]
    actor_keywords: ["KEYWORD"]
    date_range:
      start: "2020-01-01"
      end: "2024-12-31"
```

2. Add ground truth events in `src/data_collection/create_ground_truth.py`:

```python
CONFLICT_EVENTS = {
    'new_region': [
        ('2020-01-01', '2020-02-01', 'Event Name'),
        # Add more events...
    ]
}
```

3. Run pipeline:

```bash
python main.py --regions new_region
```

## Troubleshooting

### Out of Memory Error

Reduce batch size in config:
```yaml
model:
  training:
    batch_size: 16  # or lower
```

### GDELT Fetch Timeout

The fetcher includes automatic retry logic. If issues persist:
- Check internet connection
- Try fetching smaller date ranges
- Use `--skip-fetch` and manually download GDELT data

### GPU Not Detected

Install CUDA-compatible PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Performance Tips

1. **Use GPU**: Significantly faster training
2. **Adjust sequence length**: Shorter sequences = faster training
3. **Use validation split**: Prevent overfitting
4. **Monitor training**: Check `models/{region}/training_history.png`

## Contact

For issues and questions:
- Check logs in `logs/` directory
- Review configuration in `configs/config.yaml`
- Examine sample data in `notebooks/`
