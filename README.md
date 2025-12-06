# GDELT Conflict Prediction using Time Series Anomaly Detection

## Overview
This project predicts conflicts using GDELT event data and LSTM Autoencoder-based time series anomaly detection. The model learns the "normal state" distribution of events for specific regions and detects anomalies that correspond to conflicts.

## Case Studies
1. **Israel-Palestine Conflict**
2. **Russia-Ukraine War**
3. **India-Pakistan Conflict**

## Methodology

### 1. Data Collection
- Source: GDELT 2.0 Event Database
- Features: All available GDELT features including:
  - Event codes (CAMEO)
  - Goldstein scale
  - Actor information
  - Geographic data
  - Tone, mentions, sources

### 2. Model Architecture
- **LSTM Autoencoder**: Learns temporal patterns in normal (non-conflict) periods
- **Anomaly Detection**: High reconstruction error indicates potential conflict
- **Country-specific models**: Each region has unique normal state distributions

### 3. Ground Truth
- Wikipedia-based conflict timelines for evaluation
- Major conflict events and escalations

## Project Structure
```
team_project/
├── data/
│   ├── raw/                # Raw GDELT data
│   ├── processed/          # Preprocessed features
│   └── ground_truth/       # Wikipedia conflict labels
├── src/
│   ├── data_collection/    # GDELT data fetching
│   ├── preprocessing/      # Data cleaning and transformation
│   ├── features/           # Feature engineering
│   ├── models/             # LSTM Autoencoder
│   ├── evaluation/         # Metrics and evaluation
│   └── utils/              # Helper functions
├── notebooks/              # Exploratory analysis
├── configs/                # Configuration files
├── models/                 # Saved models
├── results/                # Outputs and visualizations
└── requirements.txt
```

## Installation

### Quick Start
```bash
# 기본 설치
pip install -r requirements.txt
```

### With GPU Support (권장)
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지
pip install -r requirements.txt
```

자세한 GPU 설정: [GPU_SETUP.md](GPU_SETUP.md)

### With BigQuery (빠른 데이터 수집)
```bash
# BigQuery 패키지 설치
pip install google-cloud-bigquery google-auth db-dtypes

# 인증 설정
gcloud auth application-default login
```

자세한 BigQuery 설정: [BIGQUERY_SETUP.md](BIGQUERY_SETUP.md)

## Usage

### Complete Pipeline (권장)
```bash
# 전체 파이프라인 실행 (BigQuery + GPU)
python main.py --bigquery

# 특정 지역만
python main.py --bigquery --regions israel_palestine

# 데이터 수집 스킵 (기존 데이터 사용)
python main.py --skip-fetch

# 학습 스킵 (기존 모델 사용)
python main.py --skip-train
```

### Step-by-Step
```bash
# 1. Collect GDELT data (BigQuery - 빠름)
python src/data_collection/fetch_gdelt_bigquery.py --limit 1000  # 테스트
python src/data_collection/fetch_gdelt_bigquery.py  # 전체

# 1-b. Or direct download (느림, GCP 불필요)
python src/data_collection/fetch_gdelt.py

# 2. Create ground truth labels
python src/data_collection/create_ground_truth.py

# 3. Preprocess data
python src/preprocessing/preprocess.py

# 4. Train models (GPU 자동 사용)
python src/models/train.py

# 5. Evaluate and visualize
python src/evaluation/evaluate.py
```

## Key Features
- Country-specific normal state modeling
- Comprehensive GDELT feature extraction
- LSTM-based temporal pattern learning
- Real-time conflict anomaly detection
- Interactive visualization dashboard

## License
MIT License
