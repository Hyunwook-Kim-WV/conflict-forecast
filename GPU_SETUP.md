# GPU Setup Guide

## GPU를 사용하는 이유

GPU를 사용하면 모델 학습 속도가 **10-50배 빠르게** 됩니다:

- **CPU**: LSTM 학습에 수 시간 소요
- **GPU**: LSTM 학습에 수 분 소요

특히 LSTM Autoencoder와 같은 RNN 모델은 GPU 가속의 이점이 큽니다.

## GPU 확인

현재 시스템에 NVIDIA GPU가 있는지 확인:

### Windows
```bash
nvidia-smi
```

### Linux/Mac
```bash
lspci | grep -i nvidia
```

출력 예시:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    10W / 250W |    500MiB /  8192MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
```

## Setup Options

### Option 1: CUDA Toolkit 설치 (권장)

#### Windows

1. **NVIDIA Driver 설치**
   - [NVIDIA Driver 다운로드](https://www.nvidia.com/Download/index.aspx)
   - 본인의 GPU 모델에 맞는 드라이버 설치

2. **CUDA Toolkit 설치**
   - [CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
   - CUDA 11.8 or 12.1 권장 (PyTorch 호환성)

3. **cuDNN 설치**
   - [cuDNN 다운로드](https://developer.nvidia.com/cudnn)
   - NVIDIA 계정 필요 (무료)
   - CUDA 버전과 호환되는 cuDNN 선택

4. **PyTorch GPU 버전 설치**
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

#### Linux (Ubuntu/Debian)

```bash
# NVIDIA Driver
sudo apt update
sudo apt install nvidia-driver-535

# CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# PyTorch
pip install torch torchvision torchaudio
```

### Option 2: Conda (간편함)

Conda를 사용하면 CUDA와 cuDNN이 자동으로 설치됩니다.

```bash
# Conda 설치 (Miniconda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 환경 생성
conda create -n gdelt python=3.10
conda activate gdelt

# PyTorch GPU 설치 (CUDA 자동 설치)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 나머지 패키지
pip install -r requirements.txt
```

## Configuration

### config.yaml 설정

```yaml
model:
  # GPU 자동 감지 (권장)
  device: "auto"
  use_gpu: true
  gpu_id: 0

  # 또는 수동으로 지정
  # device: "cuda:0"  # 첫 번째 GPU 사용
  # device: "cpu"     # CPU 강제 사용
```

### 다중 GPU 시스템

여러 GPU가 있는 경우:

```yaml
model:
  device: "auto"
  use_gpu: true
  gpu_id: 1  # 두 번째 GPU 사용 (0부터 시작)
```

GPU별로 사용 가능한 메모리 확인:
```bash
nvidia-smi
```

## Verification

### GPU 작동 확인

```python
import torch

# CUDA 사용 가능 여부
print(f"CUDA available: {torch.cuda.is_available()}")

# GPU 정보
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

### 실행 시 GPU 확인

프로젝트 실행 시 GPU 정보가 자동으로 출력됩니다:

```bash
python main.py
```

출력:
```
==================================================
GPU INFORMATION
==================================================
CUDA Available: True
CUDA Version: 12.1
PyTorch Version: 2.1.0
Number of GPUs: 1

GPU 0:
  Name: NVIDIA GeForce RTX 3080
  Total Memory: 10.00 GB
  Compute Capability: 8.6
  Multi Processors: 68
==================================================
```

## Performance Optimization

### Batch Size 조정

GPU 메모리에 맞게 batch size 조정:

```yaml
model:
  training:
    batch_size: 64  # GPU 메모리가 충분하면 증가
```

**메모리 부족 시**:
```
RuntimeError: CUDA out of memory
```
→ batch_size를 절반으로 줄이기 (64 → 32 → 16)

### Mixed Precision Training (선택사항)

더 빠른 학습을 위해:

```python
# src/models/train.py에 추가
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
with autocast():
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### CUDNN Benchmark

자동으로 활성화됨 (src/utils/device.py):
```python
torch.backends.cudnn.benchmark = True
```

## Troubleshooting

### Error: "CUDA out of memory"

**해결책 1**: Batch size 줄이기
```yaml
model:
  training:
    batch_size: 16  # 32에서 16으로
```

**해결책 2**: Sequence length 줄이기
```yaml
model:
  lstm:
    sequence_length: 15  # 30에서 15로
```

**해결책 3**: GPU 메모리 정리
```python
import torch
torch.cuda.empty_cache()
```

### Error: "CUDA driver version is insufficient"

NVIDIA 드라이버를 업데이트하세요:
```bash
# Ubuntu
sudo apt update
sudo apt upgrade nvidia-driver-535

# Windows
# NVIDIA 웹사이트에서 최신 드라이버 다운로드
```

### Error: "No CUDA runtime is found"

PyTorch GPU 버전이 제대로 설치되지 않았습니다:
```bash
# 기존 PyTorch 제거
pip uninstall torch torchvision torchaudio

# GPU 버전 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance is slower on GPU

**원인**: 데이터가 작거나 모델이 단순할 때 GPU 오버헤드가 더 클 수 있음

**해결**:
1. Batch size 증가
2. 데이터셋 크기 확인
3. CPU로 강제 실행: `device: "cpu"`

## Monitoring

### 실시간 GPU 사용률

```bash
# 1초마다 업데이트
watch -n 1 nvidia-smi

# 또는
nvidia-smi dmon -s u
```

### GPU 메모리 사용량 추적

```python
import torch

# 현재 메모리 사용량
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved: {reserved:.2f} GB")
```

## Performance Comparison

예시 (RTX 3080, LSTM Autoencoder):

| Device | Batch Size | Epoch Time | Total Training |
|--------|-----------|-----------|----------------|
| CPU (i7-10700K) | 32 | 12 min | 20 hours |
| GPU (RTX 3080) | 64 | 45 sec | 1.2 hours |
| GPU (RTX 3080) | 128 | 30 sec | 50 min |

**속도 향상**: ~16-24배

## Best Practices

1. **항상 GPU 활성화**: `use_gpu: true`
2. **Batch size 최대화**: GPU 메모리 허용 한도까지
3. **데이터 로딩 최적화**: `num_workers` 설정
4. **모니터링**: `nvidia-smi` 확인
5. **메모리 정리**: 학습 후 `torch.cuda.empty_cache()`

## Cloud GPU Options

로컬 GPU가 없다면 클라우드 사용:

### Google Colab (무료)
- GPU: Tesla T4 (무료)
- 제한: 12시간 세션

### AWS EC2
- GPU: V100, A100
- 비용: $0.50-3/hour

### Paperspace
- GPU: Various
- 비용: $0.45/hour

### Lambda Labs
- GPU: A100
- 비용: $1.10/hour

## Next Steps

1. GPU 설치 확인: `nvidia-smi`
2. PyTorch GPU 버전 설치
3. 설정 확인: `python -c "import torch; print(torch.cuda.is_available())"`
4. 학습 시작: `python main.py`

## Resources

- [PyTorch CUDA 설치 가이드](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
