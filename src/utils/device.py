"""
Device management utilities for GPU/CPU selection
"""

import torch
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger("device")


def get_device(config: Dict) -> str:
    """
    Get device based on configuration and availability

    Args:
        config: Configuration dictionary

    Returns:
        Device string ('cuda', 'cuda:0', 'cpu', etc.)
    """
    model_config = config.get('model', {})
    device_setting = model_config.get('device', 'auto')
    use_gpu = model_config.get('use_gpu', True)
    gpu_id = model_config.get('gpu_id', 0)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if device_setting == 'auto':
        # Auto-detect: use GPU if available and enabled
        if cuda_available and use_gpu:
            device = f'cuda:{gpu_id}' if gpu_id is not None else 'cuda'
            gpu_name = torch.cuda.get_device_name(gpu_id if gpu_id is not None else 0)
            logger.info(f"GPU detected: {gpu_name}")
            logger.info(f"Using device: {device}")
        else:
            device = 'cpu'
            if not cuda_available:
                logger.info("CUDA not available, using CPU")
            else:
                logger.info("GPU disabled in config, using CPU")

    elif device_setting == 'cpu':
        device = 'cpu'
        logger.info("Using device: CPU (forced by config)")

    elif device_setting.startswith('cuda'):
        if cuda_available:
            device = device_setting
            gpu_name = torch.cuda.get_device_name(
                int(device_setting.split(':')[1]) if ':' in device_setting else 0
            )
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Using device: {device}")
        else:
            logger.warning(f"CUDA not available, falling back to CPU (config requested: {device_setting})")
            device = 'cpu'

    else:
        logger.warning(f"Unknown device setting '{device_setting}', using CPU")
        device = 'cpu'

    # Print GPU memory info if using CUDA
    if device.startswith('cuda'):
        gpu_idx = int(device.split(':')[1]) if ':' in device else 0
        total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        logger.info(f"GPU Memory: {total_memory:.2f} GB")

    return device


def print_gpu_info():
    """Print detailed GPU information"""
    if torch.cuda.is_available():
        logger.info("="*50)
        logger.info("GPU INFORMATION")
        logger.info("="*50)
        logger.info(f"CUDA Available: True")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"\nGPU {i}:")
            logger.info(f"  Name: {props.name}")
            logger.info(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            logger.info(f"  Multi Processors: {props.multi_processor_count}")

        logger.info("="*50)
    else:
        logger.info("No GPU available. Using CPU.")


def optimize_for_device(model, device: str):
    """
    Optimize model for specific device

    Args:
        model: PyTorch model
        device: Device string

    Returns:
        Optimized model
    """
    model = model.to(device)

    if device.startswith('cuda'):
        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled CUDNN benchmark for faster training")

    return model


if __name__ == "__main__":
    # Test GPU detection
    print_gpu_info()

    # Test device selection
    test_config = {
        'model': {
            'device': 'auto',
            'use_gpu': True,
            'gpu_id': 0
        }
    }

    device = get_device(test_config)
    print(f"\nSelected device: {device}")
