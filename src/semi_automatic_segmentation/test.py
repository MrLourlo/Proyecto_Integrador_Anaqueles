import torch
print(f"PyTorch version: {torch.__version__}")
is_available = torch.cuda.is_available()
print(f"CUDA available: {is_available}")
if is_available:
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")