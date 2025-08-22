import torch

if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA not available")