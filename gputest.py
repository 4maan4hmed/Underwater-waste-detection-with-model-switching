import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU will likely be used.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Ultralytics will likely use the CPU.")