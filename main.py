import torch

gpu_available = torch.cuda.is_available()

if gpu_available:
    print("GPU is available!")
else:
    print("GPU is not available. Using CPU.")
