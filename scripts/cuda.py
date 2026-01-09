import torch

print("=====================================")
print(" PyTorch CUDA Info")
print("=====================================")
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name())
else:
    print("⚠️  CUDA is NOT available. Training will run on CPU.")
print("=====================================\n")
