# Check if GPU or CPU is being used
import torch

if torch.cuda.is_available():
    device_message = f"Training will start using GPU: {torch.cuda.get_device_name(0)}"
else:
    device_message = "Training will start using CPU"

print(device_message)
