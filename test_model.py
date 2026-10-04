import torch

from model import UNet3D

model = UNet3D()

model.load_state_dict(
    torch.load(
        "3D_UNet_best_model (1).pth",
        map_location="cpu",
        weights_only=False
    )
)

model.eval()

print("3D U-Net model loaded successfully!")

print(
    "Number of parameters:",
    sum(p.numel() for p in model.parameters())
)