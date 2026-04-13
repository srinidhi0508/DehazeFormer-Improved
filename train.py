import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loader import PairLoader
from models import dehazeformer_t

from pytorch_msssim import ssim
import torchvision.models as models
import torch.nn.functional as F

# -------- LOAD CONFIG --------
config_path = "configs/OTS_8K/dehazeformer_final.json"

with open(config_path, 'r') as f:
    cfg = json.load(f)

# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- DATA --------
dataset_dir = "./datasets/OTS_8K"

train_loader = DataLoader(
    PairLoader(dataset_dir, 'train', patch_size=cfg["patch_size"]),
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=cfg["num_workers"],
    pin_memory=True
)

# -------- MODEL --------
model = dehazeformer_t().to(device)

# -------- VGG (PERCEPTUAL LOSS) --------
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# -------- OPTIMIZER --------
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

# -------- BASIC LOSS --------
l1_loss = nn.L1Loss()

# -------- EDGE LOSS --------
def edge_map(x):
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).to(x.device)
    sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).to(x.device)

    C = x.shape[1]
    sobel_x = sobel_x.expand(C,1,3,3)
    sobel_y = sobel_y.expand(C,1,3,3)

    edge_x = F.conv2d(x, sobel_x, padding=1, groups=C)
    edge_y = F.conv2d(x, sobel_y, padding=1, groups=C)

    return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

def edge_loss(x, y):
    return torch.mean(torch.abs(edge_map(x) - edge_map(y)))

# -------- PERCEPTUAL LOSS --------
def perceptual_loss(x, y):
    return torch.mean((vgg(x) - vgg(y))**2)

# -------- SSIM --------
def calc_ssim(x, y):
    return ssim(x, y, data_range=1.0)

# -------- PSNR --------
def calc_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# -------- TRAIN --------
epochs = cfg["epochs"]
best_psnr = 0

print("🚀 Training FINAL MODEL (MCA + Enhancement + Hybrid Loss)")

for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    loop = tqdm(train_loader)

    for hazy, gt in loop:
        hazy = hazy.to(device)
        gt = gt.to(device)

        output = model(hazy)

        # -------- HYBRID LOSS (BALANCED) --------
        l1 = l1_loss(output, gt)
        ssim_loss = 1 - calc_ssim(output, gt)
        perc = perceptual_loss(output, gt)
        edge = edge_loss(output, gt)

        loss = (
            0.5 * l1 +
            0.2 * ssim_loss +
            0.2 * perc +
            0.1 * edge
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr_val = calc_psnr(output, gt)
        ssim_val = calc_ssim(output, gt)

        total_loss += loss.item()
        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            psnr=f"{psnr_val.item():.2f}",
            ssim=f"{ssim_val.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    avg_psnr = total_psnr / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)

    print(f"\n📊 Epoch {epoch+1}:")
    print(f"Loss : {avg_loss:.4f}")
    print(f"PSNR : {avg_psnr:.3f}")
    print(f"SSIM : {avg_ssim:.4f}")

    os.makedirs("saved_models/final_model", exist_ok=True)

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "saved_models/final_model/best.pth")
        print("✅ Best FINAL model saved!")

print("\n🏁 Final Training Complete")