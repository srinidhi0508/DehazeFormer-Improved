import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loader import PairLoader
from models import dehazeformer_t
from pytorch_msssim import ssim

# -------- DEVICE --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- DATASET --------
dataset_dir = "./datasets/SOTS/outdoor"

test_loader = DataLoader(
    PairLoader(dataset_dir, mode='test', patch_size=None),
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# -------- MODEL (WITH MCA) --------
model = dehazeformer_t().to(device)

# 🔥 IMPORTANT: load MCA weights
model_path = "saved_models/final_model/best.pth"
checkpoint = torch.load(model_path, map_location=device)

model.load_state_dict(checkpoint)   # should NOT give error now

model.eval()
print("✅ Loaded final model")

# -------- METRICS --------
def calc_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_ssim(x, y):
    return ssim(x, y, data_range=1.0)

# -------- EVALUATION --------
total_psnr = 0
total_ssim = 0

print("\n🚀 Evaluating final model on SOTS outdoor...\n")

loop = tqdm(test_loader)

with torch.no_grad():
    for hazy, gt in loop:
        hazy = hazy.to(device)
        gt = gt.to(device)

        output = model(hazy)

        # ensure valid output range
        output = torch.clamp(output, 0, 1)

        psnr_val = calc_psnr(output, gt)
        ssim_val = calc_ssim(output, gt)

        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()

        loop.set_postfix(
            psnr=f"{psnr_val.item():.2f}",
            ssim=f"{ssim_val.item():.4f}"
        )

# -------- FINAL RESULTS --------
avg_psnr = total_psnr / len(test_loader)
avg_ssim = total_ssim / len(test_loader)

print("\n📊 ===== final RESULTS =====")
print(f"PSNR: {avg_psnr:.3f}")
print(f"SSIM: {avg_ssim:.4f}")