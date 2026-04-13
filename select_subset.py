import os
import random
import shutil

# Paths
hazy_dir = "datasets/OTS/hazy"
gt_dir = "datasets/OTS/GT"

out_hazy = "datasets/OTS_10K/train/hazy"
out_gt = "datasets/OTS_10K/train/GT"

os.makedirs(out_hazy, exist_ok=True)
os.makedirs(out_gt, exist_ok=True)

# Get all hazy images
hazy_images = os.listdir(hazy_dir)

# Randomly pick 10K
selected = random.sample(hazy_images, 10000)

for img in selected:
    shutil.copy(os.path.join(hazy_dir, img), os.path.join(out_hazy, img))

    # Extract GT name
    gt_name = img.split('_')[0] + ".png"

    if os.path.exists(os.path.join(gt_dir, gt_name)):
        shutil.copy(os.path.join(gt_dir, gt_name), os.path.join(out_gt, gt_name))

print("✅ 10K subset created!")