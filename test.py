import torch
import cv2
import os
import numpy as np
from models import dehazeformer_t

# =====================
# CONFIG
# =====================
MODEL_PATH = "saved_models/mca_ots/best.pth"
INPUT_DIR = "test_images"
BASE_OUTPUT_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# CREATE VERSIONED FOLDER
# =====================
def get_next_test_folder(base_dir):
    i = 1
    while True:
        folder = os.path.join(base_dir, f"test{i}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        i += 1

OUTPUT_DIR = get_next_test_folder(BASE_OUTPUT_DIR)
print(f"📁 Saving results in: {OUTPUT_DIR}")

# =====================
# LOAD MODEL
# =====================
model = dehazeformer_t().to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint, strict=False)
model.eval()

print("✅ Model Loaded")

# =====================
# INFERENCE
# =====================
for img_name in os.listdir(INPUT_DIR):

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Skipping {img_name}")
        continue

    # ---- Normalize ----
    img_input = img.astype(np.float32) / 255.0
    img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # ---- Inference ----
    with torch.no_grad():
        output = model(img_input)

    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0, 1)
    output_img = (output * 255).astype(np.uint8)
    

    # =====================
    # SIDE-BY-SIDE COMPARISON
    # =====================
    h, w, _ = img.shape
    output_img = cv2.resize(output_img, (w, h))

    comparison = np.hstack((img, output_img))

    # Optional labels
    #cv2.putText(comparison, "Input", (20, 40),
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #cv2.putText(comparison, "Enhanced", (w+20, 40),
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # =====================
    # SAVE
    # =====================
    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, comparison)

    print(f"✅ Saved: {img_name}")

print("🎉 Testing Complete")