#  DehazeFormer-Improved: Multi-Cue Attention & Enhancement-Based Image Dehazing

##  Overview

This project enhances the original **DehazeFormer-T** model by integrating:

* **Multi-Cue Attention (MCA)**
* **Illumination-aware enhancement**
* **Hybrid loss functions (L1 + SSIM + Perceptual + Edge)**

The goal is to improve both **quantitative performance (PSNR/SSIM)** and **visual quality** for single image dehazing under diverse conditions.

---

##  Methodology

We performed a structured **ablation study**:

### 🔹 1. Baseline (DehazeFormer-T)

* Standard architecture
* L1 loss
* No additional enhancements

### 🔹 2. DehazeFormer-T + MCA + L1

* Added **Multi-Cue Attention module**
* Same training setup as baseline
* Measures impact of attention mechanism

### 🔹 3. Final Model (Proposed)

* MCA integrated
* Illumination-aware enhancement
* Contrast & color refinement
* Hybrid loss:

  * L1 Loss
  * SSIM Loss
  * Perceptual Loss (VGG16)
  * Edge Loss

---

##  Results

### Training Results (OTS Dataset)

| Model       | Loss       | PSNR       | SSIM       |
| ----------- | ---------- | ---------- | ---------- |
| Baseline    | 0.0247     | 30.341     | 0.9786     |
| MCA + L1    | 0.0246     | 30.351     | 0.9786     |
| Final Model | **0.0230** | **30.478** | **0.9817** |

---

##  Key Observations

* MCA improves feature representation slightly.
* Final model significantly improves:

  * **Structural similarity (SSIM ↑)**
  * **Perceptual quality**
* Hybrid loss helps preserve:

  * Edges
  * Contrast
  * Color consistency

---

##  Sample Results

| Input      | Output         |
![0591_0 9_0 2](https://github.com/user-attachments/assets/3b34ca2d-e6c7-4c3a-83df-130138a28287)
![0636_0 95_0 2](https://github.com/user-attachments/assets/455e3850-2a41-4c63-957d-9a422d277a6b)
![CCD_4_frame_0162](https://github.com/user-attachments/assets/e98ab887-7bc0-47bc-aa7f-cef1f7ad26dd)


---

## Project Structure

```
DehazeFormer/
├── configs/
├── models/
├── utils/
├── test_images/

├── train.py
├── evaluate.py
├── test.py

├── README.md
├── requirements.txt
```

---

##  How to Run

### 🔹 1. Install Dependencies

```
pip install -r requirements.txt
```

---

### 🔹 2. Train Model

```
python train.py
```

---

### 🔹 3. Evaluate Model

```
python evaluate.py
```

---

### 🔹 4. Test on Custom Images

```
python test.py
```

---

##  Dataset

* **Training Dataset**: OTS (Outdoor Training Set)
* **Evaluation Dataset**: SOTS Indoor (RESIDE benchmark)

---

##  Contributions

*  Integrated **Multi-Cue Attention (MCA)** into DehazeFormer
*  Designed **hybrid loss function** for better perceptual quality
*  Added **illumination-aware enhancement module**
*  Conducted **systematic ablation study**
*  Improved generalization on unseen haze conditions

---

##  Future Work

* Extend to **real-world haze datasets**
* Optimize for **edge devices (embedded systems)**
* Explore **real-time dehazing applications**

---

##  License

This project follows the license provided in the original repository.

---

##  Author

**Srinidhi Vodnala**
B.Tech ECE | Embedded Systems & AI Enthusiast

---

##  Acknowledgement

Based on the original DehazeFormer architecture and improved with custom enhancements.
