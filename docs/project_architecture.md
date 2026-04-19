# AFS-VFM: Architecture & Methodology

## 1. Project Overview

**AFS-VFM** (Anticipatory Failure Signals in Vision Foundation Models) is a research framework built to study the precise failure boundaries of state-of-the-art computer vision models. High-performing models often achieve near-perfect results on clean data but fail unpredictably in real-world environments.

Our goal is to simulate real-world conditions (dirt to a camera lens, power drops leading to underexposure) mathematically and incrementally, and feed this data into advanced Vision Foundation Models to discover exactly when their decision boundaries collapse.

---

## 2. Methodology: Dual-Track Architecture

The system is constructed with two distinct tracks communicating through a deterministic interface.

### Track A: The Degradation Engine
Written entirely in Python using NumPy and OpenCV, the degradation engine simulates severe physical corruptions.

1. **Input:** The engine accepts an RGB image.
2. **Process:** It constructs a highly controlled, deterministic sequence of 20 frames where the severity of the corruption scales linearly from `0%` to `100%`.
3. **Core Corruptions simulated:**
   * `Motion Blur`: Evaluating resilience against spatial frequency loss.
   * `Occlusion`: Testing the boundary of local vs. global feature correlation.
   * `Lighting`: Investigating robustness under extreme luminance drops.
   * `Scale`: Destroying pixel density to simulate distance artifacts.
   * `Viewpoint`: Calculating extreme perspective shear.

**Output:** A `(20, H, W, 3)` memory-safe NumPy array, ensuring proper geometric alignment across the temporal axis. Frame 0 is mathematically identical to the source image, serving as the clean validation baseline.

### Track B: Foundation Model Inference
The engine passes the tensor array to a PyTorch-based Evaluation loop containing HuggingFace transformers. We currently compare three fundamentally different vision architectures:
* **DINOv2** (Self-Supervised / Embedding-focused)
* **CLIP** (Contrastive Language-Image / Zero-Shot)
* **DETR** (ResNet-Transformer / Detection Pipeline)

The system records the model's confidence and prediction on Frame 0, then tracks the classification drift across the remaining 19 degraded frames. The exact frame where the prediction diverges from the baseline marks the failure point.

---

## 3. Data Infrastructure

Generating deep analytics requires heavy inference computation. We process 1,500 validation images from standard datasets.
* **Datasets utilized:** COCO `val2017` & ImageNet-1k `validation`
* **Volume:** 3 models × 5 degradations × 20 frames × 1,500 images = **150,000 independent inferences**.

To circumvent 12-hour hardware limits on cloud kernels (like Kaggle T4 GPUs), we constructed an autonomous fault-tolerant batch system:
* Automatic CSV checkpointing prevents data loss during OOM or kernel timeouts.
* Soft-resume functionality correctly scans existing results to continue halfway through batches.
* Native PyTorch device bridging for immediate CUDA acceleration.

---

## 4. Current Phase & Future Objectives

Phase 1 (Data Generation) is completed. Our current active step is Phase 2:
- Statistically reverse-analyzing the 150,000 row outcomes.
- Mining internal state data (specifically attention-head entropy and embedding margin drops).
- Designing a web visualization dashboard for failure patterns.
