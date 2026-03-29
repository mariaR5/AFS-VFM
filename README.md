# 👁️ AFS-VFM: Artificial Fail-States for Vision Foundation Models

> **Status:** 🚧 Work in Progress (Phase 1 Complete / ~40% Finished) 
> *Actively researching the physical failure thresholds of state-of-the-art vision models.*

## 🎯 The Vision
Modern Vision Foundation Models (VFMs) like **DINOv2**, **CLIP**, and **DETR** report near-perfect accuracy on clean academic datasets. However, what happens when these models are deployed in autonomous vehicles, robotics, or medical imaging and encounter chaotic, real-world edge cases?

**AFS-VFM** answers this question. This project is a rigorous, custom-built research pipeline designed to synthetically generate physical-world corruptions and mathematically prove the exact boundaries where state-of-the-art models hallucinate, diverge, or completely fail.

## 🧠 Core Architecture

Our architecture is split into targeted phases. Currently, the **Degradation Engine** and **Data Generation (Phase 1)** are fully complete and operational.

### 1. The Degradation Engine
A deterministic computer vision pipeline that corrupts clean images across 5 physical dimensions using a strict, linearly increasing severity scale (0% to 100%):
- 🌧️ **Motion Blur:** Simulates fast-moving camera capture
- ⬛ **Occlusion:** Randomly drops physical features and structures
- 🌑 **Lighting:** Progressive low-light / nighttime conditions
- 📉 **Scale:** Extreme frequency destruction and aggressive pixelation
- 📐 **Viewpoint:** Extreme 3D perspective skewing and shear

### 2. Multi-Architecture Inference
The engine dynamically evaluates corrupted image sequences across three fundamentally different model architectures to expose their unique vulnerabilities:
* `DINOv2` (Self-Supervised / Attention-based)
* `CLIP` (Contrastive / Zero-Shot Reasoning)
* `DETR` (Transformer / Object Detection)

## 📊 Current Progress & Achievements

**Phase 1 is completely finished.** We successfully tackled large-scale cloud GPU orchestration to generate a massive, foundational dataset:
* Processed **1,500 validation images** (COCO val2017 + ImageNet-1k).
* Handled complex cloud infrastructure by engineering a crash-safe, resumable batching system to bypass strict 12-hour free-tier GPU limits.
* Successfully generated exactly **150,000 extreme edge-case model inferences**.

## 🚀 What's Next? (The Roadmap)
The project is currently ~40% complete. We have a long and exciting way to go before final publication:

- [x] **Phase 1: Data Infrastructure:** Build the degradation engine and generate the 150,000-row benchmark dataset.
- [ ] **Phase 2: Threshold Analysis:** Parse the generated CSV data to identify at exactly what severity percentage each model architecture fundamentally breaks.
- [ ] **Phase 3: Visual Dashboards:** Build compelling UI/data-visualizations to map exactly how model confidence decays under physical stress.
- [ ] **Phase 4: Research Publication:** Synthesize findings into an academic technical paper detailing VFM reliability for edge-case deployment.
