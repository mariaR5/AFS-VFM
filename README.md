# AFS-VFM (Artificial Fail-States for Vision Foundation Models)

**Phase 1 Complete**: A scalable, cloud-agnostic benchmarking pipeline designed to find the exact degradation thresholds where state-of-the-art vision models fundamentally break.

## 🎯 Research Objective
Vision Foundation Models (VFMs) like **DINOv2**, **CLIP**, and **DETR** achieve near-perfect metrics on clean academic datasets. However, their reliability in chaotic, real-world edge cases (autonomous driving, robotics, surveillance) remains dangerously unmapped.

This project creates a deterministic **Degradation Engine** to synthetically corrupt clean images across 5 physical dimensions (Blur, Occlusion, Lighting, Scale, Viewpoint) to map the exact failure boundaries of these models.

## ⚙️ Architecture pipeline

### Track A: Degradation Engine
A specialized computer vision pipeline that applies physical-world corruptions using a deterministic, linearly increasing severity scale (0% to 100%).
1. **Motion Blur:** Simulates fast-moving camera capture
2. **Occlusion:** Randomly drops physical features 
3. **Lighting:** Progressive low-light / night conditions
4. **Scale:** Extreme frequency destruction and pixelation
5. **Viewpoint:** Extreme 3D perspective skewing

### Track B: Vision Model Inference
The pipeline dynamically loads, processes, and evaluates three distinct model architectures:
* `facebook/dinov2-base-imagenet1k-1-layer` (Self-Supervised / Classification)
* `openai/clip-vit-base-patch32` (Contrastive / Zero-Shot)
* `facebook/detr-resnet-50` (Transformer / Object Detection)

## 📊 Phase 1 Deliverable: The Benchmark Dataset
We successfully orchestrated a large-scale evaluation using **1,500 validation images** (COCO val2017 + ImageNet-1k).

* **Scale:** 1,500 images × 5 degradations × 20 frames per sequence = **150,000 total benchmark inferences**.
* **Cloud Architecture:** Implemented crash-safe, resumable batch processing (`--batch 1`, `--batch 2`) to execute seamlessly inside Kaggle's strict 12-hour free-tier GPU limits.
* **Result:** A comprehensive `full_benchmark.csv` detailing the exact frame severity at which each model predictions begin to silently fail or hallucinate.

## 🚀 How to Reproduce
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt timm datasets huggingface_hub`
3. Login to Huggingface (required for ImageNet streaming): `huggingface-cli login`
4. Run the benchmark pipeline loop (requires GPU):
   ```bash
   python main.py --batch 1  # Process images 1-500
   python main.py --batch 2  # Process images 501-1000
   python main.py --batch 3  # Process images 1001-1500
   ```
