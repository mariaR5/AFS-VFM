"""
AFS-VFM Full Benchmark — main.py
==================================

Master orchestration script for Phase 1.

For each base image in data/coco_val/ and data/imagenet_val/:
    For each degradation type (blur, occlusion, lighting, scale, viewpoint):
        1. Generate a 20-frame degradation sequence  (Track A)
        2. Feed each frame through DINOv2, CLIP, DETR  (Track B)
        3. Record when each model's prediction diverges from its
           clean-image baseline  (is_failure = True)

Results are exported to  results/full_benchmark.csv

Usage:
    python main.py              # Full benchmark (1500 images)
    python main.py --pilot      # Quick pilot run (~20 images)
"""

import os
import sys
import csv
import time
import random
import argparse

# Force unbuffered output so progress shows up in terminal immediately
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from degradation import generate_degradation_sequence
from models.model_loader import VisionModelLoader

# ── Configuration ───────────────────────────────────────────────────────────
COCO_DIR = os.path.join(ROOT_DIR, "data", "coco_val")
IMAGENET_DIR = os.path.join(ROOT_DIR, "data", "imagenet_val")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DEGRADATION_TYPES = ["blur", "occlusion", "lighting", "scale", "viewpoint"]
NUM_FRAMES = 20

# Pilot mode: grab a small sample to validate the pipeline quickly
PILOT_COCO_COUNT = 5
PILOT_IMAGENET_COUNT = 15

# CLIP zero-shot candidate labels (mapped to common ImageNet / COCO classes)
CLIP_CANDIDATE_LABELS = [
    "a dog", "a cat", "a car", "a bicycle", "an airplane",
    "a clock", "fruit", "a person", "a coffee cup", "a traffic scene",
    "a bird", "a boat", "a horse", "a chair", "a bottle",
    "a bus", "a truck", "a train", "a flower", "a tree",
]


# ── Collect image paths ────────────────────────────────────────────────────

def collect_images(pilot=False):
    """Collect image file paths from both dataset directories."""
    images = []

    for directory, label in [(COCO_DIR, "coco"), (IMAGENET_DIR, "imagenet")]:
        if not os.path.isdir(directory):
            print(f"  WARNING: {directory} not found, skipping.")
            continue
        files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if pilot:
            count = PILOT_COCO_COUNT if label == "coco" else PILOT_IMAGENET_COUNT
            random.seed(42)
            files = random.sample(files, min(count, len(files)))
        images.extend(files)

    return images


# ── Inference helpers ───────────────────────────────────────────────────────

def infer_dinov2(model, processor, frame, device) -> str:
    """Run DINOv2 classification on a single frame and return the label."""
    inputs = processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]


def infer_clip(model, processor, frame, device, labels: list) -> str:
    """Run CLIP zero-shot classification and return the best label."""
    inputs = processor(text=labels, images=frame, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = probs.argmax(-1).item()
    return labels[best_idx]


def infer_detr(model, processor, frame, device) -> str:
    """Run DETR object detection and return the top-confidence class label."""
    inputs = processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.softmax(-1)[0, :, :-1]  # drop the 'no-object' class
    max_score, max_idx = logits.max(dim=1)
    best_det = max_score.argmax().item()
    class_id = max_idx[best_det].item()
    return model.config.id2label[class_id]


# ── Main benchmark loop ────────────────────────────────────────────────────

def run_benchmark(pilot=False):
    mode_label = "PILOT (quick validation)" if pilot else "FULL (1500 images)"
    csv_name = "pilot_benchmark.csv" if pilot else "full_benchmark.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)

    print("=" * 70)
    print(f"  AFS-VFM Benchmark — {mode_label}")
    print("=" * 70)

    # Collect images
    image_files = collect_images(pilot=pilot)
    if not image_files:
        print("\n  ERROR: No images found. Run 'python download_dataset.py' first.")
        sys.exit(1)

    total_inferences = len(image_files) * len(DEGRADATION_TYPES) * NUM_FRAMES * 3
    print(f"\n  Images:             {len(image_files)}")
    print(f"  Degradation types:  {DEGRADATION_TYPES}")
    print(f"  Frames/sequence:    {NUM_FRAMES}")
    print(f"  Total inferences:   {total_inferences}")

    # ── Load all 3 models ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Loading Vision Foundation Models ...")
    print("-" * 70)

    loader = VisionModelLoader()
    device = loader.device

    dino_model, dino_proc = loader.load_dinov2()
    clip_model, clip_proc = loader.load_clip()
    detr_model, detr_proc = loader.load_detr()

    print("  All 3 models loaded successfully.\n")

    # ── Run the benchmark ───────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    start_time = time.time()

    for img_idx, img_path in enumerate(image_files, 1):
        img_file = os.path.basename(img_path)
        dataset_source = "coco" if "coco_val" in img_path else "imagenet"

        print(f"\n{'='*70}")
        print(f"  Image [{img_idx}/{len(image_files)}]: {img_file} ({dataset_source})")
        print(f"{'='*70}")

        for deg_type in DEGRADATION_TYPES:
            print(f"\n  >> Degradation: {deg_type}")

            # Generate the 20-frame sequence (Track A)
            try:
                frames = generate_degradation_sequence(img_path, deg_type, num_frames=NUM_FRAMES)
            except Exception as exc:
                print(f"     ERROR generating sequence: {exc}")
                continue

            # Track baselines for each model
            baselines = {"dinov2": None, "clip": None, "detr": None}

            for frame_idx in range(NUM_FRAMES):
                frame = frames[frame_idx]
                severity_pct = int(frame_idx / (NUM_FRAMES - 1) * 100)

                # Convert numpy array to PIL Image for model processors
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame)
                else:
                    frame_pil = frame

                # ── DINOv2 ──
                try:
                    dino_pred = infer_dinov2(dino_model, dino_proc, frame_pil, device)
                except Exception:
                    dino_pred = "ERROR"
                if frame_idx == 0:
                    baselines["dinov2"] = dino_pred
                dino_fail = (dino_pred != baselines["dinov2"])

                # ── CLIP ──
                try:
                    clip_pred = infer_clip(clip_model, clip_proc, frame_pil, device, CLIP_CANDIDATE_LABELS)
                except Exception:
                    clip_pred = "ERROR"
                if frame_idx == 0:
                    baselines["clip"] = clip_pred
                clip_fail = (clip_pred != baselines["clip"])

                # ── DETR ──
                try:
                    detr_pred = infer_detr(detr_model, detr_proc, frame_pil, device)
                except Exception:
                    detr_pred = "ERROR"
                if frame_idx == 0:
                    baselines["detr"] = detr_pred
                detr_fail = (detr_pred != baselines["detr"])

                # Log to console
                status = ""
                if dino_fail:
                    status += " [DINOv2 FAIL]"
                if clip_fail:
                    status += " [CLIP FAIL]"
                if detr_fail:
                    status += " [DETR FAIL]"

                print(f"     Frame {frame_idx+1:02d} ({severity_pct:3d}%){status}")

                # Append result row
                all_results.append({
                    "image": img_file,
                    "dataset": dataset_source,
                    "degradation": deg_type,
                    "frame": frame_idx + 1,
                    "severity_pct": severity_pct,
                    "dinov2_prediction": dino_pred,
                    "dinov2_failure": dino_fail,
                    "clip_prediction": clip_pred,
                    "clip_failure": clip_fail,
                    "detr_prediction": detr_pred,
                    "detr_failure": detr_fail,
                })

    # ── Export CSV ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    fieldnames = [
        "image", "dataset", "degradation", "frame", "severity_pct",
        "dinov2_prediction", "dinov2_failure",
        "clip_prediction", "clip_failure",
        "detr_prediction", "detr_failure",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Total rows:       {len(all_results)}")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFS-VFM Benchmark")
    parser.add_argument("--pilot", action="store_true", help="Run a small pilot test (~20 images)")
    args = parser.parse_args()
    run_benchmark(pilot=args.pilot)
