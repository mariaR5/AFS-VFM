"""
Dataset Downloader — AFS-VFM Research Project
===============================================

Downloads the two standard benchmark datasets required for Phase 1:
  1. COCO val2017      (~500 images sampled) — direct from cocodataset.org
  2. ImageNet-1k val   (~1000 images sampled) — via HuggingFace datasets

Usage:
    python download_dataset.py
"""

import os
import sys
import random
import shutil
import zipfile
import urllib.request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
COCO_DIR = os.path.join(DATA_DIR, "coco_val")
IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet_val")

COCO_SAMPLE_COUNT = 500
IMAGENET_SAMPLE_COUNT = 1000

random.seed(42)  # Reproducible sampling


# ── COCO val2017 (direct download, no auth) ─────────────────────────────────

COCO_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ZIP_PATH = os.path.join(DATA_DIR, "val2017.zip")


def download_coco():
    """Download COCO val2017, extract, and sample 500 images."""
    if os.path.isdir(COCO_DIR) and len(os.listdir(COCO_DIR)) >= COCO_SAMPLE_COUNT:
        print(f"  [SKIP] COCO already has {len(os.listdir(COCO_DIR))} images")
        return True

    os.makedirs(DATA_DIR, exist_ok=True)

    # Download the zip
    if not os.path.exists(COCO_ZIP_PATH):
        print(f"  Downloading COCO val2017 (~1 GB) ...")
        print(f"  URL: {COCO_ZIP_URL}")
        print(f"  This may take a few minutes depending on your internet speed.")
        try:
            req = urllib.request.Request(COCO_ZIP_URL, headers={
                "User-Agent": "AFS-VFM-Research/1.0"
            })
            with urllib.request.urlopen(req) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(COCO_ZIP_PATH, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            mb = downloaded / (1024 * 1024)
                            print(f"\r  Progress: {mb:.0f} MB ({pct}%)", end="", flush=True)
            print()  # newline after progress
        except Exception as exc:
            print(f"\n  FAILED to download COCO: {exc}")
            return False
    else:
        print(f"  [SKIP] COCO zip already downloaded")

    # Extract the zip
    print(f"  Extracting COCO val2017 ...")
    extract_dir = os.path.join(DATA_DIR, "val2017_full")
    try:
        with zipfile.ZipFile(COCO_ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        # The zip extracts to DATA_DIR/val2017/
        extracted = os.path.join(DATA_DIR, "val2017")
        if os.path.isdir(extracted):
            os.rename(extracted, extract_dir)
    except Exception as exc:
        print(f"  FAILED to extract: {exc}")
        return False

    # Sample 500 random images
    print(f"  Sampling {COCO_SAMPLE_COUNT} images ...")
    all_images = [f for f in os.listdir(extract_dir) if f.endswith(".jpg")]
    sampled = random.sample(all_images, min(COCO_SAMPLE_COUNT, len(all_images)))

    os.makedirs(COCO_DIR, exist_ok=True)
    for img in sampled:
        shutil.copy2(os.path.join(extract_dir, img), os.path.join(COCO_DIR, img))

    print(f"  COCO: {len(sampled)} images saved to {COCO_DIR}")

    # Clean up the full extraction and zip to save disk space
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(COCO_ZIP_PATH):
        os.remove(COCO_ZIP_PATH)
        print(f"  Cleaned up zip and temp files to save space")

    return True


# ── ImageNet-1k validation (via HuggingFace datasets) ───────────────────────

def download_imagenet():
    """Download ImageNet-1k validation split and sample 1000 images."""
    if os.path.isdir(IMAGENET_DIR) and len(os.listdir(IMAGENET_DIR)) >= IMAGENET_SAMPLE_COUNT:
        print(f"  [SKIP] ImageNet already has {len(os.listdir(IMAGENET_DIR))} images")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed.")
        print("  Run: pip install datasets huggingface_hub")
        return False

    print(f"  Loading ImageNet-1k validation split from HuggingFace ...")
    print(f"  (First run will download ~6 GB — subsequent runs use cache)")

    try:
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,  # <--- THIS FIXES THE NETWORK CRASH
        )
    except Exception as exc:
        error_msg = str(exc)
        if "gated" in error_msg.lower() or "access" in error_msg.lower() or "401" in error_msg:
            print(f"\n  ACCESS DENIED — You need to accept the ImageNet license first:")
            print(f"  1. Go to: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            print(f"  2. Click 'Agree and access repository'")
            print(f"  3. Then run: huggingface-cli login")
            print(f"  4. Re-run this script")
        else:
            print(f"\n  FAILED to load ImageNet: {exc}")
        return False

    os.makedirs(IMAGENET_DIR, exist_ok=True)
    print(f"  Streaming {IMAGENET_SAMPLE_COUNT} images (this is much faster and more stable) ...")

    count = 0
    for sample in dataset:
        count += 1
        if count > IMAGENET_SAMPLE_COUNT:
            break
            
        image = sample["image"]  # PIL Image
        label = sample["label"]

        # Convert grayscale to RGB to avoid OpenCV shape errors later
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"imagenet_{count:06d}_class{label}.jpg"
        filepath = os.path.join(IMAGENET_DIR, filename)
        image.save(filepath)

        if count % 100 == 0:
            print(f"    Saved {count}/{IMAGENET_SAMPLE_COUNT} ...")

    print(f"  ImageNet: {IMAGENET_SAMPLE_COUNT} images saved to {IMAGENET_DIR}")
    return True



# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AFS-VFM Dataset Downloader")
    print("=" * 60)

    print("\n--- COCO val2017 ---")
    coco_ok = download_coco()

    print("\n--- ImageNet-1k validation ---")
    imagenet_ok = download_imagenet()

    print("\n" + "=" * 60)
    print(f"  COCO:     {'OK' if coco_ok else 'FAILED'}")
    print(f"  ImageNet: {'OK' if imagenet_ok else 'FAILED'}")

    if coco_ok and imagenet_ok:
        coco_count = len(os.listdir(COCO_DIR))
        inet_count = len(os.listdir(IMAGENET_DIR))
        print(f"\n  Total base images: {coco_count + inet_count}")
        print(f"    COCO:     {coco_count} images in {COCO_DIR}")
        print(f"    ImageNet: {inet_count} images in {IMAGENET_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
