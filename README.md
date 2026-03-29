# AFS-VFM: Artificial Fail-States for Vision Foundation Models

> **Status:** 🚧 Work in Progress (~40% Finished) 
> *Currently building the core dataset. Finding out exactly when the biggest AI vision models break.*

## The Core Idea

People use huge AI vision models (like DINOv2, CLIP, and DETR) for everything from self-driving cars to medical imaging. On paper, these models score close to 100% on clean benchmark datasets. But the real world is messy. 

What happens to a self-driving car's vision when a camera lens gets partially blocked by dirt? Or when it suddenly gets dark? At what exact point does the AI stop "seeing" correctly and start guessing?

**AFS-VFM** is a project built to answer that. I am building a custom pipeline that takes clean images, simulates real-world physical degradations (like motion blur or lighting drops) on a sliding scale from 1% to 100%, and tests exactly when these massive models fail.

## How It Works

Right now, Phase 1 (Data Generation) is fully complete. The pipeline works in two main tracks:

### 1. The Degradation Engine
I built a python engine from scratch that takes a normal image and applies one of 5 corruptions, smoothly increasing the intensity across 20 frames:
- 🌧️ **Motion Blur:** Mimicking fast-moving cameras
- ⬛ **Occlusion:** Dropping random black boxes over important features
- 🌑 **Lighting:** Progressively plunging the image into darkness
- 📉 **Scale:** Destroying image quality via extreme pixelation
- 📐 **Viewpoint:** Warping the 3D perspective

### 2. Model Inference
The engine then feeds these corrupted frames into three very different types of vision models to compare how they react:
* `DINOv2` (Classification - Self-Supervised)
* `CLIP` (Zero-Shot Reasoning)
* `DETR` (Object Detection)

## What We've Built So Far

Phase 1 was all about infrastructure and generating the benchmark data. 
To get around the strict 12-hour free-tier GPU limits on Kaggle, I engineered a highly resilient, auto-resuming batch processing script. 

It worked perfectly: we successfully processed 1,500 validation images (from COCO and ImageNet) across 5 degradations and generated exactly **150,000 extreme edge-case model inferences**. 

## What's Next?
The project is currently about 40% done. 

Phase 1 solved the massive infrastructure and data generation problem. Now that we have the 150,000-row CSV dataset showing exactly where these models fracture under physical stress, the real fun begins. 

The next step is to dive into the data, mathematically figure out the exact breaking points for each model, and build visual dashboards to showcase the results. 

*Stay tuned for the analysis phase...*
