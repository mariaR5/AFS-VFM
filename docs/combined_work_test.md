# Combined Work Test: Milestone 2 Analysis
**Authors:** Aman (Lead) & Maria (Co-Author)
**Phase:** Phase 1 - Week 2 (Single Feature Test)
**Degradation Evaluated:** Motion Blur
**Inference Model:** Meta DINOv2 (facebook/dinov2-base-imagenet1k-1-layer)

---

## 1. Architectural Flowchart (The Exchange)
Below is the logical execution path detailing how Aman’s and Maria’s individual codebases interacted during the integration test:

```mermaid
graph TD
    A([Input: Clear Image]) --> B[Aman's Degradation Engine]
    
    subgraph Track A: Degradation Generation (Aman)
    B -->|degradation.py| C(generate_degradation_sequence)
    C -->|transformations.py| D[Apply 20-Step Motion Blur]
    D --> E[(NumPy Tensor: 20xHxWx3)]
    end
    
    E -->|Handshake Protocol| F[Maria's Inference Loop]
    
    subgraph Track B: Failure Detection (Maria)
    F -->|model_loader.py| G(Load DINOv2 Weights)
    G --> H[Run Evaluation on Frame N]
    H -->|Frame == 0?| I[Establish Baseline Guess]
    H -->|Frame > 0?| J{Prediction == Baseline?}
    J -->|Yes| K[is_failure = False]
    J -->|No| L[is_failure = True]
    end
    
    K --> M[(Export CSV)]
    L --> M[(Export CSV)]
```

---

## 2. Division of Logic & Responsibilities

### Aman's Code (The Methodology)
*   **The Engine (`degradation.py`):** Structured the primary `DegradationPipeline` class to guarantee an API strict output of 20 mathematical frames, mapping directly to `(20, H, W, 3)`.
*   **The Math (`transformations.py`):** Executed the specific mathematical smearing logic (Motion Blur kernels) required for Week 2, aggressively stretching the pixels horizontally.
*   **Purpose:** The engine acts as the "adversary," actively working to systematically break the visual integrity of the input image up to severe extremes.

### Maria's Code (The Pipeline)
*   **The Loader (`model_loader.py`):** Instantiates the state-of-the-art DINOv2 architecture directly from HuggingFace via `AutoImageProcessor` and PyTorch.
*   **The Logic (`run_combined_test.py` loop):** Continuously feeds Aman's NumPy array into the PyTorch tensor pipeline. It mathematically evaluates the AI's highest-confidence prediction array `outputs.logits.argmax(-1)`.
*   **Purpose:** Tracks exactly *when* the model's spatial reasoning breaks, verifying the effectiveness of Aman's engine by recording the failure drop-off.

---

## 3. How and Why It Failed (The Results)
In our automated run, the pure-static dummy image was passed through the pipeline:
*   **Frame 1 (Prediction: `analog clock`):** The image had minimal blur. The model recognized the static patterns as a clock dial. `Failure: False`
*   **Frame 2 (Prediction: `chain mail`):** Just one frame into Aman's degradation, the horizontal smearing destroyed the sharp spatial frequencies. The AI's mathematical confidence shattered, misinterpreting the noise as metal links. **`Failure: True`**
*   **Frame 6 (Prediction: `window screen`):** The blur deepened into heavy lines.
*   **Frame 9 to 20 (Prediction: `ruler`):** Maximum intensity. The image became a complete horizontal smear. The AI permanently defaulted to guessing a straight ruler.

**Conclusion:** The breakdown proves that state-of-the-art vision models (DINOv2) rely heavily on distinct, high-frequency edges for classification. By removing these via Motion Blur (Aman's engine), the model fails rapidly (Maria's logic tracked failure precisely at 5% degradation progression).

---

## 4. Technical Problems & Environment Resolutions
During the final combination, we encountered multiple severe environment clashes on the test benchmark:
1.  **Missing Architecture Dependency:** Maria's code required `transformers`, which was missing from `requirements.txt`.
    *   *Fix:* Ran `pip install transformers` to fetch the HuggingFace loaders.
2.  **Protobuf Collision:** The `transformers` module triggered a `runtime_version` protocol buffer crash when attempting to load Google architecture dependencies.
    *   *Fix:* Ran `pip install --upgrade protobuf` to update the C-bindings to v4.
3.  **NumPy ABI Mismatch:** `scikit-learn` threw `Expected 96 from C header, got 88` because the underlying Python numeric stack was fundamentally out of sync due to TensorFlow.
    *   *Fix:* Aggressively uninstalled the conflicting `tensorflow` and `keras` packages, then forced a binary sync with `pip install --upgrade numpy scikit-learn`. This completely unbricked the PyTorch pipeline.
