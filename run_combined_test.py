import os
import csv
import torch
import numpy as np

# Import Aman's Engine (Track A)
from src.degradation.degradation import generate_degradation_sequence

# Import Maria's PyTorch Models (Track B)
from src.models.model_loader import VisionModelLoader

def run_combined_test(image_path: str):
    print("--- 1. Generating Aman's 20-Frame Degradation Sequence ---")
    # Calling Aman's API
    frames = generate_degradation_sequence(image_path, "blur")
    print(f"Success: Aman returned sequence of shape {frames.shape}")
    
    print("\n--- 2. Loading Maria's DINOv2 PyTorch Model ---")
    # Calling Maria's API
    loader = VisionModelLoader()
    model, processor = loader.load_dinov2()
    device = loader.device

    print("\n--- 3. Running the Inference Loop (The Exchange) ---")
    results = []
    baseline_prediction = None
    first_failure_found = False

    for i, frame in enumerate(frames):
        # Convert Aman's NumPy array slice to a format Maria's processor accepts
        inputs = processor(images=frame, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get the AI's top guess
        predicted_idx = outputs.logits.argmax(-1).item()
        prediction_label = model.config.id2label[predicted_idx]
        
        # Determine failure logic: Does the guess diverge from the clear Frame 0?
        if i == 0:
            baseline_prediction = prediction_label
            
        is_failure = (prediction_label != baseline_prediction)
        
        if is_failure and not first_failure_found:
            print(f">>> MODEL FAILED AT FRAME {i+1} (Guessed: {prediction_label}) <<<")
            first_failure_found = True

        results.append({
            "frame_index": i + 1,
            "prediction": prediction_label,
            "is_failure": is_failure
        })
        
        print(f"Frame {i+1:02d} | Prediction: {prediction_label:25s} | Failure: {is_failure}")

    print("\n--- 4. Exporting Final Outcome to CSV ---")
    csv_file = "week2_final_outcome.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "prediction", "is_failure"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Success! Check '{csv_file}' for the final milestone data.")

if __name__ == "__main__":
    # Feel free to change this to any real image path you have!
    test_image = "dummy.png" 
    
    # Generate a dummy image if one doesn't exist just for the test
    if not os.path.exists(test_image):
        import cv2
        cv2.imwrite(test_image, np.random.randint(0, 255, (256,256,3), dtype=np.uint8))
        
    run_combined_test(test_image)
