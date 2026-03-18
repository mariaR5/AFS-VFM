import os
import json
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def curate_imagenet(num_images=1000, target_size=(224, 224)):
    print(f"Downloading/Loading ImageNet-1k validation split...")
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    
    # Create target directory
    save_dir = os.path.join(BASE_DIR, "data", "imagenet")
    os.makedirs(save_dir, exist_ok=True)
    
    ground_truth_map = {}
    
    print(f"Extracting {num_images} images...")
    for i, item in enumerate(dataset):
        # Stop when target number of images reached
        if i >= num_images:
            break
        
        image = item['image']
        label = item['label']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard
        image = image.resize(target_size)
        
        # Save the image
        image_filename = f"imagenet_{i:04d}.jpg"
        image_path = os.path.join(save_dir, image_filename)
        image.save(image_path)
        
        # Log the mapping
        ground_truth_map[image_filename] = label
    
    # Save ground truth dict as a json file
    labels_path = os.path.join(save_dir, "imagenet_labels.json")
    with open(labels_path, "w") as f:
        json.dump(ground_truth_map, f, indent=4)
    
    print(f"Success! {num_images} images saved to {save_dir}")
    print(f"Ground truth mapping saved to {labels_path}")
    
if __name__ == "__main__":
    curate_imagenet(1000)