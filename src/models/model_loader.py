import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel,
    DetrImageProcessor,
    DetrForObjectDetection
)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class VisionModelLoader:
    def __init__(self):
        self.device = get_device()
        print(f"Loading models onto device: {self.device}")
    
    def load_dinov2(self):
        print(f"Loading DINOv2...")
        model_id = "facebook/dinov2-base-imagenet1k-1-layer"
        
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        
        model.to(self.device)
        model.eval()
        
        return model, processor
    
    def load_clip(self):
        print("Loading CLIP...")
        model_id = "openai/clip-vit-base-patch32"
        
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        
        model.to(self.device)
        model.eval()
        
        return model, processor

    def load_detr(self):
        print("Loading DETR...")
        model_id = "facebook/detr-resnet-50"
        
        processor = DetrImageProcessor.from_pretrained(model_id)
        model = DetrForObjectDetection.from_pretrained(model_id)
        
        model.to(self.device)
        model.eval()
        
        return model, processor
    
if __name__ == "__main__":
    loader = VisionModelLoader()
    
    dino_model, dino_processor = loader.load_dinov2()
    print(f"DINOv2 successfully loaded and set to eval mode")
