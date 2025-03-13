# inference/chexnet_model.py
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import logging
from inference.base_model import BaseModel

logger = logging.getLogger(__name__)

class CheXNetModel(BaseModel):
    """CheXNet model implementation"""
    
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CheXNet disease labels
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        
    def load(self):
        """Load the CheXNet model from checkpoint"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize DenseNet-121
            model = models.densenet121(pretrained=False)
            
            # Modify final layer to match 14 disease classes
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, len(self.labels))
            
            # Handle potential key mismatches in state dict
            state_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            model.load_state_dict(new_state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            self.model = model
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess(self, image):
        """Prepare image for model inference"""
        if isinstance(image, str):
            # If image is a file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # If image is a numpy array (from CV2)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a file path, numpy array, or PIL Image")
        
        # Store original image dimensions for visualization
        self.original_dims = image.size
        self.original_image = image
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, input_tensor):
        """Run model inference on preprocessed input"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output
    
    def postprocess(self, output, original_image=None, visualization=False):
        """Process model output into standardized format"""
        # Convert logits to probabilities using sigmoid (multi-label classification)
        probabilities = torch.sigmoid(output[0]).cpu().numpy()
        
        # Create result dictionary with all probabilities
        results = {
            "predictions": [
                {"label": self.labels[i], "probability": float(prob)} 
                for i, prob in enumerate(probabilities)
            ],
            "top_predictions": []
        }
        
        # Sort by probability and get top findings
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices[:5]:  # Top 5 findings
            if probabilities[idx] > 0.5:  # Only include reasonably confident predictions
                results["top_predictions"].append({
                    "label": self.labels[idx],
                    "probability": float(probabilities[idx])
                })
        
        # Generate visualization if requested
        if visualization and (original_image is not None or hasattr(self, 'original_image')):
            img_to_use = original_image if original_image is not None else self.original_image
            img = np.array(img_to_use)
            
            # Create an overlay with text for top predictions
            output_img = img.copy()
            y_pos = 30
            for pred in results["top_predictions"]:
                text = f"{pred['label']}: {pred['probability']:.2f}"
                cv2.putText(output_img, text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 30
                
            results["visualization"] = output_img
        
        return results
