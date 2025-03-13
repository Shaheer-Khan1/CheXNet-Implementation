# inference/service.py
import os
import json
import logging
from inference.chexnet_model import CheXNetModel

logger = logging.getLogger(__name__)

class InferenceService:
    """Service for running medical image inference"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join('models', 'weights', 'chexnet_model.pth.tar')
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize the model"""
        try:
            logger.info("Initializing inference service")
            self.model = CheXNetModel(self.model_path)
            success = self.model.load()
            if not success:
                logger.error("Failed to load model")
            return success
        except Exception as e:
            logger.error(f"Error initializing inference service: {e}")
            return False
    
    def process_image(self, image_path, task="detection", output_format="json"):
        """
        Process an image through the inference pipeline
        
        Args:
            image_path: Path to the input image
            task: Type of task ("detection", "classification", "segmentation")
            output_format: Format of the output ("json" or "image")
            
        Returns:
            Dictionary with results or processed image
        """
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            if task not in ["detection", "classification", "segmentation"]:
                return {"error": f"Unsupported task: {task}"}
            
            if output_format not in ["json", "image"]:
                return {"error": f"Unsupported output format: {output_format}"}
            
            # Process the image
            input_tensor = self.model.preprocess(image_path)
            output = self.model.predict(input_tensor)
            
            # Generate appropriate output based on format
            visualization = (output_format == "image")
            results = self.model.postprocess(output, visualization=visualization)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
