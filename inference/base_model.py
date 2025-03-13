# inference/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def load(self):
        """Load model weights"""
        pass
    
    @abstractmethod
    def preprocess(self, image):
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def predict(self, input_tensor):
        """Run prediction on preprocessed input"""
        pass
    
    @abstractmethod
    def postprocess(self, output, original_image=None, visualization=False):
        """Process model output into standardized format"""
        pass
