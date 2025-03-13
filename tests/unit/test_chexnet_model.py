# tests/unit/test_chexnet_model.py
import unittest
import os
import numpy as np
from PIL import Image
import torch
from inference.chexnet_model import CheXNetModel

class TestCheXNetModel(unittest.TestCase):
    
    def setUp(self):
        # Create a mock model path
        self.model_path = os.path.join('models', 'weights', 'chexnet_model.pth.tar')
        
        # Skip tests if model file doesn't exist
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model file not found at {self.model_path}")
        
        # Initialize model
        self.model = CheXNetModel(self.model_path)
        
        # Create a dummy test image (black square)
        self.test_image = Image.new('RGB', (224, 224), color='black')
    
    def test_model_initialization(self):
        """Test that model can be initialized"""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.labels), 14)  # Should have 14 disease labels
    
    def test_model_loading(self):
        """Test model loading"""
        success = self.model.load()
        self.assertTrue(success)
        self.assertIsNotNone(self.model.model)
    
    def test_preprocessing(self):
        """Test image preprocessing"""
        # Process image path
        input_tensor = self.model.preprocess(self.test_image)
        
        # Check tensor properties
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertEqual(input_tensor.shape[0], 1)  # Batch size
        self.assertEqual(input_tensor.shape[1], 3)  # RGB channels
        self.assertEqual(input_tensor.shape[2], 224)  # Height
        self.assertEqual(input_tensor.shape[3], 224)  # Width
    
    def test_prediction(self):
        """Test model prediction"""
        # Load model
        self.model.load()
        
        # Preprocess image
        input_tensor = self.model.preprocess(self.test_image)
        
        # Run prediction
        output = self.model.predict(input_tensor)
        
        # Check output properties
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[1], len(self.model.labels))  # Should match number of classes
    
    def test_postprocessing(self):
        """Test result postprocessing"""
        # Load model
        self.model.load()
        
        # Preprocess image
        input_tensor = self.model.preprocess(self.test_image)
        
        # Run prediction
        output = self.model.predict(input_tensor)
        
        # Process results
        results = self.model.postprocess(output)
        
        # Check results structure
        self.assertIn("predictions", results)
        self.assertIn("top_predictions", results)
        self.assertEqual(len(results["predictions"]), len(self.model.labels))
        
        # Each prediction should have label and probability
        for pred in results["predictions"]:
            self.assertIn("label", pred)
            self.assertIn("probability", pred)
            self.assertIsInstance(pred["probability"], float)
            self.assertTrue(0 <= pred["probability"] <= 1)  # Probability should be between 0 and 1

if __name__ == '__main__':
    unittest.main()
