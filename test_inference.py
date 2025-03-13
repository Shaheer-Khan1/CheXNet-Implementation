# test_inference.py
import os
import cv2
from inference.service import InferenceService

def main():
    # Create inference service
    service = InferenceService()
    
    # Specify path to a test image (you'll need to have one)
    test_image = "chest.jpeg"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Please provide a valid image.")
        return
    
    # Process in JSON format
    print("Processing image for detection task with JSON output...")
    result_json = service.process_image(test_image, task="detection", output_format="json")
    print("Results:")
    for pred in result_json.get("top_predictions", []):
        print(f"- {pred['label']}: {pred['probability']:.4f}")
    
    # Process with image visualization
    print("\nProcessing image for detection task with image output...")
    result_img = service.process_image(test_image, task="detection", output_format="image")
    
    # Save visualization if available
    if "visualization" in result_img:
        output_path = "output_visualization.jpg"
        cv2.imwrite(output_path, result_img["visualization"])
        print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
