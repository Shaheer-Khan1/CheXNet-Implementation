# utils/download_model.py
import os
import requests
import torch

def download_chexnet_model():
    """Downloads pre-trained CheXNet model weights"""
    # You'll need to find a valid source. This is a placeholder URL.
    url = "https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar"
    model_dir = "models/weights"
    model_path = os.path.join(model_dir, "chexnet_model.pth.tar")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading CheXNet model to {model_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download completed successfully!")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    download_chexnet_model()
