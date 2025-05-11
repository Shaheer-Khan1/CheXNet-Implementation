import os
import uuid
import cv2
import numpy as np
import logging
import requests
import json
from flask import Flask, request, jsonify, send_file, render_template, make_response, redirect, url_for
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create results directory for output JSON files
os.makedirs('results', exist_ok=True)

# Initialize model
model = None
model_path = os.getenv('MODEL_PATH', 'model.pth.tar')

# Add this class definition before the load_model function
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, index=None):
        # Forward pass
        output = self.model(x)
        
        if index is None:
            index = torch.argmax(output)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[0][index].backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Create weighted activation map
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=x.shape[2:], 
                          mode='bilinear', 
                          align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

def load_model():
    global model, gradcam
    try:
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Initialize DenseNet-121
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, 14)
        
        # Load state dictionary
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Initialize GradCAM with the last DenseNet block
        target_layer = model.features.denseblock4
        gradcam = GradCAM(model, target_layer)
        
        # Verify GradCAM initialization
        logger.info("GradCAM initialized with target layer: %s", target_layer)
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load model at startup
load_model()

# Create Swagger API
api = Api(app, 
    version='1.0', 
    title='CheXNet API',
    description='API for chest X-ray image analysis using CheXNet',
    doc='/api/docs',
    prefix='/api'
)

# Define namespaces
ns = api.namespace('v1', description='CheXNet operations')

# Define models for Swagger documentation
prediction_model = api.model('Prediction', {
    'label': fields.String(required=True, description='Disease label'),
    'probability': fields.Float(required=True, description='Prediction confidence')
})

response_model = api.model('Response', {
    'predictions': fields.List(fields.Nested(prediction_model)),
    'top_predictions': fields.List(fields.Nested(prediction_model))
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

# Define parsers for Swagger documentation
upload_parser = api.parser()
upload_parser.add_argument('file', 
    location='files', 
    type='file', 
    required=True, 
    help='Medical image file (JPEG, PNG, DICOM, TIFF, BMP)'
)
upload_parser.add_argument('callback_url',
    location='form',
    type=str,
    required=False,
    help='URL to send results to (optional)'
)

def get_image_from_file(file):
    """Process various file formats and return a PIL Image"""
    try:
        filename = file.filename.lower()
        
        # Check if file is DICOM
        if filename.endswith('.dcm'):
            # Save temporarily to read with pydicom
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.dcm")
            file.save(temp_path)
            
            try:
                import pydicom
                ds = pydicom.dcmread(temp_path)
                img_array = ds.pixel_array
                # Convert to 8-bit if needed
                if img_array.dtype != np.uint8:
                    img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                # Convert grayscale to RGB
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=2)
                image = Image.fromarray(img_array)
                os.remove(temp_path)  # Clean up
            except ImportError:
                os.remove(temp_path)
                raise Exception("DICOM support requires pydicom package")
                
        # For other image formats, use PIL directly
        else:
            image = Image.open(file).convert("RGB")
            
        return image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def analyze_image(file, callback_url=None):
    """Analyze an image and return predictions"""
    try:
        if model is None:
            return {"error": "Model not loaded"}, 500
            
        # Get the mode from the form data
        mode = request.form.get('mode', 'classification')
        logger.info(f"Analysis mode: {mode}")
            
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Process the image based on file format
        image = get_image_from_file(file)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert logits to probabilities
        probabilities = torch.sigmoid(output[0]).numpy()
        
        # Disease labels
        labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        
        # Create results
        results = {
            "predictions": [
                {"label": labels[i], "probability": float(prob)} 
                for i, prob in enumerate(probabilities)
            ],
            "top_predictions": [],
            "mode": mode  # Add mode to results
        }
        
        # Get top predictions
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices[:5]:
            if probabilities[idx] > 0.5:
                results["top_predictions"].append({
                    "label": labels[idx],
                    "probability": float(probabilities[idx])
                })
        
        # Add GradCAM visualization if in segmentation mode
        if mode == 'segmentation':
            logger.info("Generating GradCAM visualization")
            # Get the index of the highest probability class
            class_idx = np.argmax(probabilities)
            
            # Generate GradCAM
            cam = gradcam(input_tensor, class_idx)
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Convert original image to numpy array
            original_image = np.array(image)
            
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Create overlay
            alpha = 0.5
            overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', overlay)
            segmentation_base64 = base64.b64encode(buffer).decode('utf-8')
            
            results["segmentation"] = segmentation_base64
        
        # Send results to callback URL if provided
        if callback_url:
            try:
                logger.info(f"Sending results to callback URL: {callback_url}")
                response = requests.post(
                    callback_url,
                    json=results,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                logger.info(f"Callback response: {response.status_code}")
                results["callback_status"] = "sent" if response.status_code == 200 else "failed"
                results["callback_status_code"] = response.status_code
            except Exception as e:
                logger.error(f"Error sending to callback URL: {e}")
                results["callback_status"] = "failed"
                results["callback_error"] = str(e)
        
        # Save results to file for viewing later
        result_id = str(uuid.uuid4())
        result_file = os.path.join('results', f"{result_id}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        results["result_id"] = result_id
            
        return results
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"error": str(e)}, 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        callback_url = request.form.get('callback_url', None)
        view_results = request.form.get('view_results', 'false').lower() == 'true'
            
        try:
            results = analyze_image(file, callback_url)
            
            # If the user wants to view results on a new page
            if view_results and isinstance(results, dict) and 'error' not in results:
                return redirect(url_for('view_results', result_id=results.get('result_id')))
            
            return jsonify(results), 200
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return jsonify({"error": str(e)}), 500
        
    # For GET requests, check Accept header
    if 'application/json' in request.headers.get('Accept', ''):
        return jsonify({
            "message": "Send a POST request with a chest X-ray image to get predictions",
            "endpoints": {
                "predict": "/api/v1/predict",
                "health": "/api/v1/health",
                "docs": "/api/docs",
                "results": "/results/<result_id>"
            }
        }), 200
    
    # Return HTML interface for browser requests
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CheXNet - Chest X-ray Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-form {
                text-align: center;
                margin: 20px 0;
            }
            .form-group {
                margin: 15px 0;
                text-align: left;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .upload-btn {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .upload-btn:hover {
                background-color: #45a049;
            }
            .results {
                margin-top: 20px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: none;
            }
            .prediction {
                margin: 10px 0;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 4px;
            }
            .probability {
                font-weight: bold;
                color: #4CAF50;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
            }
            .error {
                color: #ff0000;
                margin: 10px 0;
                padding: 10px;
                background-color: #ffeeee;
                border-radius: 4px;
                display: none;
            }
            .info {
                background-color: #e8f4ff;
                padding: 10px;
                border-radius: 4px;
                margin: 15px 0;
            }
            .mode-selector {
                margin: 15px 0;
                text-align: left;
            }
            .mode-selector label {
                display: inline-block;
                margin-right: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CheXNet - Chest X-ray Analysis</h1>
            <div class="info">
                <p>Supported image formats: JPEG, PNG, DICOM (.dcm), TIFF, BMP</p>
            </div>
            <div class="upload-form">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="fileInput">Upload X-ray Image:</label>
                        <input id="fileInput" type="file" name="file" accept="image/jpeg,image/png,image/tiff,image/bmp,application/dicom" required>
                    </div>
                    
                    <div class="mode-selector">
                        <label>Analysis Mode:</label>
                        <label>
                            <input type="radio" name="mode" value="classification" checked> Classification
                        </label>
                        <label>
                            <input type="radio" name="mode" value="segmentation"> Segmentation
                        </label>
                    </div>
                    
                    <div class="form-group">
                        <label for="callbackUrl">Callback URL (Optional):</label>
                        <input id="callbackUrl" type="text" name="callback_url" placeholder="https://your-server.com/webhook">
                        <small>If provided, analysis results will be sent to this URL</small>
                    </div>
                    
                    <div class="form-group">
                        <label for="viewResults">
                            <input id="viewResults" type="checkbox" name="view_results" value="true">
                            View results on a new page
                        </label>
                    </div>
                    
                    <button type="submit" class="upload-btn">Analyze Image</button>
                </form>
            </div>
            <div id="error" class="error"></div>
            <div id="results" class="results">
                <h2>Analysis Results</h2>
                <div id="predictions"></div>
                <h3>Raw JSON Response</h3>
                <pre id="jsonResponse"></pre>
            </div>
        </div>
        <script>
        // Add this at the top of the script
        const labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ];

        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const viewResults = document.getElementById('viewResults').checked;
            const mode = document.querySelector('input[name="mode"]:checked').value;
            formData.append('mode', mode);
            
            console.log('Selected mode:', mode);  // Debug log
            
            // Reset display
            document.getElementById('error').style.display = 'none';
            document.getElementById('results').style.display = 'none';

            if (viewResults) {
                // Use traditional form submission to get a page redirect
                const form = document.getElementById('uploadForm');
                form.method = 'POST';
                form.action = '/';
                form.submit();
            } else {
                // Use fetch and show results on same page
                fetch('/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.error || 'Error analyzing image');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Received data:', data);  // Debug log
                    const resultsDiv = document.getElementById('results');
                    const predictionsDiv = document.getElementById('predictions');
                    const jsonResponse = document.getElementById('jsonResponse');
                    
                    resultsDiv.style.display = 'block';
                    predictionsDiv.innerHTML = '';
                    
                    // Display classification results
                    if (data.classification) {
                        data.classification.forEach((prob, index) => {
                            if (prob > 0.5) {
                                const predDiv = document.createElement('div');
                                predDiv.className = 'prediction';
                                predDiv.innerHTML = `
                                    <strong>${labels[index]}</strong>
                                    <div class="probability">${(prob * 100).toFixed(2)}%</div>
                                `;
                                predictionsDiv.appendChild(predDiv);
                            }
                        });
                    }
                    
                    // Display segmentation overlay if in segmentation mode
                    if (data.mode === 'segmentation' && data.segmentation) {
                        const segDiv = document.createElement('div');
                        segDiv.className = 'segmentation';
                        segDiv.innerHTML = `
                            <h3>Segmentation Overlay</h3>
                            <div class="segmentation-container">
                                <img src="data:image/png;base64,${data.segmentation}" 
                                     alt="Segmentation Overlay" 
                                     style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;">
                            </div>
                        `;
                        resultsDiv.appendChild(segDiv);
                    }
                    
                    // Display raw JSON
                    jsonResponse.textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    console.error('Error:', error);
                    const errorDiv = document.getElementById('error');
                    errorDiv.textContent = error.message;
                    errorDiv.style.display = 'block';
                });
            }
        });
        </script>
    </body>
    </html>
    ''', 200

@app.route('/results/<result_id>')
def view_results(result_id):
    """Display results on a dedicated page"""
    try:
        # Sanitize the result_id to prevent path traversal
        safe_result_id = secure_filename(result_id)
        result_file = os.path.join('results', f"{safe_result_id}.json")
        
        if not os.path.exists(result_file):
            return render_template('error.html', error="Results not found"), 404
        
        with open(result_file, 'r') as f:
            results = json.load(f)
            
        return render_template('results.html', results=results)
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        return render_template('error.html', error=str(e)), 500

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Create results.html template
with open('templates/results.html', 'w') as f:
    f.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CheXNet - Analysis Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2 {
                color: #333;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .home-btn {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                text-decoration: none;
            }
            .prediction {
                margin: 10px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 4px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .probability {
                font-weight: bold;
                color: #4CAF50;
            }
            .progress-container {
                flex-grow: 1;
                margin: 0 20px;
                background-color: #eee;
                border-radius: 4px;
                height: 10px;
            }
            .progress-bar {
                height: 10px;
                background-color: #4CAF50;
                border-radius: 4px;
            }
            .json-container {
                margin-top: 30px;
                padding: 20px;
                background-color: #f5f5f5;
                border-radius: 8px;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                margin: 0;
            }
            .download-btn {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>X-ray Analysis Results</h1>
                <a href="/" class="home-btn">Back to Home</a>
            </div>
            
            <h2>Findings</h2>
            {% if results.top_predictions and results.top_predictions|length > 0 %}
                {% for pred in results.top_predictions %}
                <div class="prediction">
                    <strong>{{ pred.label }}</strong>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {{ pred.probability * 100 }}%;"></div>
                    </div>
                    <div class="probability">{{ "%.2f"|format(pred.probability * 100) }}%</div>
                </div>
                {% endfor %}
            {% else %}
                <p>No significant findings detected.</p>
            {% endif %}
            
            <div class="json-container">
                <h2>Complete Analysis Data</h2>
                <pre id="jsonData">{{ results|tojson(indent=2) }}</pre>
                <button class="download-btn" onclick="downloadJSON()">Download JSON</button>
            </div>
        </div>
        
        <script>
            function downloadJSON() {
                const data = JSON.parse(document.getElementById('jsonData').textContent);
                const dataStr = JSON.stringify(data, null, 2);
                const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
                
                const exportElem = document.createElement('a');
                exportElem.setAttribute('href', dataUri);
                exportElem.setAttribute('download', 'chexnet-results.json');
                document.body.appendChild(exportElem);
                exportElem.click();
                document.body.removeChild(exportElem);
            }
        </script>
    </body>
    </html>
    ''')

# Create error.html template
with open('templates/error.html', 'w') as f:
    f.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CheXNet - Error</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 {
                color: #cc0000;
            }
            .error-message {
                padding: 20px;
                background-color: #ffeeee;
                border-radius: 8px;
                margin: 20px 0;
            }
            .home-btn {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Error</h1>
            <div class="error-message">
                {{ error }}
            </div>
            <a href="/" class="home-btn">Back to Home</a>
        </div>
    </body>
    </html>
    ''')

@ns.route('/predict')
class Predict(Resource):
    @ns.expect(upload_parser)
    @ns.response(200, 'Success', response_model)
    @ns.response(400, 'Bad Request', error_model)
    @ns.response(500, 'Internal Server Error', error_model)
    def post(self):
        """Upload an image and get prediction results"""
        if 'file' not in request.files:
            return {"error": "No file part"}, 400
            
        file = request.files['file']
        if file.filename == '':
            return {"error": "No selected file"}, 400
            
        callback_url = request.form.get('callback_url', None)
        return analyze_image(file, callback_url)

@ns.route('/health')
class Health(Resource):
    @ns.response(200, 'API is healthy')
    def get(self):
        """Check if the API is healthy"""
        return {"status": "healthy"}

def create_segmentation_visualization(image, segmentation_map):
    """Create a visualization of the segmentation overlay"""
    # Convert segmentation map to uint8
    seg_map = (segmentation_map * 255).astype(np.uint8)
    
    # Create color overlay
    overlay = np.zeros_like(image)
    overlay[..., 0] = seg_map  # Red channel
    
    # Blend original image with overlay
    alpha = 0.5
    visualization = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    return visualization

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    mode = request.form.get('mode', 'classification')
    
    # Add debug logging
    logger.info(f"Received request with mode: {mode}")
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Get predictions
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.sigmoid(output).cpu().numpy()
        
        # Get GradCAM visualization if in segmentation mode
        if mode == 'segmentation':
            logger.info("Generating GradCAM visualization")
            # Get the index of the highest probability class
            class_idx = np.argmax(probabilities)
            logger.info(f"Using class index: {class_idx}")
            
            # Generate GradCAM
            cam = gradcam(image_tensor, class_idx)
            logger.info("GradCAM generated successfully")
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Convert original image to numpy array
            original_image = np.array(image)
            
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Create overlay
            alpha = 0.5
            overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', overlay)
            segmentation_base64 = base64.b64encode(buffer).decode('utf-8')
            
            logger.info("Segmentation visualization completed")
            return jsonify({
                'mode': 'segmentation',
                'classification': probabilities.tolist(),
                'segmentation': segmentation_base64
            })
        else:
            logger.info("Returning classification results only")
            return jsonify({
                'mode': 'classification',
                'classification': probabilities.tolist()
            })
                
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)