import os
import uuid
import cv2
import numpy as np
import logging
from flask import Flask, request, jsonify, send_file, render_template
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename
from inference.service import InferenceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Basic route for web interface
@app.route('/')
def index():
    return render_template('index.html')

# Create Swagger API

api = Api(app, version='1.0', title='CheXNet API',
          description='API for chest X-ray image analysis using CheXNet')

# Define namespaces
ns = api.namespace('api/v1', description='CheXNet operations')

# Define models for Swagger documentation
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type='file', required=True, help='Medical image file')
upload_parser.add_argument('task', type=str, choices=('detection', 'classification', 'segmentation'),
                          default='detection', help='Analysis task to perform')
upload_parser.add_argument('output_format', type=str, choices=('json', 'image'),
                          default='json', help='Output format')

# Initialize inference service
inference_service = InferenceService()

@ns.route('/predict')
class Predict(Resource):
    @ns.expect(upload_parser)
    def post(self):
        """Upload an image and get prediction results"""
        try:
            # Check if file is present
            if 'file' not in request.files:
                return {"error": "No file part"}, 400
                
            file = request.files['file']
            if file.filename == '':
                return {"error": "No selected file"}, 400
                
            # Get parameters
            task = request.form.get('task', 'detection')
            output_format = request.form.get('output_format', 'json')
            
            # Save uploaded file
            # api/app.py (continued)
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            logger.info(f"File saved to {file_path}")
            
            # Process the image
            results = inference_service.process_image(
                file_path, 
                task=task,
                output_format=output_format
            )
            
            # Handle errors
            if "error" in results:
                return {"error": results["error"]}, 500
                
            # Return appropriate response based on output format
            if output_format == "image" and "visualization" in results:
                # Save visualization image
                vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vis_{unique_filename}")
                cv2.imwrite(vis_path, results["visualization"])
                
                # Return image file
                return send_file(vis_path, mimetype='image/jpeg')
            else:
                # Return JSON results
                return results
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}, 500

@ns.route('/health')
class Health(Resource):
    def get(self):
        """Check if the API is healthy"""
        return {"status": "healthy"}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
