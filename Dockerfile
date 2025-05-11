# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create uploads directory
RUN mkdir -p /app/uploads

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .
COPY gradcam.py .
COPY model.pth.tar .

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV MODEL_PATH=/app/model.pth.tar
ENV PYTHONWARNINGS="ignore::UserWarning"

# Run app.py when the container launches
CMD ["python", "app.py"] 