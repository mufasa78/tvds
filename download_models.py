#!/usr/bin/env python
"""
Model download script for Traffic Violation Detection System.
This script checks for and downloads required model files.
"""

import os
import sys
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model URLs and file paths
MODELS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8n-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt'
}

def download_file(url, filename):
    """Download a file from a URL to the specified filename."""
    try:
        logger.info(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {filename}: {str(e)}")
        return False

def check_and_download_models():
    """Check if model files exist and download them if they don't."""
    success = True
    
    for model_file, model_url in MODELS.items():
        if not os.path.exists(model_file):
            logger.info(f"{model_file} not found. Downloading...")
            if not download_file(model_url, model_file):
                success = False
        else:
            logger.info(f"{model_file} already exists.")
    
    if success:
        logger.info("All model files are available.")
    else:
        logger.warning("Some model files could not be downloaded. The application may not work correctly.")
    
    return success

if __name__ == "__main__":
    check_and_download_models()
