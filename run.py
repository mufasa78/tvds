#!/usr/bin/env python
"""
Run script for Traffic Violation Detection System.
This script ensures all necessary components are initialized before running the app.
"""

import os
import sys
import subprocess
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_models():
    """Check if model files exist."""
    required_models = ['yolov8n.pt', 'yolov8n-seg.pt']
    missing_models = [model for model in required_models if not os.path.exists(model)]
    
    if missing_models:
        logger.warning(f"Missing model files: {', '.join(missing_models)}")
        logger.info("Running download_models.py to get missing models...")
        
        # Check if download_models.py exists
        if os.path.exists('download_models.py'):
            try:
                # Import and run the download_models module
                spec = importlib.util.spec_from_file_location("download_models", "download_models.py")
                download_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(download_models)
                download_models.check_and_download_models()
            except Exception as e:
                logger.error(f"Error downloading models: {str(e)}")
                return False
        else:
            logger.error("download_models.py not found. Cannot download missing models.")
            return False
    
    return True

def check_database():
    """Check if database is initialized."""
    try:
        # Import the app and models
        from app import app, db
        import models
        
        with app.app_context():
            # Check if tables exist by querying one of them
            try:
                models.Analysis.query.first()
                logger.info("Database is initialized.")
                return True
            except Exception:
                logger.warning("Database tables not found. Initializing database...")
                
                # Initialize database
                db.create_all()
                logger.info("Database initialized successfully.")
                return True
                
    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return False

def run_app():
    """Run the Flask application."""
    logger.info("Starting Traffic Violation Detection System...")
    
    # Check models and database before running
    if not check_models():
        logger.error("Failed to ensure model files are available.")
        return False
    
    if not check_database():
        logger.error("Failed to ensure database is initialized.")
        return False
    
    # Run the Flask app
    try:
        from main import app
        logger.info("Running Flask application...")
        app.run(host="0.0.0.0", port=5000, debug=True)
        return True
    except Exception as e:
        logger.error(f"Error running Flask application: {str(e)}")
        return False

if __name__ == "__main__":
    run_app()
