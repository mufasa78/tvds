#!/usr/bin/env python
"""
Data synchronization script for Traffic Violation Detection System.
This script ensures all necessary data is properly synced between environments.
"""

import os
import sys
import subprocess
import logging
import shutil
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directories and files to sync
DIRECTORIES_TO_SYNC = [
    "static/uploads",
    "static/css",
    "static/js",
    "templates"
]

MODEL_FILES = [
    "yolov8n.pt",
    "yolov8n-seg.pt"
]

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        logger.info(f"Command output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return False, e.stderr

def check_git_status():
    """Check the status of the git repository."""
    logger.info("Checking git status...")
    success, output = run_command("git status", "Git status")
    
    if not success:
        logger.error("Failed to check git status. Make sure git is installed and you're in a git repository.")
        return False
    
    return True

def sync_directories():
    """Ensure all directories exist and are tracked in git."""
    logger.info("Syncing directories...")
    
    for directory in DIRECTORIES_TO_SYNC:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep file if directory is empty
        if os.path.exists(directory) and not os.listdir(directory):
            gitkeep_path = os.path.join(directory, ".gitkeep")
            if not os.path.exists(gitkeep_path):
                logger.info(f"Creating .gitkeep file in {directory}")
                with open(gitkeep_path, 'w') as f:
                    pass
    
    return True

def check_model_files():
    """Check if model files exist and download them if needed."""
    logger.info("Checking model files...")
    
    missing_models = [model for model in MODEL_FILES if not os.path.exists(model)]
    
    if missing_models:
        logger.info(f"Missing model files: {', '.join(missing_models)}")
        logger.info("Running download_models.py...")
        
        success, _ = run_command("python download_models.py", "Download models")
        if not success:
            logger.error("Failed to download model files.")
            return False
    else:
        logger.info("All model files are present.")
    
    return True

def create_data_manifest():
    """Create a manifest file with information about the data."""
    logger.info("Creating data manifest...")
    
    manifest = {
        "directories": {},
        "model_files": {}
    }
    
    # Get information about directories
    for directory in DIRECTORIES_TO_SYNC:
        if os.path.exists(directory):
            files = os.listdir(directory)
            manifest["directories"][directory] = {
                "exists": True,
                "file_count": len(files),
                "files": files[:10]  # List first 10 files
            }
        else:
            manifest["directories"][directory] = {
                "exists": False
            }
    
    # Get information about model files
    for model_file in MODEL_FILES:
        if os.path.exists(model_file):
            manifest["model_files"][model_file] = {
                "exists": True,
                "size_bytes": os.path.getsize(model_file)
            }
        else:
            manifest["model_files"][model_file] = {
                "exists": False
            }
    
    # Write manifest to file
    with open("data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("Data manifest created: data_manifest.json")
    return True

def main():
    """Main function to sync data."""
    logger.info("Starting data synchronization...")
    
    # Check git status
    if not check_git_status():
        return False
    
    # Sync directories
    if not sync_directories():
        return False
    
    # Check model files
    if not check_model_files():
        return False
    
    # Create data manifest
    if not create_data_manifest():
        return False
    
    logger.info("Data synchronization completed successfully!")
    logger.info("You can now commit and push your changes to the repository.")
    
    return True

if __name__ == "__main__":
    main()
