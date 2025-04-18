#!/usr/bin/env python
"""
Setup script for Traffic Violation Detection System.
This script ensures all components are properly configured and synchronized.
"""

import os
import sys
import subprocess
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        logger.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return False

def check_dependencies():
    """Check if all required Python packages are installed."""
    logger.info("Checking Python dependencies...")
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def setup_directories():
    """Ensure all required directories exist."""
    logger.info("Setting up directories...")
    directories = [
        "static/uploads",
        "static/uploads/violations"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
        else:
            logger.info(f"Directory already exists: {directory}")

    return True

def download_models():
    """Download required model files."""
    logger.info("Checking model files...")
    return run_command("python download_models.py", "Downloading model files")

def check_database():
    """Check database connection."""
    logger.info("Checking database connection...")
    return run_command("python check_db.py", "Checking database connection")

def initialize_database():
    """Initialize the database."""
    logger.info("Initializing database...")
    return run_command("python init_db.py", "Initializing database")

def main():
    """Main setup function."""
    logger.info("Starting setup for Traffic Violation Detection System...")

    steps = [
        ("Checking dependencies", check_dependencies),
        ("Setting up directories", setup_directories),
        ("Downloading models", download_models),
        ("Checking database connection", check_database),
        ("Initializing database", initialize_database)
    ]

    success = True
    for description, step_function in steps:
        logger.info(f"\n=== {description} ===")
        if not step_function():
            logger.error(f"Step failed: {description}")
            success = False
            break

    if success:
        logger.info("\n=== Setup completed successfully! ===")
        logger.info("You can now run the application with: python main.py")
    else:
        logger.error("\n=== Setup failed! ===")
        logger.error("Please fix the errors above and try again.")

    return success

if __name__ == "__main__":
    main()
