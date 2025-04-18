#!/usr/bin/env python
"""
GitHub push script for Traffic Violation Detection System.
This script helps push changes to the GitHub repository.
"""

import os
import sys
import subprocess
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GitHub repository URL
GITHUB_REPO = "https://github.com/mufasa78/tvds.git"

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
        return False, None
    
    return True, output

def sync_data():
    """Run the data synchronization script."""
    logger.info("Syncing data...")
    success, _ = run_command("python sync_data.py", "Sync data")
    return success

def add_changes(files=None):
    """Add changes to git."""
    if files:
        # Add specific files
        file_list = " ".join(files)
        command = f"git add {file_list}"
        description = f"Add specific files: {file_list}"
    else:
        # Add all changes
        command = "git add ."
        description = "Add all changes"
    
    success, _ = run_command(command, description)
    return success

def commit_changes(message):
    """Commit changes to git."""
    command = f'git commit -m "{message}"'
    success, _ = run_command(command, f'Commit with message: "{message}"')
    return success

def push_changes(branch="main"):
    """Push changes to GitHub."""
    # Check if remote exists
    _, remote_output = run_command("git remote -v", "Check remotes")
    
    if "origin" not in remote_output:
        logger.info("Remote 'origin' not found. Adding it...")
        success, _ = run_command(f"git remote add origin {GITHUB_REPO}", "Add remote")
        if not success:
            return False
    
    # Push to GitHub
    command = f"git push -u origin {branch}"
    success, _ = run_command(command, f"Push to {branch} branch")
    return success

def main():
    """Main function to push changes to GitHub."""
    parser = argparse.ArgumentParser(description="Push changes to GitHub repository")
    parser.add_argument("--message", "-m", required=True, help="Commit message")
    parser.add_argument("--files", "-f", nargs="*", help="Specific files to add (default: all changes)")
    parser.add_argument("--branch", "-b", default="main", help="Branch to push to (default: main)")
    parser.add_argument("--skip-sync", action="store_true", help="Skip data synchronization")
    
    args = parser.parse_args()
    
    logger.info("Starting GitHub push process...")
    
    # Check git status
    status_success, status_output = check_git_status()
    if not status_success:
        return False
    
    # Sync data if not skipped
    if not args.skip_sync:
        if not sync_data():
            logger.error("Data synchronization failed. Fix the issues before continuing.")
            return False
    
    # Add changes
    if not add_changes(args.files):
        logger.error("Failed to add changes to git.")
        return False
    
    # Commit changes
    if not commit_changes(args.message):
        logger.error("Failed to commit changes.")
        return False
    
    # Push changes
    if not push_changes(args.branch):
        logger.error("Failed to push changes to GitHub.")
        return False
    
    logger.info("Successfully pushed changes to GitHub!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
