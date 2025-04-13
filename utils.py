import cv2
import numpy as np
import os
import uuid
from datetime import datetime

def draw_box(frame, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw a bounding box on the frame with an optional label
    
    Args:
        frame: The image to draw on
        bbox: Bounding box in [x1, y1, x2, y2] format
        color: Box color as BGR tuple
        thickness: Line thickness
        label: Text label to display
    """
    if frame is None or len(bbox) != 4:
        return
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # Get text size
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
        
        # Calculate text position
        text_x = x1
        text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + text_size[1]
        
        # Draw filled rectangle behind text
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                     (text_x + text_size[0], text_y + 5), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_violation(frame, violation_type, bbox, license_plate=None):
    """
    Draw violation information on the frame
    
    Args:
        frame: The image to draw on
        violation_type: Type of violation
        bbox: Bounding box of the violating vehicle
        license_plate: License plate text if available
    """
    if frame is None or len(bbox) != 4:
        return
    
    # Colors for different violation types
    colors = {
        'line_crossing': (0, 0, 255),  # Red
        'not_yielding': (0, 165, 255),  # Orange
        'license_plate': (255, 0, 0),  # Blue
    }
    
    # Get color for this violation type
    color = colors.get(violation_type, (0, 255, 255))
    
    # Draw thicker box for violation
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Formatted labels for different violation types
    labels = {
        'line_crossing': "Violation: Line Crossing",
        'not_yielding': "Violation: Not Yielding",
        'license_plate': "Violation: Plate Issue"
    }
    
    # Prepare the label
    label = labels.get(violation_type, "Violation")
    if license_plate and license_plate != "Unknown":
        label += f" - {license_plate}"
    
    # Draw violation label
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def save_violation_image(frame, violation_type, output_dir, job_id):
    """
    Save an image of a detected violation
    
    Args:
        frame: The image to save
        violation_type: Type of violation
        output_dir: Directory to save the image
        job_id: Job identifier
        
    Returns:
        Path to the saved image file
    """
    if frame is None:
        return None
    
    # Create unique filename
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violation_type}_{timestamp}_{unique_id}.jpg"
    
    # Create full filepath
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    
    return filepath

def format_time(seconds):
    """
    Format time in seconds to HH:MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_violation_metrics(violations):
    """
    Calculate metrics from a list of violations
    
    Args:
        violations: List of violation objects
        
    Returns:
        Dictionary with violation metrics
    """
    if not violations:
        return {
            'total': 0,
            'by_type': {},
            'by_hour': {}
        }
    
    total = len(violations)
    by_type = {}
    by_hour = {}
    
    for violation in violations:
        # Count by violation type
        v_type = violation.violation_type
        if v_type not in by_type:
            by_type[v_type] = 0
        by_type[v_type] += 1
        
        # Count by hour of day
        if violation.created_at:
            hour = violation.created_at.hour
            if hour not in by_hour:
                by_hour[hour] = 0
            by_hour[hour] += 1
    
    return {
        'total': total,
        'by_type': by_type,
        'by_hour': by_hour
    }
