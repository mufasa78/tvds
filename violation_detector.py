import cv2
import numpy as np
from datetime import datetime

class ViolationDetector:
    """
    Class for detecting traffic violations based on tracking and detection results
    """
    def __init__(self):
        # Parameters for violation detection
        self.min_violation_confidence = 0.65
        self.zebra_violation_distance = 20  # pixels
        self.lane_violation_threshold = 0.5  # overlap ratio
        self.min_speed_for_violation = 5  # pixels per frame
        self.frames_to_confirm = 3  # number of consecutive frames to confirm violation
        
        # Store pending violations to confirm them over multiple frames
        self.pending_violations = {}
        
        # Store confirmed violations to avoid duplicates
        self.confirmed_violations = set()
        
        # Frame counter
        self.frame_counter = 0

    def detect_violations(self, frame, tracks, detections):
        """
        Detect traffic violations based on tracks and detections
        
        Args:
            frame: Current video frame
            tracks: Dictionary with tracks from the tracker
            detections: Dictionary with detections from the detector
            
        Returns:
            List of detected violations
        """
        self.frame_counter += 1
        
        if frame is None or frame.size == 0:
            return []
        
        vehicle_tracks = tracks.get('vehicles', [])
        pedestrian_tracks = tracks.get('pedestrians', [])
        zebra_crossings = detections.get('zebra_crossings', [])
        lane_lines = detections.get('lane_lines', [])
        
        # Make a copy of the frame for violation screenshots
        violation_frame = frame.copy()
        
        # List to store detected violations
        violations = []
        
        # Check for violations
        if vehicle_tracks and (zebra_crossings or lane_lines):
            # 1. Check for line crossing violations
            line_violations = self._detect_line_crossing(vehicle_tracks, lane_lines, violation_frame)
            violations.extend(line_violations)
            
            # 2. Check for not yielding to pedestrians at zebra crossings
            if pedestrian_tracks and zebra_crossings:
                yielding_violations = self._detect_not_yielding(
                    vehicle_tracks, pedestrian_tracks, zebra_crossings, violation_frame
                )
                violations.extend(yielding_violations)
            
            # 3. Check for license plate violations (missing/unreadable)
            plate_violations = self._detect_license_plate_violations(vehicle_tracks, violation_frame)
            violations.extend(plate_violations)
        
        # Update and expire pending violations
        self._update_pending_violations()
        
        return violations
    
    def _detect_line_crossing(self, vehicle_tracks, lane_lines, frame):
        """Detect vehicles crossing lane lines"""
        violations = []
        
        for lane in lane_lines:
            lane_bbox = lane['bbox']
            for vehicle in vehicle_tracks:
                vehicle_bbox = vehicle['bbox']
                
                # Check if vehicle is moving
                if self._is_vehicle_moving(vehicle):
                    # Check if vehicle and lane overlap
                    overlap_ratio = self._calculate_bbox_overlap(vehicle_bbox, lane_bbox)
                    
                    if overlap_ratio > self.lane_violation_threshold:
                        # This is a potential violation
                        vehicle_id = vehicle['id']
                        violation_key = f"line_crossing_{vehicle_id}_{self.frame_counter % 100}"
                        
                        # Check if we've already confirmed this vehicle's violation
                        if violation_key not in self.confirmed_violations:
                            # Add to pending violations
                            if violation_key not in self.pending_violations:
                                self.pending_violations[violation_key] = {
                                    'count': 1,
                                    'type': 'line_crossing',
                                    'vehicle_id': vehicle_id,
                                    'frame': frame.copy(),
                                    'license_plate': vehicle.get('license_plate', 'Unknown'),
                                    'first_detected': self.frame_counter,
                                    'confidence': overlap_ratio
                                }
                            else:
                                self.pending_violations[violation_key]['count'] += 1
                                
                            # If violation occurs for enough consecutive frames, confirm it
                            if self.pending_violations[violation_key]['count'] >= self.frames_to_confirm:
                                violations.append({
                                    'type': 'line_crossing',
                                    'vehicle_id': vehicle_id,
                                    'frame': self.pending_violations[violation_key]['frame'],
                                    'license_plate': vehicle.get('license_plate', 'Unknown'),
                                    'confidence': overlap_ratio
                                })
                                
                                # Add to confirmed violations
                                self.confirmed_violations.add(violation_key)
                                
                                # Remove from pending
                                del self.pending_violations[violation_key]
        
        return violations
    
    def _detect_not_yielding(self, vehicle_tracks, pedestrian_tracks, zebra_crossings, frame):
        """Detect vehicles not yielding to pedestrians at zebra crossings"""
        violations = []
        
        for zebra in zebra_crossings:
            zebra_bbox = zebra['bbox']
            # Expand zebra bbox slightly to include approaching area
            expanded_zebra = [
                zebra_bbox[0] - 20,
                zebra_bbox[1] - 20,
                zebra_bbox[2] + 20,
                zebra_bbox[3] + 20
            ]
            
            # Check if any pedestrian is on or near the zebra crossing
            pedestrian_on_zebra = False
            for pedestrian in pedestrian_tracks:
                ped_bbox = pedestrian['bbox']
                
                # If pedestrian overlaps with zebra crossing
                if self._bbox_overlap(ped_bbox, zebra_bbox):
                    pedestrian_on_zebra = True
                    break
            
            if pedestrian_on_zebra:
                # Check for vehicles moving through or approaching zebra crossing
                for vehicle in vehicle_tracks:
                    vehicle_bbox = vehicle['bbox']
                    
                    # Check if vehicle is moving
                    if self._is_vehicle_moving(vehicle):
                        # Check if vehicle is on or very near zebra crossing
                        if self._bbox_overlap(vehicle_bbox, expanded_zebra):
                            # This is a potential violation - vehicle not yielding to pedestrian
                            vehicle_id = vehicle['id']
                            violation_key = f"not_yielding_{vehicle_id}_{self.frame_counter % 100}"
                            
                            # Check if we've already confirmed this vehicle's violation
                            if violation_key not in self.confirmed_violations:
                                # Add to pending violations
                                if violation_key not in self.pending_violations:
                                    self.pending_violations[violation_key] = {
                                        'count': 1,
                                        'type': 'not_yielding',
                                        'vehicle_id': vehicle_id,
                                        'frame': frame.copy(),
                                        'license_plate': vehicle.get('license_plate', 'Unknown'),
                                        'first_detected': self.frame_counter,
                                        'confidence': 0.85  # High confidence for this violation type
                                    }
                                else:
                                    self.pending_violations[violation_key]['count'] += 1
                                    
                                # If violation occurs for enough consecutive frames, confirm it
                                if self.pending_violations[violation_key]['count'] >= self.frames_to_confirm:
                                    violations.append({
                                        'type': 'not_yielding',
                                        'vehicle_id': vehicle_id,
                                        'frame': self.pending_violations[violation_key]['frame'],
                                        'license_plate': vehicle.get('license_plate', 'Unknown'),
                                        'confidence': 0.85
                                    })
                                    
                                    # Add to confirmed violations
                                    self.confirmed_violations.add(violation_key)
                                    
                                    # Remove from pending
                                    del self.pending_violations[violation_key]
        
        return violations
    
    def _detect_license_plate_violations(self, vehicle_tracks, frame):
        """Detect vehicles with missing or unreadable license plates"""
        violations = []
        
        for vehicle in vehicle_tracks:
            # Check if vehicle has a license plate
            license_plate = vehicle.get('license_plate', None)
            
            if not license_plate or license_plate == "Unknown":
                # This is a potential violation - vehicle without readable license plate
                vehicle_id = vehicle['id']
                violation_key = f"license_plate_{vehicle_id}_{self.frame_counter % 300}"
                
                # Only report once every 300 frames for the same vehicle to avoid duplicates
                if violation_key not in self.confirmed_violations:
                    # Add to pending violations
                    if violation_key not in self.pending_violations:
                        self.pending_violations[violation_key] = {
                            'count': 1,
                            'type': 'license_plate',
                            'vehicle_id': vehicle_id,
                            'frame': frame.copy(),
                            'license_plate': 'Unknown',
                            'first_detected': self.frame_counter,
                            'confidence': 0.75
                        }
                    else:
                        self.pending_violations[violation_key]['count'] += 1
                        
                    # If violation occurs for more frames, confirm it
                    # For license plate violations, we require more confirmations
                    if self.pending_violations[violation_key]['count'] >= 2 * self.frames_to_confirm:
                        violations.append({
                            'type': 'license_plate',
                            'vehicle_id': vehicle_id,
                            'frame': self.pending_violations[violation_key]['frame'],
                            'license_plate': 'Unknown',
                            'confidence': 0.75
                        })
                        
                        # Add to confirmed violations
                        self.confirmed_violations.add(violation_key)
                        
                        # Remove from pending
                        del self.pending_violations[violation_key]
        
        return violations
    
    def _update_pending_violations(self):
        """Update pending violations and expire old ones"""
        keys_to_remove = []
        
        for key, violation in self.pending_violations.items():
            # Check if violation is too old (not confirmed within 30 frames)
            if self.frame_counter - violation['first_detected'] > 30:
                keys_to_remove.append(key)
        
        # Remove expired violations
        for key in keys_to_remove:
            del self.pending_violations[key]
        
        # Also limit the size of confirmed_violations set to avoid memory issues
        if len(self.confirmed_violations) > 1000:
            # Convert to list, keep only the last 500 elements, convert back to set
            confirmed_list = list(self.confirmed_violations)
            self.confirmed_violations = set(confirmed_list[-500:])
    
    def _is_vehicle_moving(self, vehicle):
        """Check if a vehicle is moving based on velocity"""
        velocity = vehicle.get('velocity', [0, 0])
        speed = (velocity[0]**2 + velocity[1]**2)**0.5
        return speed > self.min_speed_for_violation
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate the overlap ratio between two bounding boxes"""
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate area of the first bounding box
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        # Calculate overlap ratio
        if bbox1_area > 0:
            return intersection_area / bbox1_area
        else:
            return 0.0
    
    def _bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap"""
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
