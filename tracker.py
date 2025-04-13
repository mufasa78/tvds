import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    """
    Class for tracking vehicles using DeepSORT algorithm
    """
    def __init__(self, max_age=30, n_init=3):
        """
        Initialize the DeepSORT tracker
        
        Args:
            max_age: Maximum number of frames a track can be lost before being removed
            n_init: Number of consecutive detections needed to initialize a track
        """
        # Initialize DeepSORT
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder="mobilenet",
            embedder_model_name="osnet_x0_25",
            embedder_wts=None,  # Use default weights
            polygon=False,
            today=None
        )
        
        # Track history for each tracked object
        self.track_history = {}
        
        # Maximum history length
        self.max_history_len = 30

    def update(self, detections, frame):
        """
        Update tracks with new detections
        
        Args:
            detections: Dictionary containing detection results from detector
            frame: Current video frame
            
        Returns:
            Dictionary with updated tracks
        """
        if frame is None or frame.size == 0:
            return {'vehicles': [], 'pedestrians': []}
        
        # Extract vehicle detections
        vehicle_dets = []
        for vehicle in detections.get('vehicles', []):
            bbox = vehicle['bbox']
            confidence = vehicle['confidence']
            class_id = vehicle['class_id']
            license_plate = vehicle.get('license_plate', None)
            
            # Convert bbox format from [x1, y1, x2, y2] to [x1, y1, w, h]
            bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            
            vehicle_dets.append({
                'bbox': bbox_xywh,
                'confidence': confidence,
                'class_id': class_id,
                'license_plate': license_plate
            })
        
        # Extract pedestrian detections
        pedestrian_dets = []
        for pedestrian in detections.get('pedestrians', []):
            bbox = pedestrian['bbox']
            confidence = pedestrian['confidence']
            
            # Convert bbox format
            bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            
            pedestrian_dets.append({
                'bbox': bbox_xywh,
                'confidence': confidence
            })
        
        # Update vehicle tracks
        vehicle_tracks = []
        if vehicle_dets:
            # Format detections for DeepSORT
            deepsort_dets = []
            for i, det in enumerate(vehicle_dets):
                deepsort_dets.append((
                    det['bbox'],  # [x, y, w, h]
                    det['confidence'],
                    f"vehicle_{det['class_id']}"  # Class as string
                ))
            
            # Update tracker
            tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
            
            # Process tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_ltrb()  # Left, Top, Right, Bottom format
                
                # Get original detection info by matching bbox
                det_info = None
                for det in vehicle_dets:
                    det_bbox = [
                        det['bbox'][0], 
                        det['bbox'][1], 
                        det['bbox'][0] + det['bbox'][2], 
                        det['bbox'][1] + det['bbox'][3]
                    ]
                    iou = self._calculate_iou(bbox, det_bbox)
                    if iou > 0.5:
                        det_info = det
                        break
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(bbox)
                
                # Limit history length
                if len(self.track_history[track_id]) > self.max_history_len:
                    self.track_history[track_id] = self.track_history[track_id][-self.max_history_len:]
                
                # Calculate velocity based on history
                velocity = self._calculate_velocity(self.track_history[track_id])
                
                vehicle_tracks.append({
                    'id': track_id,
                    'bbox': bbox,
                    'class_id': det_info['class_id'] if det_info else None,
                    'license_plate': det_info['license_plate'] if det_info else None,
                    'history': self.track_history[track_id],
                    'velocity': velocity
                })
        
        # Update pedestrian tracks
        pedestrian_tracks = []
        if pedestrian_dets:
            # Format detections for DeepSORT
            deepsort_dets = []
            for i, det in enumerate(pedestrian_dets):
                deepsort_dets.append((
                    det['bbox'],  # [x, y, w, h]
                    det['confidence'],
                    "pedestrian"  # Class as string
                ))
            
            # Update tracker
            tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
            
            # Process tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_ltrb()  # Left, Top, Right, Bottom format
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(bbox)
                
                # Limit history length
                if len(self.track_history[track_id]) > self.max_history_len:
                    self.track_history[track_id] = self.track_history[track_id][-self.max_history_len:]
                
                # Calculate velocity based on history
                velocity = self._calculate_velocity(self.track_history[track_id])
                
                pedestrian_tracks.append({
                    'id': track_id,
                    'bbox': bbox,
                    'history': self.track_history[track_id],
                    'velocity': velocity
                })
        
        return {
            'vehicles': vehicle_tracks,
            'pedestrians': pedestrian_tracks
        }
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate area of intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas of both bounding boxes
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        area_u = area_1 + area_2 - area_i
        iou = area_i / area_u if area_u > 0 else 0.0
        
        return iou
    
    def _calculate_velocity(self, history):
        """Calculate velocity based on position history"""
        if len(history) < 2:
            return [0, 0]
        
        # Get center points of the last two bounding boxes
        last_bbox = history[-1]
        prev_bbox = history[-2]
        
        last_center = [(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2]
        prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2]
        
        # Calculate displacement vector
        dx = last_center[0] - prev_center[0]
        dy = last_center[1] - prev_center[1]
        
        return [dx, dy]
