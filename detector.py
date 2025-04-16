import cv2
import numpy as np
from ultralytics import YOLO
import torch

class VehicleDetector:
    """
    Class for detecting vehicles, pedestrians, lane lines, and license plates using YOLOv8
    """
    def __init__(self):
        # Load YOLOv8 models
        try:
            # For vehicles and pedestrians
            self.yolo_model = YOLO('yolov8n.pt')  # using nano model for efficiency

            # For lane line detection
            self.lane_model = YOLO('yolov8n-seg.pt')  # Using segmentation model for lane lines

            # For license plate detection and recognition
            # In a production environment, you would use a specialized license plate model
            # For this demo, we'll use the same general model but focus on relevant classes
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            print(f"Models loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading YOLOv8 models: {str(e)}")
            raise

        # Class indices for relevant objects in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.pedestrian_class = 0  # person

        # Confidence thresholds
        self.vehicle_threshold = 0.5
        self.pedestrian_threshold = 0.5
        self.lane_threshold = 0.45
        self.license_plate_threshold = 0.6

    def detect(self, frame):
        """
        Detect vehicles, pedestrians, lane lines, and license plates in a frame

        Args:
            frame: Image/video frame to process

        Returns:
            Dictionary with detection results
        """
        if frame is None or frame.size == 0:
            return {
                'vehicles': [],
                'pedestrians': [],
                'lane_lines': [],
                'license_plates': []
            }

        # Make a copy of the frame to avoid modifying the original
        detection_frame = frame.copy()

        # Get frame dimensions
        height, width = detection_frame.shape[:2]

        # Run YOLOv8 detection
        try:
            # Detect objects (vehicles and pedestrians)
            results = self.yolo_model(detection_frame, verbose=False)

            # Detect lane lines using segmentation model
            lane_results = self.lane_model(detection_frame, verbose=False)

            # Process detections
            vehicles = []
            pedestrians = []
            license_plates = []

            # Extract bounding boxes, classes and confidence scores
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Check if detection is a vehicle
                    if cls_id in self.vehicle_classes and conf > self.vehicle_threshold:
                        # Extract potential license plate region (bottom half of vehicle)
                        plate_y1 = int(y1 + (y2 - y1) * 0.6)  # Bottom 40% of vehicle
                        plate_region = detection_frame[plate_y1:y2, x1:x2]

                        license_plate = None
                        # Only try to detect license plate if region is large enough
                        if plate_region.size > 0 and plate_region.shape[0] > 10 and plate_region.shape[1] > 10:
                            # We would use a specialized license plate detector here
                            # For demo purposes, we'll use a placeholder detection
                            license_plate = "ABC123"  # Placeholder
                            license_plate_conf = 0.8  # Placeholder confidence
                            license_plates.append({
                                'bbox': [x1, plate_y1, x2, y2],
                                'confidence': license_plate_conf,
                                'text': license_plate
                            })

                        vehicles.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls_id,
                            'license_plate': license_plate
                        })

                    # Check if detection is a pedestrian
                    elif cls_id == self.pedestrian_class and conf > self.pedestrian_threshold:
                        pedestrians.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })

            # Process lane line detections from segmentation model
            lane_lines = []
            if lane_results and len(lane_results) > 0:
                for r in lane_results:
                    if hasattr(r, 'masks') and r.masks is not None and r.masks.data is not None:
                        try:
                            masks = r.masks.data
                            for _, mask in enumerate(masks):
                                # Convert mask to binary image
                                mask_array = mask.cpu().numpy()
                                # Resize mask_array to match the frame dimensions if needed
                                if mask_array.shape[0] != height or mask_array.shape[1] != width:
                                    # Create a properly sized binary mask
                                    binary_mask = np.zeros((height, width), dtype=np.uint8)
                                    # Resize the mask array to match the frame dimensions
                                    resized_mask = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
                                    binary_mask[resized_mask > 0.5] = 255
                                else:
                                    # If dimensions already match, proceed as before
                                    binary_mask = np.zeros((height, width), dtype=np.uint8)
                                    binary_mask[mask_array > 0.5] = 255

                                # Find contours in the mask
                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                # Check if contour is likely a lane line (elongated shape)
                                for contour in contours:
                                    # Get bounding rectangle
                                    x, y, w, h = cv2.boundingRect(contour)

                                    # Calculate aspect ratio
                                    aspect_ratio = float(h) / w if w > 0 else 0

                                    # Lane lines are usually elongated
                                    if aspect_ratio > 3 or aspect_ratio < 0.33:
                                        area = cv2.contourArea(contour)
                                        # Minimum area to avoid noise
                                        if area > 100:
                                            # Simplify contour for efficiency
                                            epsilon = 0.02 * cv2.arcLength(contour, True)
                                            approx = cv2.approxPolyDP(contour, epsilon, True)

                                            lane_lines.append({
                                                'contour': approx.tolist(),
                                                'bbox': [x, y, x + w, y + h],
                                                'area': area
                                            })
                        except Exception as e:
                            print(f"Error processing mask: {str(e)}")
                            continue

            # Detect zebra crossings (horizontal white stripes pattern)
            # This is a simplified approach for demo purposes
            # In production, you would use a specialized model or more sophisticated image processing

            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Apply morphological operations to enhance zebra patterns
            kernel = np.ones((3, 15), np.uint8)  # Horizontal kernel to connect zebra stripes
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours of potential zebra crossings
            zebra_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            zebra_crossings = []
            for contour in zebra_contours:
                # Filter by size and shape
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                # Zebra crossings usually have significant area
                if area > 1000:
                    # Check if it has horizontal stripe pattern
                    roi = thresh[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Count horizontal lines using horizontal projection
                        h_projection = np.sum(roi, axis=1)
                        # Count transitions (white to black)
                        transitions = np.sum(np.abs(np.diff(h_projection > 0)))

                        # Zebra crossings have multiple horizontal stripes
                        if transitions >= 4:
                            zebra_crossings.append({
                                'bbox': [x, y, x + w, y + h],
                                'area': area
                            })

            return {
                'vehicles': vehicles,
                'pedestrians': pedestrians,
                'lane_lines': lane_lines,
                'license_plates': license_plates,
                'zebra_crossings': zebra_crossings
            }

        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return {
                'vehicles': [],
                'pedestrians': [],
                'lane_lines': [],
                'license_plates': [],
                'zebra_crossings': []
            }
