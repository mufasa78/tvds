# Traffic Violation Detection System

A web-based application for automated detection of traffic violations in video footage using computer vision and machine learning.

![Traffic Violation Detection System](https://via.placeholder.com/800x400?text=Traffic+Violation+Detection+System)

## Features

- **Video Upload & Analysis**: Upload traffic videos for automated violation detection
- **Multi-language Support**: Available in English and Chinese
- **Real-time Processing**: Background processing with progress tracking
- **Violation Detection**:
  - Lane line crossing violations
  - License plate issues (missing, obscured)
  - Pedestrian right-of-way violations
- **Detailed Reports**: View violations with timestamps, screenshots, and confidence scores
- **Computer Vision Pipeline**:
  - Object Detection: YOLOv8 for vehicles, pedestrians, lane lines, and zebra crossings
  - Object Tracking: DeepSORT for tracking objects across video frames
  - Violation Analysis: Custom algorithms to detect rule violations

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Computer Vision**: OpenCV, YOLOv8, DeepSORT
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **Deployment**: Gunicorn (WSGI HTTP Server)

## Project Structure

```
project/
├── app.py                 # Main Flask application
├── main.py                # Entry point for running the application
├── models.py              # Database models
├── detector.py            # Vehicle and object detection using YOLOv8
├── tracker.py             # Object tracking using DeepSORT
├── violation_detector.py  # Traffic violation detection logic
├── requirements.txt       # Python dependencies
├── static/                # Static assets
│   ├── css/               # CSS stylesheets
│   ├── js/                # JavaScript files
│   └── uploads/           # Uploaded videos and screenshots
└── templates/             # HTML templates
    ├── index.html         # Homepage
    ├── layout.html        # Base template
    ├── analysis.html      # Analysis status page
    ├── violations.html    # Violation details page
    └── analyses.html      # List of all analyses
```

## Installation

### Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-violation-detection.git
   cd traffic-violation-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # On Windows
   set DATABASE_URL=postgresql://username:password@localhost/traffic_violations
   set SESSION_SECRET=your-secret-key
   
   # On macOS/Linux
   export DATABASE_URL=postgresql://username:password@localhost/traffic_violations
   export SESSION_SECRET=your-secret-key
   ```

5. Download YOLOv8 models (will be downloaded automatically on first run):
   - `yolov8n.pt` - YOLOv8 nano model for object detection
   - `yolov8n-seg.pt` - YOLOv8 nano segmentation model for lane lines

## Running the Application

### Development Mode

```bash
python main.py
```

The application will be available at http://localhost:5000

### Production Deployment

For production deployment, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

## Usage

1. Access the web interface at http://localhost:5000
2. Upload a traffic video file (MP4, AVI, MOV, or WEBM format)
3. Wait for the analysis to complete (progress is shown in real-time)
4. View the detected violations with details and screenshots
5. Filter violations by type and export reports as needed

## Fallback Mode

The system includes a fallback mode that uses mock implementations if OpenCV or other dependencies are not available. This allows the application to run for demonstration purposes even without all the required libraries.

## Database Configuration

The application uses PostgreSQL by default, but you can modify the database connection in `app.py`:

```python
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "postgresql://username:password@localhost/traffic_violations")
```

## Customization

### Adding New Violation Types

To add new violation types:

1. Update the detection logic in `violation_detector.py`
2. Add translations for the new violation type in `app.py`
3. Update the UI templates to display the new violation type

### Changing Detection Parameters

Adjust detection thresholds and parameters in:
- `detector.py` - For object detection sensitivity
- `tracker.py` - For object tracking parameters
- `violation_detector.py` - For violation detection thresholds

## License

[MIT License](LICENSE)

## Acknowledgements

- YOLOv8 by Ultralytics
- DeepSORT for object tracking
- Flask web framework
- Bootstrap for UI components
