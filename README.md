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
├── setup.py               # Setup script for initializing the application
├── init_db.py             # Database initialization script
├── download_models.py     # Script to download required model files
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
   git clone https://github.com/mufasa78/tvds.git
   cd tvds
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Run the setup script to initialize everything:
   ```bash
   python setup.py
   ```
   This script will:
   - Install all required dependencies
   - Set up necessary directories
   - Download required model files
   - Initialize the database

4. Alternatively, you can perform each step manually:
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Download model files
   python download_models.py

   # Initialize database
   python init_db.py
   ```

5. Set up environment variables (or create a .env file):
   ```bash
   # On Windows
   set DATABASE_URL=postgresql://username:password@localhost/traffic_violations
   set SESSION_SECRET=your-secret-key

   # On macOS/Linux
   export DATABASE_URL=postgresql://username:password@localhost/traffic_violations
   export SESSION_SECRET=your-secret-key
   ```
   You can also copy the .env.example file to .env and edit it with your settings.

## Running the Application

### Development Mode

Use the run script which ensures all components are properly initialized:

```bash
python run.py
```

Or run the main application directly:

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

## Data Synchronization

To ensure all data is properly synchronized between environments, use the sync_data.py script:

```bash
python sync_data.py
```

This script will:
- Check and create necessary directories
- Verify model files are present and download them if needed
- Create a data manifest file with information about the current state

After running this script, you can commit and push your changes to the repository using the push_to_github.py script:

```bash
python push_to_github.py -m "Your commit message here"
```

This script will:
- Check git status
- Run the data synchronization script
- Add all changes to git
- Commit with the provided message
- Push to the GitHub repository

You can also specify specific files to add:

```bash
python push_to_github.py -m "Update specific files" -f file1.py file2.py
```

## License

[MIT License](LICENSE)

## Acknowledgements

- YOLOv8 by Ultralytics
- DeepSORT for object tracking
- Flask web framework
- Bootstrap for UI components
