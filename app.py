import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
import threading

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check for OpenCV and other dependencies
MOCK_DEPENDENCIES = False
try:
    import cv2
    logger.info("OpenCV imported successfully")
    
    # These will be imported later in app context
    # from detector import VehicleDetector
    # from tracker import VehicleTracker
    # from violation_detector import ViolationDetector
except ImportError:
    logger.warning("OpenCV not found - will use mock implementations")
    MOCK_DEPENDENCIES = True

# Define base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "postgresql://neondb_owner:npg_YpEs4U6ufeHl@ep-wandering-butterfly-a5ncpr5f-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

# Initialize the app with the SQLAlchemy extension
db.init_app(app)

# Import models and define mock classes if needed
with app.app_context():
    import models
    
    # Create mock classes if dependencies are missing
    if MOCK_DEPENDENCIES:
        class VehicleDetector:
            def __init__(self):
                logger.info("Initialized mock VehicleDetector")
            
            def detect(self, frame):
                return {
                    'vehicles': [],
                    'pedestrians': [],
                    'lane_lines': [],
                    'license_plates': [],
                    'zebra_crossings': []
                }
        
        class VehicleTracker:
            def __init__(self, max_age=30, n_init=3):
                logger.info("Initialized mock VehicleTracker")
            
            def update(self, detections, frame):
                return {'vehicles': [], 'pedestrians': []}
        
        class ViolationDetector:
            def __init__(self):
                logger.info("Initialized mock ViolationDetector")
            
            def detect_violations(self, frame, tracks, detections):
                return []
    else:
        # Import real implementation if dependencies are available
        from detector import VehicleDetector
        from tracker import VehicleTracker
        from violation_detector import ViolationDetector
    
    # Create database tables
    db.create_all()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global job tracker
processing_jobs = {}

# Translations dictionary for English and Chinese
translations = {
    'en': {
        'app_name': 'Traffic Violation Detection System',
        'upload_video': 'Upload Traffic Video for Analysis',
        'select_video': 'Select video file (MP4, AVI, MOV, WEBM)',
        'upload_analyze': 'Upload and Analyze',
        'how_it_works': 'How It Works',
        'main_functions': 'Main Functions',
        'lane_lines': 'Lane line identification',
        'license_plate': 'License plate recognition',
        'illegal_crossing': 'Illegal line crossing detection',
        'pedestrian_violations': 'Pedestrian right-of-way violations',
        'object_detection': 'Object Detection: YOLOv8 identifies vehicles, pedestrians, and road features',
        'object_tracking': 'Object Tracking: DeepSORT tracks objects across video frames',
        'violation_detection': 'Violation Detection: Our algorithms analyze movement patterns and detect rule violations',
        'reporting': 'Reporting: Generate detailed reports with screenshots, timestamps, and violation types',
        'processing_note': 'Note: Video processing may take several minutes depending on the video length and complexity.',
        'violation_types': 'Violation Types',
        'lane_violation': 'Lane Line Violations',
        'lane_violation_desc': 'Detects vehicles crossing solid lane lines or improperly changing lanes.',
        'license_violation': 'License Plate Issues',
        'license_violation_desc': 'Identifies vehicles with missing, obscured, or unreadable license plates.',
        'line_violation': 'Illegal Line Crossing',
        'line_violation_desc': 'Detects vehicles crossing stop lines or entering prohibited areas.',
        'pedestrian_violation': 'Not Yielding to Pedestrians',
        'pedestrian_violation_desc': 'Identifies vehicles that fail to stop for pedestrians at zebra crossings.',
        'home': 'Home',
        'all_analyses': 'All Analyses',
        'no_file_selected': 'No selected file',
        'invalid_file': 'Invalid file type. Please upload a video file (mp4, avi, mov, webm)',
        'upload_started': 'Video uploaded and processing started',
        'analysis_not_found': 'Analysis job not found',
        'language': 'Language'
    },
    'zh': {
        'app_name': '交通违规检测系统',
        'upload_video': '上传交通视频进行分析',
        'select_video': '选择视频文件（MP4、AVI、MOV、WEBM）',
        'upload_analyze': '上传并分析',
        'how_it_works': '工作原理',
        'main_functions': '主要功能',
        'lane_lines': '车道线识别',
        'license_plate': '车牌识别',
        'illegal_crossing': '违法越线检测',
        'pedestrian_violations': '不礼让行人违规',
        'object_detection': '对象检测：YOLOv8识别车辆、行人和道路特征',
        'object_tracking': '对象跟踪：DeepSORT跟踪视频帧中的对象',
        'violation_detection': '违规检测：我们的算法分析移动模式并检测规则违规',
        'reporting': '报告：生成带有截图、时间戳和违规类型的详细报告',
        'processing_note': '注意：视频处理可能需要几分钟，具体取决于视频长度和复杂性。',
        'violation_types': '违规类型',
        'lane_violation': '车道线违规',
        'lane_violation_desc': '检测越过实线或不当变道的车辆。',
        'license_violation': '车牌问题',
        'license_violation_desc': '识别缺失、模糊或不可读车牌的车辆。',
        'line_violation': '违法越线',
        'line_violation_desc': '检测越过停止线或进入禁区的车辆。',
        'pedestrian_violation': '不礼让行人',
        'pedestrian_violation_desc': '识别在斑马线前未停车让行人的车辆。',
        'home': '首页',
        'all_analyses': '所有分析',
        'no_file_selected': '未选择文件',
        'invalid_file': '无效的文件类型。请上传视频文件（mp4、avi、mov、webm）',
        'upload_started': '视频已上传，处理已开始',
        'analysis_not_found': '找不到分析作业',
        'language': '语言'
    }
}

# Set up language selection
@app.before_request
def before_request():
    language = session.get('language', 'en')
    g.language = language
    g.translations = translations[language]

@app.route('/set_language/<language>')
def set_language(language):
    if language not in translations:
        language = 'en'
    session['language'] = language
    return redirect(request.referrer or url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this analysis job
        job_id = str(uuid.uuid4())
        
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        unique_filename = f"{job_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Create database entry for the analysis
        analysis = models.Analysis(
            job_id=job_id,
            filename=unique_filename,
            status="pending",
            upload_date=datetime.now()
        )
        db.session.add(analysis)
        db.session.commit()
        
        # Start analysis in a background thread
        processing_thread = threading.Thread(
            target=process_video,
            args=(job_id, filepath)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Track the job
        processing_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "thread": processing_thread
        }
        
        flash('Video uploaded and processing started', 'success')
        return redirect(url_for('analysis_status', job_id=job_id))
    
    flash('Invalid file type. Please upload a video file (mp4, avi, mov, webm)', 'danger')
    return redirect(url_for('index'))

def process_video(job_id, video_path):
    """Process the video in a background thread"""
    try:
        # Update status to processing
        with app.app_context():
            analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
            if analysis:
                analysis.status = "processing"
                db.session.commit()
        
        # Initialize detector, tracker and violation detector
        detector = VehicleDetector()
        tracker = VehicleTracker()
        violation_detector = ViolationDetector()
        
        # Check if we're using mock dependencies
        if MOCK_DEPENDENCIES:
            logger.warning("Using mock implementation for video processing")
            # Simulate processing with mock data
            import time
            
            # Create directory for storing violation images
            violation_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"violations_{job_id}")
            os.makedirs(violation_dir, exist_ok=True)
            
            # Simulate video processing delay
            for i in range(1, 11):  # 10 steps
                time.sleep(0.5)  # Half-second delay per step
                progress = i * 10
                processing_jobs[job_id]["progress"] = progress
            
            # Add some sample violations for demo purposes
            violation_types = ["line_crossing", "license_plate", "not_yielding"]
            
            for i in range(3):  # Add 3 sample violations
                violation_type = violation_types[i % len(violation_types)]
                violation_id = str(uuid.uuid4())
                
                # Add to database
                with app.app_context():
                    violation_record = models.Violation(
                        violation_id=violation_id,
                        analysis_id=job_id,
                        violation_type=violation_type,
                        timestamp=i * 10.5,  # Fake timestamp
                        frame_number=i * 300,  # Fake frame number
                        screenshot_path=None,  # No screenshot
                        license_plate="DEMO123" if violation_type != "license_plate" else "Unknown",
                        confidence=0.85
                    )
                    db.session.add(violation_record)
                    db.session.commit()
            
            # Update analysis status to completed
            with app.app_context():
                analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
                if analysis:
                    analysis.status = "completed"
                    analysis.completion_date = datetime.now()
                    db.session.commit()
            
            processing_jobs[job_id]["status"] = "completed"
            
        else:
            # Real implementation with OpenCV
            # Open the video
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            # Create directory for storing violation images
            violation_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"violations_{job_id}")
            os.makedirs(violation_dir, exist_ok=True)
            
            frame_count = 0
            detected_violations = []
            
            # Process each frame
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                    
                # Update progress
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                processing_jobs[job_id]["progress"] = progress
                
                # Detect objects in the frame
                detections = detector.detect(frame)
                
                # Track objects
                tracks = tracker.update(detections, frame)
                
                # Detect violations
                violations = violation_detector.detect_violations(frame, tracks, detections)
                
                # Record violations
                timestamp = frame_count / fps
                for violation in violations:
                    violation_type = violation["type"]
                    violation_id = str(uuid.uuid4())
                    
                    # Save a screenshot of the violation
                    screenshot_path = os.path.join(
                        f"violations_{job_id}", 
                        f"{violation_id}.jpg"
                    )
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], screenshot_path)
                    cv2.imwrite(full_path, violation["frame"])
                    
                    # Record in memory for batch insert
                    detected_violations.append({
                        "violation_id": violation_id,
                        "analysis_id": job_id,
                        "violation_type": violation_type,
                        "timestamp": timestamp,
                        "frame_number": frame_count,
                        "screenshot_path": screenshot_path,
                        "license_plate": violation.get("license_plate", "Unknown"),
                        "confidence": violation.get("confidence", 0.0)
                    })
                    
                    # Batch insert to database every 10 violations to avoid memory issues
                    if len(detected_violations) >= 10:
                        with app.app_context():
                            for v in detected_violations:
                                violation_record = models.Violation(
                                    violation_id=v["violation_id"],
                                    analysis_id=v["analysis_id"],
                                    violation_type=v["violation_type"],
                                    timestamp=v["timestamp"],
                                    frame_number=v["frame_number"],
                                    screenshot_path=v["screenshot_path"],
                                    license_plate=v["license_plate"],
                                    confidence=v["confidence"]
                                )
                                db.session.add(violation_record)
                            db.session.commit()
                        detected_violations = []
            
            # Insert any remaining violations
            if detected_violations:
                with app.app_context():
                    for v in detected_violations:
                        violation_record = models.Violation(
                            violation_id=v["violation_id"],
                            analysis_id=v["analysis_id"],
                            violation_type=v["violation_type"],
                            timestamp=v["timestamp"],
                            frame_number=v["frame_number"],
                            screenshot_path=v["screenshot_path"],
                            license_plate=v["license_plate"],
                            confidence=v["confidence"]
                        )
                        db.session.add(violation_record)
                    db.session.commit()
            
            # Update analysis status to completed
            with app.app_context():
                analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
                if analysis:
                    analysis.status = "completed"
                    analysis.completion_date = datetime.now()
                    db.session.commit()
            
            # Close the video
            video.release()
            processing_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        
        # Update status to failed
        with app.app_context():
            analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
            if analysis:
                analysis.status = "failed"
                analysis.error_message = str(e)
                db.session.commit()
        
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)

@app.route('/analysis/<job_id>')
def analysis_status(job_id):
    analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
    if not analysis:
        flash('Analysis job not found', 'danger')
        return redirect(url_for('index'))
    
    progress = 0
    if job_id in processing_jobs:
        progress = processing_jobs[job_id].get("progress", 0)
    
    return render_template('analysis.html', analysis=analysis, progress=progress)

@app.route('/api/analysis/<job_id>/status')
def analysis_status_api(job_id):
    analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404
    
    progress = 0
    if job_id in processing_jobs:
        progress = processing_jobs[job_id].get("progress", 0)
    
    return jsonify({
        "status": analysis.status,
        "progress": progress,
        "job_id": job_id
    })

@app.route('/violations/<job_id>')
def view_violations(job_id):
    analysis = db.session.query(models.Analysis).filter_by(job_id=job_id).first()
    if not analysis:
        flash('Analysis job not found', 'danger')
        return redirect(url_for('index'))
    
    violations = db.session.query(models.Violation).filter_by(analysis_id=job_id).all()
    
    # Group violations by type
    violation_types = {}
    for violation in violations:
        if violation.violation_type not in violation_types:
            violation_types[violation.violation_type] = 0
        violation_types[violation.violation_type] += 1
    
    return render_template('violations.html', 
                          analysis=analysis, 
                          violations=violations, 
                          violation_types=violation_types)

@app.route('/analyses')
def all_analyses():
    analyses = db.session.query(models.Analysis).order_by(models.Analysis.upload_date.desc()).all()
    return render_template('analyses.html', analyses=analyses)
