from app import db
from datetime import datetime

class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), unique=True, nullable=False)
    filename = db.Column(db.String(256), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="pending")  # pending, processing, completed, failed
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    completion_date = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # Relationship with violations
    violations = db.relationship('Violation', backref='analysis', lazy=True)
    
    def __repr__(self):
        return f'<Analysis {self.job_id}>'

class Violation(db.Model):
    __tablename__ = 'violations'
    
    id = db.Column(db.Integer, primary_key=True)
    violation_id = db.Column(db.String(36), unique=True, nullable=False)
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.job_id'), nullable=False)
    violation_type = db.Column(db.String(50), nullable=False)  # lane, license_plate, line_crossing, not_yielding
    timestamp = db.Column(db.Float, nullable=False)  # Time in seconds from video start
    frame_number = db.Column(db.Integer, nullable=False)
    screenshot_path = db.Column(db.String(256), nullable=True)
    license_plate = db.Column(db.String(20), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Violation {self.violation_id} - {self.violation_type}>'
