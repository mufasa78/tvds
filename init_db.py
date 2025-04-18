#!/usr/bin/env python
"""
Database initialization script for Traffic Violation Detection System.
This script ensures all database tables are created and properly migrated.
"""

import os
import sys
from app import app, db
import models  # Import models to register them with SQLAlchemy

def init_db():
    """Initialize the database and create all tables."""
    print("Initializing database...")
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Database tables created successfully!")
        
        # Print information about the database
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        
        print("\nDatabase tables:")
        for table_name in inspector.get_table_names():
            print(f"- {table_name}")
            columns = inspector.get_columns(table_name)
            for column in columns:
                print(f"  - {column['name']}: {column['type']}")
        
        print("\nDatabase initialization complete!")

if __name__ == "__main__":
    init_db()
