#!/usr/bin/env python
"""
Database connection check script for Traffic Violation Detection System.
This script tests the database connection and reports any issues.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_database_url():
    """Get the database URL from environment variables or use the default."""
    return os.environ.get("DATABASE_URL", 
                         "postgresql://neondb_owner:npg_YpEs4U6ufeHl@ep-wandering-butterfly-a5ncpr5f-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require")

def check_database_connection():
    """Check if the database connection is working."""
    db_url = get_database_url()
    logger.info(f"Testing connection to database...")
    
    try:
        # Create an engine and connect to the database
        engine = create_engine(db_url)
        with engine.connect() as connection:
            # Execute a simple query to test the connection
            result = connection.execute(text("SELECT 1"))
            if result.scalar() == 1:
                logger.info("Database connection successful!")
                return True
            else:
                logger.error("Database connection test failed.")
                return False
    except SQLAlchemyError as e:
        logger.error(f"Database connection error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def check_database_tables():
    """Check if the required database tables exist."""
    from app import app, db
    import models
    
    with app.app_context():
        try:
            # Get all table names in the database
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            # Check if our required tables exist
            required_tables = ['analyses', 'violations']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.warning(f"Missing tables: {', '.join(missing_tables)}")
                return False
            else:
                logger.info(f"All required tables exist: {', '.join(required_tables)}")
                return True
        except Exception as e:
            logger.error(f"Error checking database tables: {str(e)}")
            return False

if __name__ == "__main__":
    logger.info("Checking database connection and tables...")
    
    # Check database connection
    if check_database_connection():
        # If connection is successful, check tables
        if check_database_tables():
            logger.info("Database is properly configured and ready to use.")
            sys.exit(0)
        else:
            logger.warning("Database connection is working, but tables are missing.")
            logger.info("Run 'python init_db.py' to create the required tables.")
            sys.exit(1)
    else:
        logger.error("Failed to connect to the database.")
        logger.info("Please check your DATABASE_URL environment variable or database server.")
        sys.exit(1)
