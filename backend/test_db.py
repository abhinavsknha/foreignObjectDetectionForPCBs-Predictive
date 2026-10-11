from app.database import engine, Base
from app import models
from sqlalchemy import text
import time

def init_db():
    print("Testing Postgres Database Connection...")
    # Give it a second just in case
    time.sleep(2)
    try:
        # Create pgvector extension first
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            print("Successfully ensured pgvector extension exists.")
            
        # Then create tables
        Base.metadata.create_all(bind=engine)
        print("Successfully connected to Postgres and created tables (Users, Uploads, DetectionResults)!")
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__":
    init_db()
