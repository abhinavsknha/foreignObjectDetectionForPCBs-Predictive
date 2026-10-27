from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

raw_url = os.getenv("DATABASE_URL", "postgresql+pg8000://postgres:postgres@localhost:5432/pcb_defect_db")
if raw_url.startswith("postgresql://"):
    raw_url = raw_url.replace("postgresql://", "postgresql+pg8000://", 1)

SQLALCHEMY_DATABASE_URL = raw_url

# Inspection records are a convenience, so a slow or dead database must never freeze inference:
# time out socket operations, test pooled connections before use, and recycle old ones.
_engine_args = {"pool_pre_ping": True, "pool_recycle": 240}
if SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
    _engine_args["connect_args"] = {"timeout": 15}
engine = create_engine(SQLALCHEMY_DATABASE_URL, **_engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
