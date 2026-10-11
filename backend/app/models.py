from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    uploads = relationship("Upload", back_populates="owner")

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    s3_key = Column(String, unique=True, index=True)
    s3_url = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Store the 512-d (or appropriate) vector embedding from RT-DETR
    embedding = Column(Vector(512)) 
    
    owner = relationship("User", back_populates="uploads")
    detections = relationship("DetectionResult", back_populates="upload")

class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"))
    
    # Bounding boxes stored as JSON (e.g. [{class: 'scratch', x: 0.1, y: 0.2, w: 0.05, h: 0.05, conf: 0.9}])
    boxes = Column(JSON) 
    
    # Track if this result has been used for incremental training
    used_for_training = Column(Integer, default=0) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    upload = relationship("Upload", back_populates="detections")
