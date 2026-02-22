from sqlalchemy import Column, Integer, String, Float, Text
from database.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True)
    username = Column(String)
    role = Column(String)
    country = Column(String)
    opportunity = Column(Float)
    probability = Column(Float)
    percentile = Column(Float)
