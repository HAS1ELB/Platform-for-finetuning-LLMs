from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from datetime import datetime
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
default_db = os.path.join(repo_root, 'data', 'minicloud.db')
DATABASE_URL = f"sqlite:///{os.getenv('DB_PATH', default_db)}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

db_dir = os.path.dirname(DATABASE_URL.split(':///')[1])
# Ensure the database directory exists (create at repo root data/ if not present)
os.makedirs(db_dir, exist_ok=True)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    # Relationship to trainings
    trainings = relationship('Training', back_populates='user')

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Training(Base):
    __tablename__ = "trainings"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    dataset_name = Column(String)
    status = Column(String)
    mlflow_run_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    # Relationship back to user
    user = relationship('User', back_populates='trainings')

    @property
    def username(self):
        return self.user.username if self.user else None

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
