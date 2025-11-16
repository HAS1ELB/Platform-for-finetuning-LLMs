from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DatasetCreate(BaseModel):
    name: str
    source: str

class DatasetResponse(BaseModel):
    id: int
    name: str
    source: str
    created_at: datetime

class TrainingConfig(BaseModel):
    model_name: str
    dataset_name: str
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 512

class TrainingResponse(BaseModel):
    id: int
    model_name: str
    dataset_name: str
    status: str
    mlflow_run_id: Optional[str]
    created_at: datetime

class TrainingStatus(BaseModel):
    id: int
    status: str
    mlflow_run_id: Optional[str]

class DatasetValidationResponse(BaseModel):
    valid: bool
    dataset_name: str
    config_used: Optional[str] = None
    num_samples: Optional[int] = None
    columns: Optional[list] = None
    text_column: Optional[str] = None
    sample_texts: Optional[list] = None
    error: Optional[str] = None
