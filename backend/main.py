from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from huggingface_hub import list_models, list_datasets
import mlflow
import uuid
import os

from database import init_db, get_db, User, Dataset, Training
from auth import hash_password, verify_password, create_access_token, get_current_user
from models import UserCreate, Token, DatasetCreate, DatasetResponse, TrainingConfig, TrainingResponse, TrainingStatus, DatasetValidationResponse
from crew_agents import MLCrewAgents
from curated_datasets import CURATED_DATASETS, get_all_datasets_flat, get_dataset_by_name, search_datasets

app = FastAPI(title="Mini Cloud Training")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

crew_agents = MLCrewAgents()

@app.on_event("startup")
def startup_event():
    init_db()
    db = next(get_db())
    if not db.query(User).filter(User.username == "admin").first():
        user = User(username="admin", hashed_password=hash_password("admin123"))
        db.add(user)
        db.commit()

@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    db_user = User(username=user.username, hashed_password=hash_password(user.password))
    db.add(db_user)
    db.commit()
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/models")
def get_models(search: str = "", limit: int = 20):
    models = list(list_models(search=search, limit=limit, sort="downloads", direction=-1))
    return [{"id": m.id, "downloads": m.downloads} for m in models]

@app.get("/datasets")
def get_datasets_list(search: str = "", limit: int = 20):
    datasets = list(list_datasets(search=search, limit=limit, sort="downloads", direction=-1))
    return [{"id": d.id, "downloads": d.downloads} for d in datasets]

@app.post("/datasets", response_model=DatasetResponse)
def create_dataset(dataset: DatasetCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if db.query(Dataset).filter(Dataset.name == dataset.name).first():
        raise HTTPException(status_code=400, detail="Dataset already exists")
    db_dataset = Dataset(name=dataset.name, source=dataset.source)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

@app.get("/datasets/saved")
def get_saved_datasets(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Dataset).all()

@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return {"message": "Dataset deleted"}

@app.post("/datasets/validate", response_model=DatasetValidationResponse)
def validate_dataset(dataset_name: str, user: User = Depends(get_current_user)):
    """Validate a dataset from Hugging Face before adding it."""
    result = crew_agents.validate_dataset(dataset_name)
    return result

@app.get("/datasets/curated")
def get_curated_datasets(user: User = Depends(get_current_user)):
    """Get the curated list of popular datasets organized by category."""
    return CURATED_DATASETS

@app.get("/datasets/curated/all")
def get_all_curated_datasets(user: User = Depends(get_current_user)):
    """Get a flat list of all curated datasets."""
    return get_all_datasets_flat()

@app.get("/datasets/curated/search")
def search_curated_datasets(q: str = "", user: User = Depends(get_current_user)):
    """Search curated datasets by name or description."""
    if not q:
        return get_all_datasets_flat()
    return search_datasets(q)

def run_training_task(training_id: int, config: dict, run_id: str, db_session):
    try:
        crew_agents.execute_training(config, run_id)
        training = db_session.query(Training).filter(Training.id == training_id).first()
        if training:
            training.status = "completed"
            db_session.commit()
    except Exception as e:
        training = db_session.query(Training).filter(Training.id == training_id).first()
        if training:
            training.status = f"failed: {str(e)}"
            db_session.commit()
    finally:
        db_session.close()

@app.post("/trainings", response_model=TrainingResponse)
def create_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    # Fix: Use mlflow.start_run which handles experiment creation
    mlflow.set_experiment("mini-cloud-training")
    run = mlflow.start_run(nested=True)
    run_id = run.info.run_id
    mlflow.end_run()  # End it immediately, will be used in background task
    
    db_training = Training(
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        status="running",
        mlflow_run_id=run_id
    )
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    
    from database import SessionLocal
    db_session = SessionLocal()
    background_tasks.add_task(run_training_task, db_training.id, config.dict(), run_id, db_session)
    
    return db_training

@app.get("/trainings")
def get_trainings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.query(Training).all()

@app.get("/trainings/{training_id}", response_model=TrainingStatus)
def get_training_status(training_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    training = db.query(Training).filter(Training.id == training_id).first()
    if not training:
        raise HTTPException(status_code=404, detail="Training not found")
    return training

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
