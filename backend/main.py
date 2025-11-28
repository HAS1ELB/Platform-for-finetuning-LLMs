from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from huggingface_hub import list_models, list_datasets, login
# Standard imports
try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    # Print a helpful message to the logs to make it clear why the app failed.
    # This is helpful when the Dockerfile uses a fallback install that omits heavy root
    # requirements and the backend runs without essential deps.
    print("⚠️  'mlflow' is not installed. Please ensure 'mlflow==3.6.0' is listed in 'backend/requirements.txt' and rebuild the backend image (docker-compose build --no-cache backend)")
    raise
import time
import os
from dotenv import load_dotenv
import requests
# Ensure transformers does not try to import TF. This helps avoid TensorFlow-related import errors if TF is present.
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
from fastapi.responses import JSONResponse
from fastapi import Response
from database import init_db, get_db, SessionLocal, User, Dataset, Training, engine
from sqlalchemy import inspect
from auth import hash_password, verify_password, create_access_token, get_current_user
from models import UserCreate, Token, DatasetCreate, DatasetResponse, TrainingConfig, TrainingResponse, TrainingStatus, DatasetValidationResponse
from crew_agents import MLCrewAgents
from curated_datasets import CURATED_DATASETS, get_all_datasets_flat, get_dataset_by_name, search_datasets


# Load environment variables
load_dotenv()
app = FastAPI(title="Mini Cloud Training")

frontend_origin = os.getenv('FRONTEND_ORIGIN', 'http://localhost:8501')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

crew_agents = MLCrewAgents()


@app.get("/train_env")
def train_env():
    """Return trainer availability for the current runtime environment.
    This helps the frontend display a helpful message and prevents starting
    training if Trainer is not available."""
    # Provide a friendly diagnostic message to help users enable training
    if crew_agents.trainer_available:
        return {"trainer_available": True, "message": "Trainer available"}
    else:
        return {
            "trainer_available": False,
            "message": (
                "Trainer not available. To enable training, install required packages: 'transformers', 'torch', 'datasets', 'accelerate', 'peft'.\n"
                "You can run: pip install -r requirements.backend.txt and restart the server."
            )
        }

@app.on_event("startup")
def startup_event():
    init_db()
    db = next(get_db())
    if not db.query(User).filter(User.username == "admin").first():
        user = User(username="admin", hashed_password=hash_password("admin123"))
        db.add(user)
        db.commit()
    
    # Authenticate with Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(hf_token)
            print("✅ Hugging Face authentication successful")
        except Exception as e:
            print(f"⚠️  Hugging Face authentication failed: {e}")
    else:
        print("⚠️  Hugging Face token not provided. Skipping authentication.")
    # Migration: ensure `user_id` column exists in trainings table (for simpler dev migrations)
    try:
        insp = inspect(engine)
        cols = [c['name'] for c in insp.get_columns('trainings')]
        if 'user_id' not in cols:
            print("⚠️  'user_id' column missing in 'trainings' table; attempting to add it.")
            # SQLite allows adding columns via simple ALTER TABLE
            with engine.connect() as conn:
                conn.execute("ALTER TABLE trainings ADD COLUMN user_id INTEGER")
                conn.commit()
            # Try to set existing trainings to admin user if present (development convenience migration)
            try:
                admin_user = db.query(User).filter(User.username == "admin").first()
                if admin_user:
                    with engine.connect() as conn:
                        conn.execute(f"UPDATE trainings SET user_id = {admin_user.id} WHERE user_id IS NULL")
                        conn.commit()
                    print("✅ Existing trainings assigned to admin user.")
            except Exception:
                print("⚠️  Could not assign existing trainings to admin user automatically")
            print("✅ 'user_id' column added to 'trainings'.")
    except Exception as e:
        print(f"⚠️  Migration check/alter failed: {e}")

    # Move any existing backend-local 'data' or 'mlruns' into root project directories
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        backend_data = os.path.join(os.path.dirname(__file__), 'data')
        repo_data = os.path.join(repo_root, 'data')
        backend_mlruns = os.path.join(os.path.dirname(__file__), 'mlruns')
        repo_mlruns = os.path.join(repo_root, 'mlruns')

        if os.path.exists(backend_data) and not os.path.exists(repo_data):
            print("⚠️  Moving backend data folder to repository root data/")
            os.rename(backend_data, repo_data)
        if os.path.exists(backend_mlruns) and not os.path.exists(repo_mlruns):
            print("⚠️  Moving backend mlruns folder to repository root mlruns/")
            os.rename(backend_mlruns, repo_mlruns)
    except Exception as e:
        print(f"⚠️  Failed moving data/mlruns to repo root: {e}")

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

# Middleware for managing cookies
@app.post("/login", response_model=Token)
def login_with_cookies(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.username})

    # Set the token in an HTTP-only cookie
    # Return token both in response body and as a secure, httpOnly cookie.
    response = JSONResponse(content={"access_token": token, "token_type": "bearer"})
    # Use secure cookies in production, but allow local dev with insecure cookies
    cookie_secure = True if os.getenv('ENV', 'dev') == 'production' else False
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,  # Prevent JavaScript access
        secure=cookie_secure,
        samesite=("Strict" if cookie_secure else "Lax"),
        max_age=60 * 60 * 24 * 7,  # 1 week
        path='/'
    )
    return response


@app.post("/session/set", response_model=Token)
def set_session_from_token(token: Token, resp: Response):
    """Set the authentication cookie from an access token.
    This allows a client-side script to pass the access token back and have the server
    set the secure, httpOnly cookie without exposing it to JavaScript.
    """
    cookie_secure = True if os.getenv('ENV', 'dev') == 'production' else False
    response = JSONResponse(content={"access_token": token.access_token, "token_type": token.token_type})
    response.set_cookie(
        key="auth_token",
        value=token.access_token,
        httponly=True,
        secure=cookie_secure,
        samesite=("Strict" if cookie_secure else "Lax"),
        max_age=60 * 60 * 24 * 7,
        path='/'
    )
    return response

@app.post("/logout")
def logout(response: Response):
    # Clear the authentication cookie
    response = JSONResponse(content={"message": "Logout successful"})
    response.delete_cookie("auth_token")
    return response

@app.get("/session")
def get_session(user: User = Depends(get_current_user)):
    return {"username": user.username, "message": "Session is active"}

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

def run_training_task(training_id: int, config: dict, db_session):
    try:
        # Re-check trainer availability before running in the background; if missing,
        # set training status to failed with a helpful message and exit.
        if not crew_agents.trainer_available:
            training = db_session.query(Training).filter(Training.id == training_id).first()
            if training:
                training.status = "failed: Trainer not available in this environment"
                db_session.commit()
            db_session.close()
            return
        # Quick check MLflow availability before attempting to create runs; fail fast
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        if mlflow_uri:
            try:
                requests.get(mlflow_uri, timeout=3)
            except Exception as e:
                training = db_session.query(Training).filter(Training.id == training_id).first()
                if training:
                    training.status = f"failed: MLflow not reachable: {str(e)}"
                    db_session.commit()
                db_session.close()
                return

        # Create mlflow experiment/run inside background worker to avoid blocking client requests
        client = MlflowClient()
        exp_name = "mini-cloud-training"
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                client.create_experiment(exp_name)
            elif exp.lifecycle_stage == 'deleted':
                try:
                    client.restore_experiment(exp.experiment_id)
                except Exception:
                    exp_name = f"{exp_name}-{int(time.time())}"
                    client.create_experiment(exp_name)
        except Exception:
            pass
        mlflow.set_experiment(exp_name)
        run = mlflow.start_run(nested=True)
        run_id = run.info.run_id
        mlflow.end_run()
        # Update training entry with run id and mark as running
        training = db_session.query(Training).filter(Training.id == training_id).first()
        if training:
            training.mlflow_run_id = run_id
            training.status = "running"
            db_session.commit()

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
    # Check for trainer availability to avoid creating a job that will fail
    if not crew_agents.trainer_available:
        raise HTTPException(status_code=422, detail=(
            "Trainer dependencies are missing in this environment: "
            "install 'transformers', 'torch', and optional training extras to enable training."
        ))

    # Do not attempt to contact MLflow synchronously here - perform MLflow operations
    # inside the background worker so the HTTP request returns immediately.
    
    db_training = Training(
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        status="queued",
        mlflow_run_id=None
        , user_id=user.id
    )
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    
    from database import SessionLocal
    db_session = SessionLocal()
    # Schedule background work that will create the MLflow run and execute the training
    background_tasks.add_task(run_training_task, db_training.id, config.dict(), db_session)
    
    return db_training

@app.get("/trainings")
def get_trainings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # Return trainings only for the authenticated user
    return db.query(Training).filter(Training.user_id == user.id).all()

@app.get("/trainings/{training_id}", response_model=TrainingStatus)
def get_training_status(training_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    training = db.query(Training).filter(Training.id == training_id, Training.user_id == user.id).first()
    if not training:
        raise HTTPException(status_code=404, detail="Training not found")
    return training

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/mlflow/check")
def mlflow_check():
    """Quickly check whether the MLflow tracking URI is reachable from the backend container."""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_uri:
        return {"reachable": False, "message": "MLFLOW_TRACKING_URI not set"}
    try:
        res = requests.get(mlflow_uri, timeout=2)
        return {"reachable": res.status_code == 200, "status_code": res.status_code, "uri": mlflow_uri}
    except Exception as e:
        return {"reachable": False, "message": str(e), "uri": mlflow_uri}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
