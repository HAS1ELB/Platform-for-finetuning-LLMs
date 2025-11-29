# Mini Cloud d'Entraînement Local

Plateforme locale de fine-tuning de modèles de langage avec architecture multi-agent CrewAI.

## Architecture

- **Backend**: FastAPI + CrewAI agents
- **Frontend**: Streamlit
- **ML**: Hugging Face Transformers + PEFT (LoRA)
- **Tracking**: MLflow (containerized)
- **Database**: SQLite
- **Deployment**: Docker Compose

## Quick Start 🚀

Start all services (backend, frontend, and MLflow) using Docker Compose:

```bash
docker compose up --build
```

This will start:

- Backend API on port 8000
- Frontend UI on port 8501
- MLflow tracking server on port 5555

## Access Points

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5555

## Default Login

- Username: `admin`
- Password: `admin123`

## Features

### Core Functionality

- **Model Fine-tuning**: Browse and fine-tune Hugging Face models with LoRA
- **Dataset Management**:
  - Browse Hugging Face datasets
  - Curated dataset collection organized by category
  - Dataset validation before training
- **Multi-agent Orchestration**: CrewAI agents for training workflow
- **Experiment Tracking**: MLflow integration for tracking runs and metrics
- **Background Training**: Non-blocking training jobs with queue system

### Authentication & Security

- JWT-based authentication
- HTTP-only cookie support for secure sessions
- User-specific training history
- Password hashing with bcrypt (pinned to 3.2.2 for compatibility)

### Developer Features

- Health check endpoints
- MLflow connectivity verification (`GET /mlflow/check`)
- Trainer availability check (`GET /train_env`)
- Automatic database migrations

## Project Structure

```
mini-cloud/
├── backend/              # FastAPI application
│   ├── main.py          # API endpoints and routing
│   ├── auth.py          # Authentication logic
│   ├── crew_agents.py   # CrewAI multi-agent system
│   ├── curated_datasets.py  # Curated dataset catalog
│   ├── database.py      # SQLAlchemy models
│   ├── models.py        # Pydantic schemas
│   ├── Dockerfile       # Backend container
│   └── requirements.txt # Python dependencies
├── frontend/            # Streamlit interface
│   ├── app.py          # Main UI application
│   ├── Dockerfile      # Frontend container
│   └── requirements.txt
├── data/               # Persistent data (SQLite DB, models)
├── mlruns/             # MLflow artifacts and runs
├── docker-compose.yml  # Multi-service orchestration
└── README.md
```

## Troubleshooting

### Bcrypt Compatibility Issue ⚠️

The project pins `bcrypt==3.2.2` in `backend/requirements.txt` to avoid compatibility issues with `passlib`. If you encounter errors like:

```
ValueError: Invalid salt
AttributeError: module 'bcrypt' has no attribute '__about__'
```

Ensure the backend is rebuilt with the pinned version:

```powershell
docker-compose build --no-cache backend
docker-compose up -d
```

### Missing Dependencies

If the backend reports missing modules (e.g., `ModuleNotFoundError: No module named 'mlflow'`):

```powershell
docker-compose build --no-cache backend
docker-compose up -d backend
docker-compose logs -f backend
```

The backend validates critical packages at startup and will provide helpful error messages.

### PyTorch Installation

The backend requires PyTorch for training. If you encounter `ModuleNotFoundError: No module named 'torch'`:

1. Verify `torch` is in `backend/requirements.txt`
2. Rebuild the backend image:

```powershell
docker-compose build --no-cache backend
docker-compose up -d backend
```

### MLflow Connectivity

Training jobs require MLflow to be accessible. The backend connects to MLflow at `http://mlflow:5000` (internal Docker network).

To verify connectivity:

```bash
curl http://localhost:8000/mlflow/check
```

If MLflow is unreachable, check:

- MLflow service is running: `docker-compose ps mlflow`
- MLflow logs: `docker-compose logs mlflow`
- Backend can reach MLflow: Check backend logs for connection errors

### Training Jobs

Training jobs run in the background and won't block API requests. To monitor training:

1. Check training status via API: `GET /trainings/{training_id}`
2. View MLflow UI: http://localhost:5555
3. Check backend logs: `docker-compose logs -f backend`

If training fails immediately:

- Verify Trainer dependencies are available (`GET /train_env`)
- Check MLflow connectivity (`GET /mlflow/check`)
- Review backend logs for detailed error messages

## Development

### Run Services Individually

**Backend** (from repository root):

```powershell
cd backend
py -m uvicorn main:app --reload
```

**Frontend** (from repository root):

```powershell
cd frontend
streamlit run app.py
```

**MLflow** (from repository root):

```powershell
mlflow server --host 0.0.0.0 --port 5555 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
MLFLOW_TRACKING_URI=http://mlflow:5000
DB_PATH=/app/data/minicloud.db
```

- `HF_TOKEN`: Optional, required for private Hugging Face models/datasets
- `MLFLOW_TRACKING_URI`: MLflow server URL (default: `http://mlflow:5000`)
- `DB_PATH`: SQLite database path (default: `/app/data/minicloud.db`)

## Technical Details

### Background Training System

Training jobs are executed asynchronously:

1. Client submits training config via `POST /trainings`
2. Backend creates training record with status "queued"
3. Background worker picks up the job and:
   - Creates MLflow run
   - Updates status to "running"
   - Executes training with CrewAI agents
   - Updates status to "completed" or "failed"

### Database Schema

- **users**: User accounts with hashed passwords
- **datasets**: Saved dataset references
- **trainings**: Training job records with status and MLflow run IDs

### Authentication Flow

1. User logs in via `POST /login`
2. Backend returns JWT token and sets HTTP-only cookie
3. Subsequent requests include token in Authorization header or cookie
4. Backend validates token and retrieves user context

## Known Issues

- **Python 3.12 Compatibility**: Some PyTorch wheels may not be available for Python 3.12. Consider using Python 3.11 if you encounter installation issues.
- **Windows Docker**: Use `host.docker.internal` instead of `localhost` when services need to communicate across containers.

## Future Enhancements

- [ ] Support for distributed training
- [ ] Model deployment endpoints
- [ ] Advanced hyperparameter tuning
- [ ] Multi-GPU support
- [ ] Real-time training progress streaming
