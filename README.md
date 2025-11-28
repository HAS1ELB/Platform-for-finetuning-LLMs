# Mini Cloud d'Entraînement Local

Plateforme locale de fine-tuning de modèles de langage avec architecture multi-agent CrewAI.

## Architecture

- **Backend**: FastAPI + CrewAI agents
- **Frontend**: Streamlit
- **ML**: Hugging Face Transformers + PEFT (LoRA)
- **Tracking**: MLflow
- **Database**: SQLite
- **Deployment**: Docker Compose

## Quick Start 🚀

Start both the backend and frontend using Docker Compose:

```bash
docker compose up --build
```

## Troubleshooting ⚠️

If the backend reports a missing module such as `ModuleNotFoundError: No module named 'huggingface_hub'`, rebuild the backend image to ensure updated requirements are installed:

```powershell
docker-compose build --no-cache backend; docker-compose up -d backend
```

If you're running the frontend and backend together, rebuild all services:

```powershell
docker-compose up --build
```

If you see runtime errors like `ModuleNotFoundError: No module named 'mlflow'` when the backend starts, make sure `mlflow==3.6.0` is present in `backend/requirements.txt`. This project now validates that critical backend packages are installed at build time and will fail the build with a descriptive error if they're missing.

### PyTorch / CPU wheels ⚠️
The backend requires PyTorch for training. PyTorch CPU special wheels are listed in `requirements.txt` with an extra index URL to the PyTorch wheels repository:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
```

If `torch` fails to install during the top-level install step, Docker's fallback can leave the backend without `torch`, resulting in `ModuleNotFoundError: No module named 'torch'`. You can either:

- Add `torch==2.9.1+cpu`, `torchaudio==2.9.1+cpu`, and `torchvision==0.24.1+cpu` to `backend/requirements.txt` (this repository includes them now), or
- Avoid the fallback and let the build fail loudly if top-level installation fails.

If the PyTorch wheel is not available for your chosen Python version (e.g., Python 3.12) or architecture, try:

```powershell
docker-compose build --no-cache backend
docker-compose up -d backend
# Inspect logs and fix any reported package or wheel errors
docker-compose logs -f backend
```

### Training timeouts and MLflow connectivity
- Training jobs are now queued and executed in a background worker. The HTTP POST /trainings returns quickly and does not block while the model is being trained.
- If a training job is queued but fails quickly, check if the backend can reach MLflow; the backend provides `GET /mlflow/check` to verify the `MLFLOW_TRACKING_URI` for quick connectivity tests.
- If MLflow is unreachable from the backend, the background worker will mark the training as failed and log an explanatory message in the backend logs.

If you still encounter wheel compatibility issues, consider switching to a Python version that PyTorch supports or pin a PyTorch version compatible with your Python version.

### GPU / CUDA support ⚡

If you have a CUDA-capable GPU and want to build the backend with GPU-enabled PyTorch, this repo includes a `backend/requirements-gpu.txt` file with an example GPU wheel set (e.g., `+cu118`).

To build the backend image with GPU-enabled wheels:

```powershell
# Set BUILD_FOR_GPU=true in your environment or call directly on the command line
setx BUILD_FOR_GPU true
docker-compose build --no-cache backend
docker-compose up -d
```

Notes:
- Make sure your host system has the NVIDIA Docker runtime installed and enabled (NVIDIA Container Toolkit). Follow instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- The example GPU wheel files in `backend/requirements-gpu.txt` use PyTorch +cu118; if your host/CUDA version differs, update the `+cuXXX` part to match the supported wheel for your system.
- In certain environments, wheel names and compatibility vary by Python version; if the build fails due to incompatible wheel, try a different Python base image (3.11 or 3.10) or pin a different PyTorch + CUDA tag.

## Access

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **MLflow**: http://localhost:5555

### MLflow used externally
This deployment reduces responsibility for launching MLflow. The stack no longer includes an `mlflow` service in `docker-compose.yml`, so MLflow must be started externally and made reachable from the backend container.

- Example: If you have MLflow running on your host and accessible on port 5555, set the backend environment variable (in `.env` or via the shell) to:

```powershell
$env:MLFLOW_TRACKING_URI = "http://host.docker.internal:5555"
docker compose up --build -d
```

- If you prefer `http://localhost:5555`, you can run the backend container with host networking (Linux), or you can map the host address into the container using `host.docker.internal` on Docker Desktop in Windows. If you need guidance for your environment, tell me which OS you're using and I'll add exact commands.

Use the backend endpoint `GET /mlflow/check` to confirm the backend can reach your external MLflow server.

### Environment variables

- `HF_TOKEN`: (optional) place your Hugging Face token in an `.env` file or pass as an environment variable to the `backend` service if you want to use private models/datasets.

By default the frontend is configured to point to the backend service within Docker Compose. If you run services separately, you can set `API_URL` env var on the frontend container.

### Run individually for development

Backend dev server (from repository root):

```powershell
cd backend; py -m uvicorn main:app --reload
```

Frontend dev server (from repository root):

```powershell
cd frontend; streamlit run app.py
```

- **API Docs**: <http://localhost:8000/docs>

## Default Login

- Username: `admin`
- Password: `admin123`

## Features

- Browse Hugging Face models and datasets
- Configure and launch fine-tuning jobs
- Multi-agent orchestration with CrewAI
- Track experiments with MLflow
- JWT authentication
- LoRA-based efficient fine-tuning

## Project Structure

mini-cloud/
├── backend/          # FastAPI + CrewAI agents
├── frontend/         # Streamlit interface
├── data/             # Datasets and models
└── docker-compose.yml
