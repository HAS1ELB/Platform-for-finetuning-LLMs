# Mini Cloud d'Entraînement Local

Plateforme locale de fine-tuning de modèles de langage avec architecture multi-agent CrewAI.

## Architecture

- **Backend**: FastAPI + CrewAI agents
- **Frontend**: Streamlit
- **ML**: Hugging Face Transformers + PEFT (LoRA)
- **Tracking**: MLflow
- **Database**: SQLite
- **Deployment**: Docker Compose

## Quick Start

```bash
docker-compose up --build
```

## Access

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

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

```
mini-cloud/
├── backend/          # FastAPI + CrewAI agents
├── frontend/         # Streamlit interface
├── data/             # Datasets and models
├── mlruns/           # MLflow experiments
└── docker-compose.yml
```
