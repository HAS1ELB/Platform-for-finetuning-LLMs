
# Architecture Multi-Agent

## Vue d'ensemble

Le projet utilise une architecture multi-agent avec 4 agents spécialisés qui orchestrent le fine-tuning de modèles.

## Agents Spécialisés

### 1. Data Manager Agent

- **Rôle**: Gestion des datasets
- **Responsabilités**: Charger les datasets depuis Hugging Face, validation, préparation, tokenization

### 2. Training Specialist Agent

- **Rôle**: Fine-tuning des modèles
- **Responsabilités**: Configuration LoRA, chargement des modèles, exécution du training, sauvegarde

### 3. Evaluation Expert Agent

- **Rôle**: Évaluation des modèles
- **Responsabilités**: Calcul des métriques de performance, analyse des résultats

### 4. MLflow Monitor Agent

- **Rôle**: Suivi des expériences
- **Responsabilités**: Logging des métriques, suivi des runs, versioning des modèles

## Stack Technique

### Backend (FastAPI)

- **auth.py**: Authentification JWT avec bcrypt
- **database.py**: ORM SQLAlchemy avec SQLite
- **models.py**: Schémas Pydantic pour validation
- **crew_agents.py**: Orchestration CrewAI + fine-tuning PEFT/LoRA
- **main.py**: API REST endpoints

### Frontend (Streamlit)

- **app.py**: Interface utilisateur complète en une page
- Tabs: Models/Datasets, Gestion, Training, Expériences

### Fine-tuning

- **PEFT/LoRA**: Fine-tuning efficace avec faible empreinte mémoire
- **HuggingFace Transformers**: Support de tous les modèles HF
- **Datasets HF**: Accès direct aux datasets publics

### Tracking

- **MLflow**: Suivi complet des expériences, métriques, et modèles

## Flux de travail

1. **Authentification**: Login JWT → Token
2. **Browse HF**: Recherche models/datasets sur Hugging Face
3. **Add Dataset**: Sauvegarde référence dataset en DB
4. **Configure Training**: Sélection model + dataset + hyperparamètres
5. **Start Training**:
   - Création run MLflow
   - Crew agents orchestrent le fine-tuning
   - Background task asynchrone
   - Logging temps réel dans MLflow
6. **Monitor**: Suivi status training + métriques MLflow

## Déploiement

3 containers Docker:

- **backend**: FastAPI + CrewAI (port 8000)
- **frontend**: Streamlit (port 8501)
- **mlflow**: MLflow UI (port 5000)

Volumes persistants pour data et mlruns.

## Sécurité

- JWT avec expiration configurable
- Passwords hashés avec bcrypt
- API protégée par token bearer
- Données locales (pas d'exposition internet)
