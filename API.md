# API Endpoints

## Authentication

### POST /register
Register new user
```json
{"username": "user", "password": "pass"}
```

### POST /token
Login (OAuth2 form)
```
username=admin&password=admin123
```

## Hugging Face

### GET /models?search=gpt2&limit=20
Browse HF models

### GET /datasets?search=wikitext&limit=20
Browse HF datasets

## Datasets Management

### POST /datasets
```json
{"name": "wikitext", "source": "Hugging Face"}
```

### GET /datasets/saved
List saved datasets

### DELETE /datasets/{id}
Delete dataset

## Training

### POST /trainings
```json
{
  "model_name": "gpt2",
  "dataset_name": "wikitext",
  "learning_rate": 2e-5,
  "num_epochs": 3,
  "batch_size": 4,
  "max_length": 512
}
```

### GET /trainings
List all trainings

### GET /trainings/{id}
Get training status

## Health

### GET /health
```json
{"status": "ok"}
```

All endpoints (except /register, /token, /health) require JWT bearer token in Authorization header.
