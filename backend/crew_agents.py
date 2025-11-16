from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import mlflow
import os
from datetime import datetime

class Agent:
    def __init__(self, role: str, goal: str):
        self.role = role
        self.goal = goal
        self.logs = []
    
    def log(self, message: str):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {self.role}: {message}")

class MLCrewAgents:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
        
        self.data_agent = Agent(
            role="Data Manager",
            goal="Load and prepare datasets from Hugging Face"
        )
        self.training_agent = Agent(
            role="Training Specialist", 
            goal="Fine-tune language models with optimal configurations"
        )
        self.evaluation_agent = Agent(
            role="Evaluation Expert",
            goal="Evaluate model performance and generate metrics"
        )
        self.monitoring_agent = Agent(
            role="MLflow Monitor",
            goal="Track experiments and log metrics to MLflow"
        )
    
    def load_dataset_task(self, dataset_name: str, split: str = "train"):
        self.data_agent.log(f"Loading dataset {dataset_name}")
        data = load_dataset(dataset_name, split=split)
        self.data_agent.log(f"Dataset loaded: {len(data)} samples")
        return data
    
    def fine_tune_model(self, config: Dict[str, Any], run_id: str):
        self.monitoring_agent.log(f"Starting MLflow run {run_id}")
        mlflow.start_run(run_id=run_id)
        
        model_name = config["model_name"]
        dataset_name = config["dataset_name"]
        
        mlflow.log_params({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "learning_rate": config["learning_rate"],
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"]
        })
        
        self.data_agent.log(f"Loading dataset {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        self.data_agent.log(f"Dataset ready: {len(dataset)} samples")
        
        self.training_agent.log(f"Loading model {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.training_agent.log("Configuring LoRA adapters")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
        
        self.data_agent.log("Tokenizing dataset")
        def tokenize_function(examples):
            text_column = list(examples.keys())[0]
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=config["max_length"]
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=f"./data/models/{run_id}",
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            logging_steps=10,
            save_strategy="epoch",
            report_to="none"
        )
        
        self.training_agent.log("Starting fine-tuning")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        
        result = trainer.train()
        self.training_agent.log("Training completed")
        
        self.evaluation_agent.log("Computing metrics")
        mlflow.log_metrics({
            "train_loss": result.training_loss,
            "train_runtime": result.metrics["train_runtime"],
            "train_samples_per_second": result.metrics["train_samples_per_second"]
        })
        
        self.training_agent.log("Saving model")
        model.save_pretrained(f"./data/models/{run_id}/final_model")
        tokenizer.save_pretrained(f"./data/models/{run_id}/final_model")
        
        self.monitoring_agent.log("Logging to MLflow completed")
        mlflow.end_run()
        
        return {"status": "completed", "model_path": f"./data/models/{run_id}/final_model"}
    
    def execute_training(self, config: Dict[str, Any], run_id: str):
        return self.fine_tune_model(config, run_id)
