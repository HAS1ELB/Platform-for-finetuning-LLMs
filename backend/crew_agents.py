from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, get_dataset_config_names
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
        data = self._load_dataset_with_fallback(dataset_name, split)
        self.data_agent.log(f"Dataset loaded: {len(data)} samples")
        return data
    
    def _load_dataset_with_fallback(self, dataset_name: str, split: str = "train"):
        """Load dataset with automatic config detection and fallback strategies."""
        
        # Common dataset name variations (try with and without organization prefix)
        variations = [dataset_name]
        
        # Add common organization prefixes if not already present
        if '/' not in dataset_name:
            common_orgs = ['stanfordnlp', 'rajpurkar', 'abisee', 'EdinburghNLP', 
                          'Samsung', 'Helsinki-NLP', 'fancyzhx', 'SetFit', 
                          'Salesforce', 'mandarjoshi', 'allenai', 'bigcode',
                          'HuggingFaceH4']
            for org in common_orgs:
                variations.append(f"{org}/{dataset_name}")
        
        # Try each variation
        last_error = None
        for variant in variations:
            try:
                # Strategy 1: Try loading without config (works for simple datasets)
                self.data_agent.log(f"Attempting to load: {variant}")
                return load_dataset(variant, split=split)
            except Exception as e1:
                last_error = e1
                self.data_agent.log(f"Direct load failed for {variant}: {str(e1)[:100]}")
                
                # Strategy 2: Try to get available configs and use the first one
                try:
                    configs = get_dataset_config_names(variant)
                    if configs:
                        config = configs[0]
                        self.data_agent.log(f"Found {len(configs)} configs for {variant}, using: {config}")
                        return load_dataset(variant, config, split=split)
                except Exception as e2:
                    self.data_agent.log(f"Config-based load failed for {variant}: {str(e2)[:100]}")
                
                # Strategy 3: Try loading the full dataset then accessing the split
                try:
                    full_dataset = load_dataset(variant)
                    if split in full_dataset:
                        return full_dataset[split]
                    elif "train" in full_dataset:
                        self.data_agent.log(f"Split '{split}' not found in {variant}, using 'train'")
                        return full_dataset["train"]
                    else:
                        # Return first available split
                        first_split = list(full_dataset.keys())[0]
                        self.data_agent.log(f"Using first available split for {variant}: {first_split}")
                        return full_dataset[first_split]
                except Exception as e3:
                    self.data_agent.log(f"Full dataset load failed for {variant}: {str(e3)[:100]}")
                    continue
        
        # If all strategies fail, raise helpful error
        error_msg = (
            f"Failed to load dataset '{dataset_name}'. "
            f"Tried variations: {', '.join(variations)}. "
            f"Please check: 1) Dataset name is correct, 2) Dataset exists on Hugging Face Hub, "
            f"3) You have internet access. "
            f"Last error: {str(last_error)[:200]}"
        )
        raise Exception(error_msg)
    
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
        dataset = self._load_dataset_with_fallback(dataset_name, "train")
        
        # For large datasets, take a subset for faster training
        if len(dataset) > 1000:
            self.data_agent.log(f"Dataset has {len(dataset)} samples - taking first 1000 for faster training")
            dataset = dataset.select(range(1000))
        
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
        # Automatically detect text column
        text_column = self._find_text_column(dataset)
        self.data_agent.log(f"Using text column: {text_column}")
        
        def tokenize_function(examples):
            texts = examples[text_column]
            # Handle case where text might be None or empty
            texts = [str(t) if t is not None else "" for t in texts]
            result = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=config["max_length"]
            )
            # For causal LM, labels should be the same as input_ids
            # Use list() to create a proper copy in batched mode
            result["labels"] = [ids[:] for ids in result["input_ids"]]
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names  # Remove original columns, keep only tokenized
        )
        
        training_args = TrainingArguments(
            output_dir=f"./data/models/{run_id}",
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
            dataloader_pin_memory=False  # Disable pin_memory on CPU
        )
        
        self.training_agent.log("Starting fine-tuning")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        
        try:
            self.training_agent.log("Starting training loop...")
            result = trainer.train()
            self.training_agent.log("Training completed successfully")
        except Exception as train_error:
            self.training_agent.log(f"Training failed: {str(train_error)}")
            raise
        
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
    
    def _find_text_column(self, dataset):
        """Automatically detect the text column in a dataset."""
        # Common text column names
        common_names = ["text", "content", "sentence", "document", "article", "body", "prompt"]
        
        column_names = dataset.column_names
        
        # First, check for common names
        for name in common_names:
            if name in column_names:
                return name
        
        # If not found, look for columns containing "text" in the name
        for col in column_names:
            if "text" in col.lower():
                return col
        
        # If still not found, use the first string column
        if column_names:
            return column_names[0]
        
        raise ValueError(f"Could not find a text column in dataset. Available columns: {column_names}")
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Validate a dataset and return its information."""
        try:
            self.data_agent.log(f"Validating dataset {dataset_name}")
            
            # Try to load the dataset
            dataset = self._load_dataset_with_fallback(dataset_name, "train")
            
            # Get column info
            columns = dataset.column_names
            text_column = self._find_text_column(dataset)
            
            # Get sample texts (first 3)
            sample_texts = []
            for i in range(min(3, len(dataset))):
                text = dataset[i][text_column]
                # Truncate long texts
                text_str = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
                sample_texts.append(text_str)
            
            self.data_agent.log(f"Dataset validation successful")
            
            return {
                "valid": True,
                "dataset_name": dataset_name,
                "config_used": None,  # Could be enhanced to track this
                "num_samples": len(dataset),
                "columns": columns,
                "text_column": text_column,
                "sample_texts": sample_texts,
                "error": None
            }
            
        except Exception as e:
            self.data_agent.log(f"Dataset validation failed: {str(e)}")
            return {
                "valid": False,
                "dataset_name": dataset_name,
                "config_used": None,
                "num_samples": None,
                "columns": None,
                "text_column": None,
                "sample_texts": None,
                "error": str(e)
            }
    
    def execute_training(self, config: Dict[str, Any], run_id: str):
        return self.fine_tune_model(config, run_id)
