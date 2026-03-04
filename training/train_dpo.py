"""
Stage 3: Direct Preference Optimization (DPO)
Trains Podium on competition strategy preferences:
- When to do more EDA vs. start modeling
- When to stop feature engineering and ensemble
- How to allocate time budget across competition phases
- Submission selection strategy (avoid public LB overfitting)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


@dataclass
class DPOTrainingConfig:
    model_name: str = "./checkpoints/rl"  # Load from Stage 2 RL checkpoint
    output_dir: str = "./checkpoints/dpo"

    # DPO
    beta: float = 0.1  # KL penalty coefficient
    learning_rate: float = 5e-7
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 8192
    max_prompt_length: int = 2048

    # Data
    preference_data_path: str = "./data/dpo/competition_preferences.jsonl"

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    wandb_project: str = "podium-dpo"


def load_preference_dataset(data_path: str) -> Dataset:
    """
    Load DPO preference pairs.
    Each example: {prompt, chosen, rejected}

    Preference pairs encode competition strategy wisdom:
    - chosen: grandmaster's actual decision + reasoning
    - rejected: naive/suboptimal decision
    """
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append({
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],   # Better competition strategy
                "rejected": ex["rejected"],  # Worse strategy
            })
    logger.info(f"Loaded {len(examples)} DPO preference pairs")
    return Dataset.from_list(examples)


def train(config: DPOTrainingConfig):
    logger.info(f"Loading RL checkpoint for DPO: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Reference model (frozen) for KL constraint
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = load_preference_dataset(config.preference_data_path)

    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        beta=config.beta,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="podium-dpo",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        args=dpo_config,
        train_dataset=dataset,
    )

    logger.info("Starting DPO training on competition strategy preferences...")
    trainer.train()
    trainer.save_model(config.output_dir)
    logger.info(f"DPO complete. Final model saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        model_name: str = "./checkpoints/rl",
        output_dir: str = "./checkpoints/dpo",
        data_path: str = "./data/dpo/competition_preferences.jsonl",
    ):
        config = DPOTrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            preference_data_path=data_path,
        )
        train(config)

    typer.run(main)
