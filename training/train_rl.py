"""
Stage 2: CV-Verified Reinforcement Learning (GRPO)
The core technical novelty of Podium.

Uses cross-validation score improvement as the reward signal — the same
"free verifiable reward" insight as DeepSeek-R1, applied to Kaggle competition code.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running directly: python training/train_rl.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset, load_from_disk
from loguru import logger
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from validation.validate_cv import batch_score


@dataclass
class RLTrainingConfig:
    # Model — base model used to load the SFT LoRA adapter
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    model_name: str = "./checkpoints/sft"  # Path to LoRA adapter from Stage 1
    output_dir: str = "./checkpoints/rl"

    # GRPO
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 8  # Sample 8 solutions per prompt, reward best

    # Reward
    max_execution_workers: int = 4  # Parallel Docker execution workers

    # LoRA (inherited from SFT, continue training)
    lora_r: int = 64
    lora_alpha: int = 128

    # Data
    train_data_path: str = "./data/rl/competition_execution_tasks.jsonl"

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    wandb_project: str = "podium-rl"


def build_reward_function(execution_config: dict):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    Executes generated code, runs CV, returns reward signal.

    TRL calls: reward_fn(prompts=prompts, completions=completions, **dataset_cols)
    """
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """
        prompts: list of competition prompts (contain data_path, target, metric)
        completions: list of generated code strings
        kwargs: extra dataset columns — metadata is a per-sample list of dicts
        """
        if len(completions) != len(prompts):
            raise ValueError(
                f"completions/prompts length mismatch: {len(completions)} != {len(prompts)}. "
                "TRL must pass matching lists to the reward function."
            )
        metadata_list = kwargs.get("metadata", [{} for _ in range(len(completions))])
        rewards = []

        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            # Parse execution config from per-sample metadata
            # Use modulo so metadata is reused if TRL doesn't replicate dataset
            # columns for each of the num_generations completions per prompt
            # TODO (PO-2): The modulo guard silently reuses metadata when TRL does not
            # replicate dataset columns for each of the num_generations completions per prompt.
            # If len(metadata_list) != len(completions), rewards may be assigned to wrong tasks.
            # Ideally, TRL should replicate metadata; investigate if reward quality seems off.
            meta = metadata_list[i % len(metadata_list)] if metadata_list else {}
            data_path = meta.get("data_path", execution_config.get("default_data_path"))
            target = meta.get("target_column", "target")
            metric = meta.get("metric", "auc")
            baseline_cv = meta.get("baseline_cv", 0.5)
            metric_dir = meta.get("metric_direction", "higher_is_better")

            # Execute single completion
            results = batch_score(
                [completion],
                data_path=data_path,
                target_column=target,
                metric=metric,
                baseline_cv=baseline_cv,
                metric_direction=metric_dir,
                max_workers=1,
            )
            rewards.append(results[0].reward)

        return rewards

    return reward_fn


def load_rl_dataset(data_path: str) -> Dataset:
    """
    Load competition execution tasks for RL training.
    Each example: {prompt, data_path, target_column, metric, baseline_cv}
    """
    import json
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            # Format prompt for model
            ex["prompt"] = format_rl_prompt(ex)
            # Wrap execution config fields into a "metadata" column so TRL
            # passes them to the reward_fn as kwargs["metadata"] (list of dicts)
            ex["metadata"] = {
                "data_path": ex.get("data_path"),
                "target_column": ex.get("target_column"),
                "metric": ex.get("metric"),
                "baseline_cv": ex.get("baseline_cv"),
                "metric_direction": ex.get("metric_direction", "higher_is_better"),
            }
            examples.append(ex)
    return Dataset.from_list(examples)


def format_rl_prompt(example: dict) -> str:
    """Format a competition task as a prompt for RL training."""
    return f"""<competition>
Type: {example.get('competition_type', 'tabular')}
Metric: {example.get('metric', 'auc')}
Target: {example.get('target_column', 'target')}
Data: {example.get('data_description', 'Standard tabular dataset')}
Current baseline CV: {example.get('baseline_cv', 0.5):.4f}

Task: Generate Python code to improve the cross-validation score above the baseline.
The code should load data from /data/train.csv and set the variable `cv_score`.
</competition>

<think>"""


def train(config: RLTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading SFT LoRA adapter from: {config.model_name}")
    model = PeftModel.from_pretrained(base_model, config.model_name, is_trainable=True)
    model.enable_input_require_grads()  # Required for PEFT + gradient_checkpointing

    logger.info("Loading RL training dataset...")
    dataset = load_rl_dataset(config.train_data_path)
    logger.info(f"RL dataset: {len(dataset)} competition tasks")

    reward_fn = build_reward_function({"default_data_path": "./data/rl/datasets"})

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="podium-rl-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,              # Fixed: was config=, should be args=
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    logger.info("Starting GRPO training with CV execution reward...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"RL training complete. Checkpoint saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        model_name: str = "./checkpoints/sft",
        output_dir: str = "./checkpoints/rl",
        data_path: str = "./data/rl/competition_execution_tasks.jsonl",
        num_generations: int = 8,
    ):
        config = RLTrainingConfig(
            base_model=base_model,
            model_name=model_name,
            output_dir=output_dir,
            train_data_path=data_path,
            num_generations=num_generations,
        )
        train(config)

    typer.run(main)
