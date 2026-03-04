"""
Stage 1: Supervised Fine-Tuning (SFT)
Fine-tunes Qwen2.5-7B-Coder on ~500k curated (competition, solution) pairs.
Uses LoRA rank 64 with DeepSpeed ZeRO-3 across 18× A6000.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class SFTTrainingConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    output_dir: str = "./checkpoints/sft"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 16384

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # Auto-detected

    # Data paths (after synthesis)
    notebook_pairs: str = "./data/synthesized/notebook_pairs.jsonl"
    writeup_pairs: str = "./data/synthesized/writeup_pairs.jsonl"
    discussion_pairs: str = "./data/synthesized/discussion_pairs.jsonl"
    technique_pairs: str = "./data/synthesized/technique_pairs.jsonl"
    meta_pairs: str = "./data/synthesized/meta_competition_pairs.jsonl"

    # Logging
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    wandb_project: str = "podium-sft"


SYSTEM_PROMPT = """\
You are Podium, an expert Kaggle competition AI with grandmaster-level knowledge.
You have internalized patterns from millions of competition notebooks and thousands of
winning solution writeups. When given a competition brief, you provide expert analysis,
feature engineering strategies, model architectures, and complete working code.

Always structure your response as:
<think>[Your analysis and reasoning]</think>
<code>[Complete, executable Python code]</code>
"""


def format_training_example(example: dict) -> str:
    """Format a synthesized pair into a training message."""
    # Field name fallback chain covers all 5 synthesis streams:
    # notebook/writeup/discussion → problem_summary/competition, key_insight/reasoning, solution_code/code
    # technique_pairs → competition_scenario, expert_explanation, code_example, domain
    competition_context = example.get("problem_summary", example.get("competition", example.get("competition_scenario", "")))
    reasoning = example.get("key_insight", example.get("reasoning", example.get("expert_explanation", "")))
    code = example.get("solution_code", example.get("code", example.get("code_example", "")))
    metric = example.get("evaluation_metric", "")
    comp_type = example.get("competition_type", example.get("domain", ""))

    user_msg = f"""Competition: {competition_context}
Type: {comp_type}
Metric: {metric}

Analyze this competition and provide your approach with complete code."""

    assistant_msg = f"""<think>
{reasoning}
</think>

<code>
{code}
</code>"""

    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"


def load_all_training_data(config: SFTTrainingConfig) -> Dataset:
    """Load and combine all 5 training streams."""
    import json

    all_examples = []
    streams = [
        config.notebook_pairs,
        config.writeup_pairs,
        config.discussion_pairs,
        config.technique_pairs,
        config.meta_pairs,
    ]

    for stream_path in streams:
        if not Path(stream_path).exists():
            logger.warning(f"Stream not found: {stream_path}, skipping")
            continue
        with open(stream_path) as f:
            examples = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(examples)} examples from {stream_path}")
        all_examples.extend(examples)

    logger.info(f"Total training examples: {len(all_examples)}")

    formatted = [{"text": format_training_example(ex)} for ex in all_examples]
    return Dataset.from_list(formatted)


def train(config: SFTTrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_all_training_data(config)
    logger.info(f"Training on {len(dataset)} examples")

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_seq_length=config.max_seq_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed="training/configs/deepspeed_zero3.json",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="podium-sft",
        dataset_text_field="text",
        packing=False,  # Packing corrupts chat format (concatenates <|im_end|><|im_start|>)
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"SFT training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    import typer

    def main(
        base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct",
        output_dir: str = "./checkpoints/sft",
        epochs: int = 3,
    ):
        config = SFTTrainingConfig(
            base_model=base_model,
            output_dir=output_dir,
            num_train_epochs=epochs,
        )
        train(config)

    typer.run(main)
