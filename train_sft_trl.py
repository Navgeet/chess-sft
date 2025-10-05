#!/usr/bin/env python3
"""
TRL-based SFT training script for chess move generation.
Uses data.jsonl and applies chat templates.
Based on Chess_SFT_Training.ipynb notebook.
"""

import json
import os
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, Dataset
import pandas as pd

# Try to import Unsloth for optimization
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    UNSLOTH_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "unsloth/Qwen3-0.6B"
OUTPUT_DIR = "./output"
DATA_FILE = "./data.jsonl"

# Training hyperparameters
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3  # SFT typically needs multiple epochs
MAX_STEPS = -1  # Set to -1 to use epochs instead of max_steps
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.001
LR_SCHEDULER_TYPE = "linear"
LOGGING_STEPS = 10

# LoRA Configuration
USE_LORA = True  # Set to False for full fine-tuning
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_BIAS = "none"
USE_RSLORA = False


def load_dataset_from_jsonl(data_path: str) -> Dataset:
    """Load dataset from JSONL file."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file {data_path} not found")

    logger.info(f"Loading data from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def format_chat_template(example: dict, tokenizer) -> dict:
    """
    Format training example into chat template format.
    Uses tokenizer.apply_chat_template for proper formatting.
    """
    prompt = example.get("prompt", "")
    output = example.get("output", "")

    # Create messages in chat format
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output}
    ]

    # Apply the tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def setup_model_and_tokenizer(base_model: Optional[str] = None, use_unsloth: bool = True, use_lora: bool = True):
    """Load model and tokenizer with optional Unsloth optimization.

    Args:
        base_model: Optional path to a fine-tuned model or LoRA adapters to load as base.
                   If None, uses MODEL_NAME from config.
        use_unsloth: Whether to use Unsloth if available.
        use_lora: Whether to apply LoRA adapters. If False, performs full fine-tuning.
    """
    model_name = base_model or MODEL_NAME
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if loading a fine-tuned model (merged or with LoRA adapters)
    is_fine_tuned_path = base_model and Path(base_model).exists()

    # Check if it's a LoRA adapter directory (has adapter_config.json)
    is_lora_adapters = is_fine_tuned_path and (Path(base_model) / "adapter_config.json").exists()

    # Check if it's a merged model (no adapter_config.json but has model files)
    is_merged_model = is_fine_tuned_path and not is_lora_adapters

    if is_fine_tuned_path:
        if is_merged_model:
            # Load merged model and optionally apply fresh LoRA
            logger.info(f"Detected merged model at {base_model}")

            if UNSLOTH_AVAILABLE and use_unsloth:
                # Load merged model with Unsloth
                logger.info("Loading merged model with Unsloth...")
                # Full fine-tuning requires non-quantized model, LoRA can use quantization
                load_in_4bit = torch.cuda.is_available() and use_lora
                if not use_lora:
                    logger.warning("Full fine-tuning requires non-quantized model. Loading without 4-bit quantization (requires more GPU memory)")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    load_in_4bit=load_in_4bit,
                    use_gradient_checkpointing = "unsloth",
                    full_finetuning = True
                )

                if use_lora:
                    # Apply fresh LoRA with Unsloth
                    logger.info("Applying fresh LoRA adapters with Unsloth...")
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=LORA_R,
                        lora_alpha=LORA_ALPHA,
                        lora_dropout=LORA_DROPOUT,
                        bias=LORA_BIAS,
                        use_gradient_checkpointing="unsloth",
                        use_rslora=USE_RSLORA,
                    )
                    logger.info("Merged model with fresh LoRA loaded via Unsloth")
                else:
                    # Full fine-tuning mode
                    logger.info("Full fine-tuning mode (no LoRA adapters)")
                    model = FastLanguageModel.for_training(model)

                # Log trainable params
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
                logger.info(f"Trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_pct:.4f}")
            else:
                logger.error("Unsloth not available for merged model training")
                raise RuntimeError("Unsloth is required for merged model training")
        else:
            # This is a LoRA adapter directory - load base model and merge adapters first
            logger.info(f"Detected LoRA adapters at {base_model}")
            logger.info(f"Please use merge_lora.py to merge the adapters first:")
            logger.info(f"  python merge_lora.py --lora-path {base_model} --output-dir chess-sft-qwen3-0.6b-merged")
            logger.info(f"Then run training with: --base-model chess-sft-qwen3-0.6b-merged")
            raise RuntimeError(f"LoRA adapters at {base_model} need to be merged first. Use merge_lora.py script.")

    elif UNSLOTH_AVAILABLE and use_unsloth and not base_model:
        logger.info(f"Unsloth available, using optimized loading for base model...")
        logger.info("Using Unsloth for optimized training...")
        # Full fine-tuning requires non-quantized model, LoRA can use quantization
        load_in_4bit = torch.cuda.is_available() and use_lora
        if not use_lora:
            logger.warning("Full fine-tuning requires non-quantized model. Loading without 4-bit quantization (requires more GPU memory)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing = "unsloth",
            full_finetuning = True

        )

        if use_lora:
            # Setup LoRA with Unsloth
            logger.info("Applying LoRA adapters with Unsloth...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias=LORA_BIAS,
                use_gradient_checkpointing="unsloth",
                use_rslora=USE_RSLORA,
            )
            logger.info("Model loaded with Unsloth optimization and LoRA")
        else:
            # Full fine-tuning mode
            logger.info("Full fine-tuning mode (no LoRA adapters)")
            model = FastLanguageModel.for_training(model)
    else:
        logger.info("Loading standard model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )

    # Print trainable parameters if method is available
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        # Calculate trainable params manually
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
        logger.info(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_pct:.4f}")

    return model, tokenizer


def create_sft_config() -> SFTConfig:
    """Create TRL SFTConfig with all hyperparameters."""
    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,  # Set to -1 to use epochs
        warmup_steps=WARMUP_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",  # Save at regular intervals
        save_steps=500,  # Save checkpoint every 500 steps (~1.5 hours)
        save_total_limit=3,  # Keep only the 3 most recent checkpoints to save disk space
        logging_strategy="steps",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataset_text_field="text",
        packing=False,
        max_grad_norm=1.0,  # Explicitly set gradient clipping
        #report_to=["tensorboard"],
    )
    return config


def train_with_trl(base_model: Optional[str] = None, use_lora: bool = True, resume_from_checkpoint: Optional[str] = None):
    """Train model using TRL's SFTTrainer.

    Args:
        base_model: Optional path to a fine-tuned model or LoRA adapters to use as base.
        use_lora: Whether to apply LoRA adapters. If False, performs full fine-tuning.
        resume_from_checkpoint: Optional path to checkpoint to resume training from.
    """
    logger.info("="*80)
    logger.info("Starting TRL-based SFT Training for Chess")
    logger.info(f"Training mode: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    logger.info("="*80)

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
    else:
        logger.warning("CUDA not available. Training will be slow on CPU.")

    # Load dataset
    dataset = load_dataset_from_jsonl(DATA_FILE)

    # Setup model and tokenizer first (needed for chat template formatting)
    model, tokenizer = setup_model_and_tokenizer(base_model=base_model, use_unsloth=UNSLOTH_AVAILABLE, use_lora=use_lora)

    # Format dataset with chat template
    logger.info("Formatting dataset with chat template...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names
    )
    dataset = dataset.shuffle(seed=42)

    logger.info(f"Dataset prepared: {len(dataset)} examples")
    logger.info(f"Example text:\n{dataset[0]['text'][:300]}...")
    if resume_from_checkpoint:
        logger.info(f"NOTE: Dataset is shuffled with fixed seed=42, so resume will see consistent data order")

    # Create training config
    logger.info("Creating SFT training configuration...")
    config = create_sft_config()

    # Create trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset,
    )

    # Train
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        logger.info("Starting training from scratch...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Show final memory and time stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        logger.info(f"\nTraining Statistics:")
        logger.info(f"  Time: {trainer_stats.metrics['train_runtime']} seconds ({round(trainer_stats.metrics['train_runtime']/60, 2)} minutes)")
        logger.info(f"  Peak GPU memory: {used_memory} GB ({used_percentage}%)")
        logger.info(f"  Memory used for training: {used_memory_for_lora} GB ({lora_percentage}%)")

    # Save final model
    logger.info(f"\nSaving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("="*80)
    logger.info("Training complete!")
    logger.info("="*80)


def main():
    """Main entry point."""
    global DATA_FILE, OUTPUT_DIR, NUM_TRAIN_EPOCHS, MAX_STEPS, USE_LORA

    parser = argparse.ArgumentParser(
        description="TRL-based SFT training script for chess move generation"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to a fine-tuned model or LoRA adapters to use as base "
             "(e.g., ./chess-sft-qwen3-0.6b-lora/). If not provided, uses default model."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=DATA_FILE,
        help=f"Path to training data JSONL file (default: {DATA_FILE})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for trained model (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=NUM_TRAIN_EPOCHS,
        help=f"Number of training epochs (default: {NUM_TRAIN_EPOCHS})"
    )
    parser.add_argument(
        "--full-tune",
        action="store_true",
        help="Enable full fine-tuning instead of LoRA (trains all parameters)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., ./output/checkpoint-500 or just ./output for latest)"
    )

    args = parser.parse_args()

    # Update global config if arguments provided
    if args.data_file != DATA_FILE:
        DATA_FILE = args.data_file
    if args.output_dir != OUTPUT_DIR:
        OUTPUT_DIR = args.output_dir
    NUM_TRAIN_EPOCHS = args.num_epochs
    MAX_STEPS = -1  # Always use epochs, not max_steps
    USE_LORA = not args.full_tune  # If --full-tune is set, USE_LORA becomes False

    try:
        train_with_trl(
            base_model=args.base_model,
            use_lora=not args.full_tune,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
