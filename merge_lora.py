#!/usr/bin/env python3
"""
Script to merge LoRA adapters into base model and save the merged model.
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_BASE_MODEL = "unsloth/Qwen3-0.6B"


def merge_lora_adapters(lora_path: str, output_dir: str, base_model: str = DEFAULT_BASE_MODEL):
    """Merge LoRA adapters into base model and save."""
    logger.info("=" * 80)
    logger.info("MERGING LORA ADAPTERS")
    logger.info("=" * 80)

    lora_path = Path(lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json found in {lora_path}")

    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    logger.info(f"Loading tokenizer from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    logger.info("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 80)
    logger.info("Merge complete!")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA adapters (e.g., ./chess-sft-qwen3-0.6b-lora/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model to merge into (default: {DEFAULT_BASE_MODEL})"
    )

    args = parser.parse_args()

    try:
        merge_lora_adapters(args.lora_path, args.output_dir, base_model=args.base_model)
    except Exception as e:
        logger.error(f"Merge failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
