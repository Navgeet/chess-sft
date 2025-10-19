#!/usr/bin/env python3
"""
Benchmark script for the merged Qwen3 0.6B model.
Tests inference speed and quality on chess tasks from data.jsonl.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MERGED_MODEL_PATH = "./output"
DATA_FILE = "./data.jsonl"
NUM_SAMPLES = 100  # Number of samples to benchmark
MAX_NEW_TOKENS = 30000


def load_model_and_tokenizer(model_path: str = None):
    """Load the merged model."""
    model_path = model_path or MERGED_MODEL_PATH
    logger.info(f"Loading merged model from: {model_path}")

    path_obj = Path(model_path)
    if not path_obj.exists():
        logger.error(f"Merged model path not found: {model_path}")
        logger.error("Make sure merge_lora.py has been run successfully.")
        raise FileNotFoundError(f"Merged model not found at {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load merged model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )

    logger.info("Merged model loaded")
    return model, tokenizer


def load_data(filepath: str, num_samples: int = None, random_sample: bool = True, seed: int = None) -> List[Dict]:
    """Load JSONL data file with optional random sampling.

    Args:
        filepath: Path to JSONL data file
        num_samples: Number of samples to load (None = all)
        random_sample: Whether to randomly sample (True) or take first N (False)
        seed: Random seed for reproducibility (None = random)

    Returns:
        List of loaded samples
    """
    if seed is not None:
        random.seed(seed)

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # If random sampling requested and num_samples specified
    if random_sample and num_samples and len(data) > num_samples:
        data = random.sample(data, num_samples)
    elif num_samples and len(data) > num_samples:
        # Otherwise take first num_samples
        data = data[:num_samples]

    return data


def format_prompt(example: Dict, tokenizer) -> str:
    """Format prompt for generation using chat template with thinking enabled."""
    prompt = example.get("prompt", "")
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> tuple:
    """Generate response from prompt and return (thinking_content, response)."""
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Extract only the generated tokens (exclude input)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parse thinking content by finding </think> token (151668)
    try:
        # Find the last occurrence of </think> token
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return thinking_content, response


def compute_metrics(generated: str, expected: str) -> Dict[str, float]:
    """Compute simple metrics comparing generated and expected outputs."""
    # Normalize strings: remove commas, extra whitespace, and convert to lowercase
    def normalize(text: str) -> set:
        """Normalize text by removing commas and splitting into token set."""
        normalized = text.replace(',', ' ').lower().strip()
        return set(normalized.split())

    gen_tokens = normalize(generated)
    exp_tokens = normalize(expected)

    # Exact match: all tokens must match (order-independent)
    exact_match = 1.0 if gen_tokens == exp_tokens else 0.0

    # Token overlap (F1-like metric)
    if len(gen_tokens) == 0 or len(exp_tokens) == 0:
        token_overlap = 0.0
    else:
        overlap = len(gen_tokens & exp_tokens)
        precision = overlap / len(gen_tokens) if gen_tokens else 0.0
        recall = overlap / len(exp_tokens) if exp_tokens else 0.0
        token_overlap = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "token_overlap": token_overlap,
    }


def benchmark(random_sample: bool = True, seed: int = None, model_path: str = None, data_file: str = None, failures_file: str = None, num_samples: int = NUM_SAMPLES):
    """Run benchmarks on fine-tuned model.

    Args:
        random_sample: Whether to randomly sample data (True) or take first N (False)
        seed: Random seed for reproducibility
        model_path: Path to model directory (defaults to MERGED_MODEL_PATH)
        data_file: Path to data JSONL file (defaults to DATA_FILE)
        failures_file: Path to file for logging failures (defaults to 'failures.jsonl')
    """
    model_path = model_path or MERGED_MODEL_PATH
    data_file = data_file or DATA_FILE
    failures_file = failures_file or "failures.jsonl"

    logger.info("=" * 60)
    logger.info("FINE-TUNED MODEL BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_file}")
    logger.info(f"Failures log: {failures_file}")
    logger.info(f"Random sampling: {random_sample}, Seed: {seed}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load data
    if not Path(data_file).exists():
        logger.error(f"Data file {data_file} not found!")
        return

    data = load_data(data_file, num_samples=num_samples, random_sample=random_sample, seed=seed)
    logger.info(f"Loaded {len(data)} samples for benchmarking")

    # Benchmark metrics
    generation_times = []
    token_counts = []
    exact_matches = []
    token_overlaps = []

    model.eval()
    logger.info("\nRunning inference benchmarks...")
    logger.info("-" * 60)

    # Open failures file for writing
    failures_f = open(failures_file, 'w')

    for idx, example in enumerate(data):
        prompt = format_prompt(example, tokenizer)
        expected = example.get("output", "")

        # Measure generation time
        start_time = time.time()
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        # Extract only the generated tokens (exclude input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        generated_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        gen_time = time.time() - start_time
        generation_times.append(gen_time)

        # Count tokens
        tokens = len(output_ids)
        token_counts.append(tokens)

        # Compute metrics (compare full generated output against expected)
        metrics = compute_metrics(generated_output, expected)
        exact_matches.append(metrics["exact_match"])
        token_overlaps.append(metrics["token_overlap"])

        # Log failures to file
        if metrics["exact_match"] == 0.0:
            failure_record = {
                "index": idx,
                "prompt": example.get('prompt', ''),
                "fen": example.get('fen', ''),
                "output": expected,
                "generated": generated_output,
                "metadata": example.get('metadata', {})
            }
            failures_f.write(json.dumps(failure_record) + '\n')
            failures_f.flush()

        # Log sample
        # logger.info(f"\nSample {idx + 1}/{len(data)}:")
        # logger.info(f"  Prompt: {example.get('prompt', '')}")
        # logger.info(f"  FEN: {example.get('fen', '')}")
        # logger.info(f"  Expected: {expected}")
        # logger.info(f"  Generated: {generated_output}")
        # logger.info(f"  Time: {gen_time:.3f}s")
        # logger.info(f"  Tokens: {tokens}")
        # logger.info(f"  Exact Match: {metrics['exact_match']:.1%}")
        # logger.info(f"  Token Overlap: {metrics['token_overlap']:.1%}")

    # Close failures file
    failures_f.close()

    # Aggregate results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    avg_gen_time = sum(generation_times) / len(generation_times)
    avg_tokens = sum(token_counts) / len(token_counts)
    total_correct = sum(exact_matches)
    avg_exact_match = total_correct / len(exact_matches)
    avg_token_overlap = sum(token_overlaps) / len(token_overlaps)

    logger.info(f"Total samples: {len(exact_matches)}")
    logger.info(f"Correct predictions: {int(total_correct)}")
    logger.info(f"Average generation time: {avg_gen_time:.3f}s")
    logger.info(f"Average tokens generated: {avg_tokens:.1f}")
    logger.info(f"Exact match accuracy: {avg_exact_match:.1%} ({int(total_correct)}/{len(exact_matches)})")
    logger.info(f"Average token overlap: {avg_token_overlap:.1%}")
    logger.info(f"Throughput: {1/avg_gen_time:.2f} samples/sec")

    num_failures = len(exact_matches) - int(total_correct)
    if num_failures > 0:
        logger.info(f"\n{num_failures} failures logged to: {failures_file}")
    else:
        logger.info(f"\nNo failures! All predictions correct.")

    logger.info("\n" + "=" * 60)

    return {
        "model": "fine-tuned",
        "avg_generation_time": avg_gen_time,
        "avg_tokens": avg_tokens,
        "exact_match": avg_exact_match,
        "token_overlap": avg_token_overlap,
        "throughput": 1 / avg_gen_time,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark merged Qwen3 0.6B model")
    parser.add_argument(
        "--model",
        type=str,
        default=MERGED_MODEL_PATH,
        help=f"Path to model directory (default: {MERGED_MODEL_PATH})"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of samples (default: {NUM_SAMPLES})"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_FILE,
        help=f"Path to data JSONL file (default: {DATA_FILE})"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Load samples sequentially (default: random)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--failures",
        type=str,
        default="failures.jsonl",
        help="Path to failures log file (default: failures.jsonl)"
    )

    args = parser.parse_args()

    benchmark(
        random_sample=not args.sequential,
        seed=args.seed,
        model_path=args.model,
        data_file=args.data,
        failures_file=args.failures,
        num_samples=args.num_samples
    )
