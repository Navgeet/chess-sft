#!/bin/bash
# Quick Start Script for Chess Move Legality Training
# This script demonstrates basic usage of the training code

set -e  # Exit on error

echo "================================"
echo "Chess Training Quick Start"
echo "================================"
echo ""

# Check if example data exists
if [ ! -f "example_data.jsonl" ]; then
    echo "ERROR: example_data.jsonl not found!"
    echo "Please make sure you're running this script from the project directory."
    exit 1
fi

echo "1. Training with LoRA (recommended for quick testing)"
echo "   Command: python train_sft_trl.py --data-file example_data.jsonl --output-dir output/test-lora --num-epochs 1"
echo ""

echo "2. Full fine-tuning (requires more GPU memory)"
echo "   Command: python train_sft_trl.py --data-file example_data.jsonl --output-dir output/test-full --num-epochs 1 --full-tune"
echo ""

echo "3. Resume training from checkpoint"
echo "   Command: python train_sft_trl.py --data-file example_data.jsonl --output-dir output/test-lora --resume-from-checkpoint output/test-lora/checkpoint-500"
echo ""

echo "4. Merge LoRA adapters (after LoRA training)"
echo "   Command: python merge_lora.py --lora-path output/test-lora --output-dir output/test-merged"
echo ""

echo "5. Benchmark trained model"
echo "   Command: python benchmark.py --model output/test-lora --data example_data.jsonl --num-samples 3"
echo ""

echo "================================"
echo "Ready to train!"
echo "================================"
echo ""
echo "To start a quick test run with the example data:"
echo ""
echo "  python train_sft_trl.py --data-file example_data.jsonl --output-dir output/test-lora --num-epochs 1"
echo ""
echo "Note: The example dataset is very small (3 examples) and is only for testing."
echo "For real training, you'll need a larger dataset (100k+ examples recommended)."
