# Chess Move Legality Training Code

This repository contains training code for developing chess models with move generation and reasoning capabilities. The project fine-tunes language models on chess move legality verification as a foundation for building more capable chess-playing systems.

## Overview

This is a **research project** aimed at training language models to:
1. Understand chess positions and piece movements
2. Verify move legality from board positions
3. Develop reasoning capabilities about chess (chain-of-thought)
4. Serve as a foundation for training models that can generate best moves

The approach uses supervised fine-tuning (SFT) on chess move legality data to teach models fundamental chess concepts before scaling to more complex tasks like move generation.

## Pre-trained Models

A trained model is available on Hugging Face Hub:
- **[navgeet/chess-sft-merged](https://huggingface.co/navgeet/chess-sft-merged)** - Merged chess move legality model ready for inference

## Requirements

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Data Format

The training script expects a JSONL file where each line contains:

```json
{
  "prompt": "Consider the position below and answer the query:\n\n[ASCII board representation]\n\nQuery: Is it legal for the white bishop at c4 to move to f7? Answer only yes or no",
  "output": "<think>\n[optional reasoning]\n</think>\n\nyes",
  "fen": "rnbq1rk1/ppp1ppbp/5np1/6B1/2BP4/2N2N2/PPP2PPP/R2QK2R w KQ - 4 8",
  "metadata": {
    "piece_type": "bishop",
    "piece_color": "white",
    "from_square": "c4",
    "to_square": "f7",
    "is_legal": true,
    "category": "legal_capture"
  }
}
```

Fields:
- `prompt`: Question about move legality with board state
- `output`: Answer (yes/no) with optional chain-of-thought reasoning
- `fen`: FEN notation of the position (optional, for reference)
- `metadata`: Additional information about the move (optional)

## Training

### Basic SFT Training (LoRA)

```bash
python train_sft_trl.py \
  --data-file path/to/dataset.jsonl \
  --output-dir output/chess-model \
  --num-epochs 3
```

### Full Fine-tuning

For full parameter training (requires more VRAM):

```bash
python train_sft_trl.py \
  --data-file path/to/dataset.jsonl \
  --output-dir output/chess-model-full \
  --num-epochs 3 \
  --full-tune
```

### Resume from Checkpoint

```bash
python train_sft_trl.py \
  --data-file path/to/dataset.jsonl \
  --output-dir output/chess-model \
  --resume-from-checkpoint output/chess-model/checkpoint-500
```

### Continue Training from Existing Model

```bash
# First, merge LoRA adapters if using LoRA
python merge_lora.py \
  --lora-path output/chess-model-lora \
  --output-dir output/chess-model-merged

# Then continue training
python train_sft_trl.py \
  --base-model output/chess-model-merged \
  --data-file path/to/new_dataset.jsonl \
  --output-dir output/chess-model-v2 \
  --num-epochs 3
```

## Merging LoRA Adapters

After training with LoRA, merge adapters into the base model:

```bash
python merge_lora.py \
  --lora-path output/chess-model-lora \
  --output-dir output/chess-model-merged
```

Options:
- `--base-model`: Override base model (default: reads from adapter config)
- `--quantization`: Quantize merged model (4bit, 8bit)

## Model Usage

After training, use the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output/chess-model")
tokenizer = AutoTokenizer.from_pretrained("output/chess-model")

# Format your chess position question
prompt = """Consider the position below and answer the query:

[board representation]

Query: Is it legal for the white knight at f3 to move to g5? Answer only yes or no"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Benchmarking

The `benchmark.py` script evaluates trained models on chess move legality tasks, measuring both accuracy and inference performance.

### Basic Usage

Benchmark a trained model on test data:

```bash
python benchmark.py \
  --model output/chess-model \
  --data path/to/test_data.jsonl \
  --num-samples 1000
```

### Example Output

```
==============================================================
RESULTS SUMMARY
==============================================================
Total samples: 1000
Correct predictions: 847
Average generation time: 0.132s
Average tokens generated: 12.3
Exact match accuracy: 84.7% (847/1000)
Average token overlap: 91.2%
Throughput: 7.58 samples/sec

153 failures logged to: failures.jsonl
==============================================================
```

### Analyzing Failures

Failed predictions are logged to `failures.jsonl` for analysis:

```python
import json

# Read failures
with open('failures.jsonl', 'r') as f:
    failures = [json.loads(line) for line in f]

# Analyze common failure patterns
for failure in failures[:5]:
    print(f"Position: {failure['fen']}")
    print(f"Expected: {failure['output']}")
    print(f"Generated: {failure['generated']}")
    print(f"Metadata: {failure['metadata']}")
    print("-" * 60)
```

### Performance Optimization

For faster benchmarking:
- Use GPU for inference (`device_map="auto"` in script)
- Reduce `--num-samples` for quick tests
- Use smaller batch sizes if running out of memory

## License

MIT License

