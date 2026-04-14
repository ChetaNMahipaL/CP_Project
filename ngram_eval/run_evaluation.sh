#!/bin/bash

# Complete n-gram evaluation pipeline
# This script runs the full workflow end-to-end

set -e

echo "==================== N-gram Language Model Evaluation ===================="
echo ""

# Step 1: Train n-gram model
echo "Step 1: Training 5-gram model using SRILM..."
echo "(This may take 20-60 minutes depending on training data size)"
bash train_ngram.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to train n-gram model"
    exit 1
fi

echo ""
echo "✓ Training complete"
echo ""

MODEL_FILE="../models/ngram_model.lm"
VOCAB_FILE="../models/vocab.txt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file not found: $MODEL_FILE"
    exit 1
fi

# Step 2: Evaluate on test sentences
echo "Step 2: Evaluating n-gram model on syntactic constructions..."
echo "(This will take 30-60 minutes)"
python evaluate_ngram.py \
  --template_dir ../EMNLP2018/templates \
  --model "$MODEL_FILE" \
  --vocab "$VOCAB_FILE" \
  --output_file ngram_results.pickle \
  --tests all

echo ""
echo "✓ Evaluation complete"
echo ""

# Step 3: Analyze results
echo "Step 3: Analyzing results (overall)..."
python analyze_ngram_results.py \
  --results_file ../EMNLP2018/templates/ngram_results.pickle \
  --output_dir ../results/ngram \
  --mode overall

echo ""
echo "Step 4: Analyzing results (condensed by construction)..."
python analyze_ngram_results.py \
  --results_file ../EMNLP2018/templates/ngram_results.pickle \
  --output_dir ../results/ngram \
  --mode condensed

echo ""
echo "==================== Evaluation Complete ===================="
echo ""
echo "Results saved to:"
echo "  - Raw results: ../EMNLP2018/templates/ngram_results.pickle"
echo "  - Analysis: ../results/ngram/"
echo ""
echo "Optional: Compare with other models"
echo "python compare_ngram_results.py \\"
echo "  --ngram_results ../EMNLP2018/templates/ngram_results.pickle \\"
echo "  --gpt2_results ../EMNLP2018/templates/gpt2_results.pickle \\"
echo "  --lstm_results ../results/lstm/rnn_results.pickle"
echo ""
