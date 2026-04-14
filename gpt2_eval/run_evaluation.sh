#!/bin/bash

# Quick start script for GPT-2 evaluation
# This script runs the complete pipeline end-to-end

set -e

echo "==================== GPT-2 Syntactic Evaluation ===================="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 2: Run evaluation
echo "Step 2: Evaluating GPT-2 on syntactic constructions..."
echo "(This may take 5-15 minutes depending on your GPU)"
python gpt2_eval.py \
  --template_dir ../EMNLP2018/templates \
  --output_file gpt2_results.pickle \
  --model_name gpt2 \
  --device cuda \
  --tests all

echo "✓ Evaluation complete"
echo ""

# Step 3: Analyze results
echo "Step 3: Analyzing results (overall)..."
python analyze_gpt2_results.py \
  --results_file gpt2_results.pickle \
  --output_dir ../results/gpt2 \
  --mode overall

echo ""
echo "Step 4: Analyzing results (condensed by construction)..."
python analyze_gpt2_results.py \
  --results_file gpt2_results.pickle \
  --output_dir ../results/gpt2 \
  --mode condensed

echo ""
echo "==================== Evaluation Complete ===================="
echo ""
echo "Results saved to:"
echo "  - Raw results: gpt2_results.pickle"
echo "  - Analysis: ../results/gpt2/"
echo ""
echo "Optional: Compare with LSTM results"
echo "python compare_models.py \\"
echo "  --gpt2_results gpt2_results.pickle \\"
echo "  --lstm_results ../results/lstm/rnn_results.pickle"
echo ""
