#!/bin/bash

# Complete LSTM Multitask Evaluation Pipeline
# This script runs the full evaluation workflow end-to-end

set -e

echo "==================== LSTM Multitask Evaluation Pipeline ===================="
echo ""

# Check if model exists
MODEL_FILE="../models/lstm.pt"
LM_DATA_FILE="../models/lm_data.bin"
TEMPLATE_DIR="../EMNLP2018/templates"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file not found: $MODEL_FILE"
    echo "Please train the multitask model first using:"
    echo "  cd ../word-language-model"
    echo "  python main.py --lm_data ../data/lm_data --ccg_data ../data/ccg_data \\"
    echo "    --save ../models/lstm.pt --save_lm_data ../models/lm_data.bin"
    exit 1
fi

if [ ! -f "$LM_DATA_FILE" ]; then
    echo "ERROR: LM data file not found: $LM_DATA_FILE"
    exit 1
fi

echo "✓ Model files found"
echo ""

# Step 1: Generate templates if needed
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "Step 0: Generating test templates..."
    mkdir -p "$TEMPLATE_DIR"
    cd ../src
    python make_templates.py "$TEMPLATE_DIR"
    cd ../lstm_multitask_eval
    echo "✓ Templates generated"
    echo ""
fi

# Step 1: Evaluate
echo "Step 1: Evaluating LSTM multitask model on syntactic constructions..."
echo "(This will take 20-60 minutes depending on test data size)"
echo ""

python lstm_multitask_eval.py \
    --template_dir "$TEMPLATE_DIR" \
    --model "$MODEL_FILE" \
    --lm_data "$LM_DATA_FILE" \
    --output_file lstm_multitask_results.pickle \
    --device cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "✓ Evaluation complete"
echo ""

# Step 2: Analyze - Overall
echo "Step 2: Analyzing results (overall)..."
echo ""

python analyze_lstm_multitask_results.py \
    --results_file "$TEMPLATE_DIR/lstm_multitask_results.pickle" \
    --output_dir ../results/lstm_multitask \
    --mode overall

echo ""
echo "✓ Overall analysis complete"
echo ""

# Step 3: Analyze - Condensed
echo "Step 3: Analyzing results (condensed by construction type)..."
echo ""

python analyze_lstm_multitask_results.py \
    --results_file "$TEMPLATE_DIR/lstm_multitask_results.pickle" \
    --output_dir ../results/lstm_multitask \
    --mode condensed

echo ""
echo "✓ Condensed analysis complete"
echo ""

# Step 4: Analyze - Full
echo "Step 4: Analyzing results (full with examples)..."
echo ""

python analyze_lstm_multitask_results.py \
    --results_file "$TEMPLATE_DIR/lstm_multitask_results.pickle" \
    --output_dir ../results/lstm_multitask \
    --mode full

echo ""
echo "✓ Full analysis complete"
echo ""

echo "==================== Evaluation Complete ===================="
echo ""
echo "Results saved to:"
echo "  - ../results/lstm_multitask/analysis_overall.txt"
echo "  - ../results/lstm_multitask/analysis_condensed.txt"
echo "  - ../results/lstm_multitask/analysis_full.txt"
echo ""
echo "Next steps:"
echo "  1. View overall results: cat ../results/lstm_multitask/analysis_overall.txt"
echo "  2. View detailed results: cat ../results/lstm_multitask/analysis_full.txt"
echo "  3. Compare with other models: python ../ngram_eval/compare_ngram_results.py \\"
echo "       --ngram_results ../EMNLP2018/templates/ngram_results.pickle \\"
echo "       --gpt2_results ../EMNLP2018/templates/gpt2_results.pickle \\"
echo "       --lstm_results ../EMNLP2018/templates/lstm_multitask_results.pickle"
echo ""
