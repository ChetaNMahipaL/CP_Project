#!/bin/bash

set -e

SRILM_BIN_DIR="." 
WORKDIR="../"
LM_DATA_DIR="$WORKDIR/data/lm_data"
MODEL_DIR="$WORKDIR/models"
NGRAM_MODEL="$MODEL_DIR/ngram_model.lm"
NGRAM_ORDER=5
VOCAB_FILE="$MODEL_DIR/vocab.txt"

echo "=========================================="
echo "Training N-gram Language Model (SRILM)"
echo "=========================================="
echo ""

mkdir -p "$MODEL_DIR"

if [ ! -f "$LM_DATA_DIR/train.txt" ]; then
    echo "ERROR: Training data not found at $LM_DATA_DIR/train.txt"
    echo "Please ensure language model data is in the correct location"
    exit 1
fi

if [ ! -f "$VOCAB_FILE" ]; then
    echo "Step 1: Generating vocabulary from training data..."
    cat "$LM_DATA_DIR/train.txt" | tr ' ' '\n' | sort | uniq > "$VOCAB_FILE"
    echo "✓ Vocabulary generated: $VOCAB_FILE"
    echo ""
fi

echo "Step 2: Training $NGRAM_ORDER-gram model..."
echo "This may take 10-30 minutes depending on data size and system..."
echo ""

# Use ngram-count to build the model
# -order: order of n-gram model
# -text: training data file
# -lm: output model file
# -vocab: vocabulary file
# -unk: treat OOV words as <unk>
# -interpolate: use interpolation
# -kndiscount: Kneser-Ney discounting

"$SRILM_BIN_DIR/ngram-count" \
    -order "$NGRAM_ORDER" \
    -text "$LM_DATA_DIR/train.txt" \
    -lm "$NGRAM_MODEL" \
    -vocab "$VOCAB_FILE" \
    -unk \
    -interpolate \
    -kndiscount

if [ $? -eq 0 ]; then
    echo "✓ Model trained successfully: $NGRAM_MODEL"
    echo ""
    SIZE=$(du -h "$NGRAM_MODEL" | cut -f1)
    echo "Model size: $SIZE"
else
    echo "ERROR: Failed to train n-gram model"
    exit 1
fi

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: $NGRAM_MODEL"
echo "Vocabulary saved to: $VOCAB_FILE"
echo ""
echo "Next step: Run evaluation"
echo "  python evaluate_ngram.py --model $NGRAM_MODEL --vocab $VOCAB_FILE"
