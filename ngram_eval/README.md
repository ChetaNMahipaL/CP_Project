# N-gram Language Model Evaluation

Complete pipeline for training and evaluating an n-gram language model using SRILM, following the paper "Targeted Syntactic Evaluation of Language Models" (Marvin & Linzen, EMNLP 2018).

## Overview

This directory provides tools to:

1. **Train a 5-gram language model** using SRILM's `ngram-count` tool
2. **Evaluate on syntactic constructions** using SRILM's `ngram` tool
3. **Analyze results** similar to the paper's evaluation
4. **Compare with other models** (GPT-2, LSTM, etc.)

## Prerequisites

- SRILM binaries: `ngram` and `ngram-count` (already in this directory)
- Training data: Language model data split (from wiki-103 or Wikipedia)
- Python 3.7+

## Setup

### 1. Prepare Language Model Data

The evaluation expects training data at:
```
../data/lm_data/train.txt
../data/lm_data/valid.txt
../data/lm_data/test.txt
```

Each file should contain one sentence per line, tokenized and lowercased.

### 2. Prepare Test Templates

Test sentences should be pre-generated at:
```
../EMNLP2018/templates/
```

Generate them using:
```bash
cd ../src
python make_templates.py ../EMNLP2018/templates
```

## Usage

### Quick Start (Complete Pipeline)

```bash
bash run_evaluation.sh
```

This runs the entire workflow: training → evaluation → analysis

### Step-by-Step Usage

#### Step 1: Train the Model

```bash
bash train_ngram.sh
```

**What it does:**
- Creates vocabulary from training data
- Trains a 5-gram model using Kneser-Ney smoothing
- Saves model to `../models/ngram_model.lm`
- Saves vocabulary to `../models/vocab.txt`

**Time:** 20-60 minutes (depends on data size and system)

**Output:**
- `../models/ngram_model.lm` - Trained language model
- `../models/vocab.txt` - Vocabulary file

#### Step 2: Evaluate

```bash
python evaluate_ngram.py \
  --template_dir ../EMNLP2018/templates \
  --model ../models/ngram_model.lm \
  --vocab ../models/vocab.txt \
  --output_file ngram_results.pickle
```

**Parameters:**
- `--template_dir`: Directory with template pickles
- `--model`: Path to trained .lm model
- `--vocab`: Vocabulary file used during training
- `--output_file`: Where to save results
- `--srilm_bin`: Path to ngram binary (default: ./ngram)
- `--order`: N-gram order (default: 5)
- `--tests`: Which tests to run (agrmt/npi/all)

**Time:** 30-60 minutes (depends on number of test sentences)

**Output:**
- `ngram_results.pickle` - Scores for all test sentences

#### Step 3: Analyze Results

**Overall summary:**
```bash
python analyze_ngram_results.py \
  --results_file ../EMNLP2018/templates/ngram_results.pickle \
  --mode overall
```

**Grouped by construction:**
```bash
python analyze_ngram_results.py \
  --results_file ../EMNLP2018/templates/ngram_results.pickle \
  --mode condensed
```

**Detailed with examples:**
```bash
python analyze_ngram_results.py \
  --results_file ../EMNLP2018/templates/ngram_results.pickle \
  --mode full
```

#### Step 4: Compare with Other Models

```bash
python compare_ngram_results.py \
  --ngram_results ../EMNLP2018/templates/ngram_results.pickle \
  --gpt2_results ../EMNLP2018/templates/gpt2_results.pickle \
  --lstm_results ../results/lstm/rnn_results.pickle \
  --output_file comparison.txt
```

## Files

### Scripts

- **`train_ngram.sh`** - Train 5-gram model
- **`evaluate_ngram.py`** - Evaluate on test sentences  
- **`analyze_ngram_results.py`** - Analyze results
- **`compare_ngram_results.py`** - Compare models
- **`run_evaluation.sh`** - Complete pipeline

### Binaries (SRILM)

- **`ngram`** - Score sentences with trained model
- **`ngram-count`** - Train n-gram model

## Output Files

After complete evaluation:

```
../models/
  ├── ngram_model.lm          # Trained model
  └── vocab.txt               # Vocabulary

../EMNLP2018/templates/
  └── ngram_results.pickle    # Raw evaluation results

../results/ngram/
  ├── analysis_overall.txt    # Summary statistics
  ├── analysis_condensed.txt  # By construction type
  └── analysis_full.txt       # Detailed with examples

comparison.txt                # Side-by-side comparison
```

## Expected Results (From Paper)

Based on the paper's reported n-gram results on 5-gram models:

**Agreement tasks:**
- Simple agreement: ~95%
- Object relatives: ~70-85%
- Subject relatives: ~80-90%
- Average: ~85-90%

**NPI tasks:**
- Simple NPI: ~45-55%
- Across NPIs: ~70-80%
- Average: ~60-70%

**Overall:** ~80-85%

## SRILM Parameters

The training uses:
- **Order:** 5-gram
- **Smoothing:** Kneser-Ney discounting
- **Interpolation:** Yes
- **Unknown handling:** Enabled

For different parameters, edit `train_ngram.sh` and modify the `ngram-count` command.

## Troubleshooting

### Training Too Slow

N-gram training can be slow on large datasets. To speed up:

1. Use a subset of training data initially
2. Check disk I/O (n-gram-count writes temporary files)
3. Monitor memory (allocate more if needed)

### Memory Issues

If out of memory during training, reduce data size or adjust SRILM parameters:
- Use `-sort` to reduce memory
- Process data in chunks

### Evaluation Errors

**"SRILM error for sentence"**
- Check vocabulary includes all test words
- Verify SRILM binary is correct

**"Model file not found"**
- Ensure training completed successfully
- Check file permissions

### Slow Evaluation

N-gram evaluation calls external SRILM binary for each test sentence, which is slower than neural models:
- Expect: 1-5 sentences/second depending on sentence length
- Total: 30-60 minutes for full test set

## References

- SRILM: http://www.speech.sri.com/projects/srilm/
- Paper: [Targeted Syntactic Evaluation of Language Models](https://arxiv.org/abs/1808.09031)
- Kneser-Ney smoothing: Kneser & Ney (1995)

## Notes

- The SRILM binaries are architecture-specific. They were compiled for Linux.
- Results will differ slightly if using different SRILM versions or parameters.
- Vocabulary must include `<unk>` for unknown words.
