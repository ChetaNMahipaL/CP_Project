# N-gram Evaluation - Quick Reference

## Files in This Directory

```
ngram                          # SRILM ngram binary (score sentences)
ngram-count                    # SRILM ngram-count binary (train model)

train_ngram.sh                 # Train 5-gram model script
evaluate_ngram.py              # Evaluate on test sentences
analyze_ngram_results.py       # Analyze results
compare_ngram_results.py       # Compare with other models
run_evaluation.sh              # Complete pipeline

utils.py                       # Utility functions
simple_example.py              # Demo script
README.md                      # Full documentation
QUICKSTART.md                  # This file
```

## Quick Start (2 commands)

```bash
# 1. Train the model (20-60 minutes)
bash train_ngram.sh

# 2. Run complete evaluation (requires trained model)
bash run_evaluation.sh
```

## Manual Step-by-Step

```bash
# Train model
bash train_ngram.sh

# Evaluate
python evaluate_ngram.py \
  --template_dir ../EMNLP2018/templates \
  --model ../models/ngram_model.lm \
  --vocab ../models/vocab.txt

# Analyze
python analyze_ngram_results.py \
  --results_file ../EMNLP2018/templates/ngram_results.pickle \
  --mode overall

# Compare
python compare_ngram_results.py \
  --ngram_results ../EMNLP2018/templates/ngram_results.pickle \
  --gpt2_results ../EMNLP2018/templates/gpt2_results.pickle
```

## Try Examples

```bash
python simple_example.py
```

Shows 4 examples:
1. Simple agreement
2. Object relative clause
3. Negative Polarity Items
4. Reflexive binding

## Expected Results

- Simple agreement: ~95%
- Complex constructions: 70-90%
- NPI licensing: 45-70%
- **Overall: ~80-85%**

## Troubleshooting

### "Model file not found"
→ Run `bash train_ngram.sh` first

### "Permission denied" on .sh files
→ Run `chmod +x *.sh`

### Training is slow
→ Check disk I/O and available memory

### Evaluation is slow
→ Normal - SRILM calls are slower than neural models (~1 sentence/sec)

## Model Details

- **Order:** 5-gram
- **Smoothing:** Kneser-Ney discounting
- **Training data:** ~80M tokens from Wikipedia
- **Vocabulary:** ~100-200K words

## Output Locations

```
../models/
  ├── ngram_model.lm          # Trained model
  └── vocab.txt               # Vocabulary

../results/ngram/
  ├── analysis_overall.txt
  ├── analysis_condensed.txt
  └── analysis_full.txt

../EMNLP2018/templates/
  └── ngram_results.pickle    # Raw results
```

## Paper Reference

"Targeted Syntactic Evaluation of Language Models" 
- Marvin & Linzen (EMNLP 2018)
- https://arxiv.org/abs/1808.09031

## SRILM Documentation

- http://www.speech.sri.com/projects/srilm/
- Includes detailed parameter documentation
