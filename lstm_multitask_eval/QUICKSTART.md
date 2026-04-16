# LSTM Multitask Evaluation - Quick Reference

## Quick Start (Copy-Paste Commands)

### 1. Full Automated Pipeline
```bash
bash run_evaluation.sh
```
This runs everything: evaluation and all three levels of analysis.

### 2. Manual Step-by-Step

```bash
# Evaluate the model
python lstm_multitask_eval.py \
  --template_dir ../EMNLP2018/templates \
  --model ../models/lstm.pt \
  --lm_data ../models/lm_data.bin

# Analyze - overall summary
python analyze_lstm_multitask_results.py \
  --results_file ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --mode overall

# Analyze - by construction
python analyze_lstm_multitask_results.py \
  --results_file ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --mode condensed

# Analyze - with examples
python analyze_lstm_multitask_results.py \
  --results_file ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --mode full
```

## Quick Examples

### Test on Specific Constructions Only
```bash
# Only agreement tests
python lstm_multitask_eval.py --tests agrmt

# Only NPI tests
python lstm_multitask_eval.py --tests npi
```

### Use CPU Instead of GPU
```bash
python lstm_multitask_eval.py --device cpu
```

### Save Results to Custom Location
```bash
python lstm_multitask_eval.py --output_file my_results.pickle
```

## Viewing Results

### View Overall Summary
```bash
cat ../results/lstm_multitask/analysis_overall.txt
```

### View Results By Construction Type
```bash
cat ../results/lstm_multitask/analysis_condensed.txt
```

### View Full Results with Examples
```bash
cat ../results/lstm_multitask/analysis_full.txt
```

## Comparing Models

To compare LSTM multitask with GPT-2 and N-gram results:
```bash
python ../ngram_eval/compare_ngram_results.py \
  --ngram_results ../EMNLP2018/templates/ngram_results.pickle \
  --gpt2_results ../EMNLP2018/templates/gpt2_results.pickle \
  --lstm_results ../EMNLP2018/templates/lstm_multitask_results.pickle
```

## Troubleshooting

### Model Not Found
Make sure you've trained it:
```bash
cd ../word-language-model
python main.py --lm_data ../data/lm_data --ccg_data ../data/ccg_data \
  --save ../models/lstm.pt --save_lm_data ../models/lm_data.bin
```

### Templates Not Found
Generate them:
```bash
cd ../src
python make_templates.py ../EMNLP2018/templates
```

### GPU Out of Memory
Use CPU instead: `--device cpu`

### Want to Redo Evaluation
Delete the results file:
```bash
rm ../EMNLP2018/templates/lstm_multitask_results.pickle
```

## Files

- `lstm_multitask_eval.py` - Main evaluation script
- `analyze_lstm_multitask_results.py` - Analysis script
- `utils.py` - Helper functions
- `run_evaluation.sh` - Full pipeline script
- `README.md` - Full documentation
- `requirements.txt` - Python dependencies

## Test Cases Included

### Agreement (18 cases)
- simple_agrmt
- obj_rel_across (anim/inanim)
- obj_rel_within (anim/inanim)
- obj_rel_no_comp_across (anim/inanim)
- obj_rel_no_comp_within (anim/inanim)
- subj_rel
- prep (anim/inanim)
- sent_comp
- vp_coord
- long_vp_coord
- reflexives_across
- simple_reflexives
- reflexive_sent_comp

### NPI (4+ cases)
- Various NPI licensing conditions

## Expected Timing

- **Evaluation**: 20-60 minutes (depends on GPU)
- **Analysis**: < 1 minute
- **Total**: 20-65 minutes

## Output Files

After running, you'll have:
- `../results/lstm_multitask/analysis_overall.txt` - Summary accuracies
- `../results/lstm_multitask/analysis_condensed.txt` - Grouped by construction
- `../results/lstm_multitask/analysis_full.txt` - With example sentence pairs
