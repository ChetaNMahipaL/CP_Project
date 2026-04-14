# GPT-2 Syntactic Evaluation

This directory contains code to evaluate GPT-2 on the same syntactic constructions tested in the paper "Targeted Syntactic Evaluation of Language Models" (Marvin & Linzen, EMNLP 2018).

## Overview

The evaluation tests whether GPT-2 (a transformer-based language model) can capture specific linguistic phenomena, particularly:

- Subject-verb agreement in complex syntactic structures
- Negative Polarity Items (NPIs) licensing
- Reflexive binding
- Relative clauses
- And more...

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Test Sentences

The test sentences should already be generated from templates in the main project. They're located in:
```
../EMNLP2018/templates/
```

Each test case has a corresponding pickle file containing sentence pairs.

## Usage

### Step 1: Evaluate GPT-2 on Test Sentences

```bash
python gpt2_eval.py \
  --template_dir ../EMNLP2018/templates \
  --output_file gpt2_results.pickle \
  --model_name gpt2 \
  --device cuda
```

**Parameters:**
- `--template_dir`: Directory containing template pickle files
- `--output_file`: Where to save evaluation results
- `--model_name`: GPT-2 variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- `--device`: cuda or cpu
- `--tests`: Which tests to run (agrmt, npi, or all)

**Output:** `gpt2_results.pickle` containing scores for all test sentences

### Step 2: Analyze GPT-2 Results

```bash
python analyze_gpt2_results.py \
  --results_file gpt2_results.pickle \
  --output_dir ../results/gpt2 \
  --mode overall
```

**Modes:**
- `overall`: Summary accuracy for each test case
- `condensed`: Grouped by construction type (e.g., all object relatives together)
- `full`: Detailed results with sample sentences and token-level scores

### Step 3: Compare with Other Models

If you have LSTM and N-gram results from the original paper:

```bash
python compare_models.py \
  --gpt2_results gpt2_results.pickle \
  --lstm_results ../results/lstm/rnn_results.pickle \
  --ngram_results ../results/ngram/ngram_results.pickle \
  --output_file comparison.txt
```

This generates a comparison table across all models.

## Test Cases

The evaluation includes two main categories:

### Agreement Tests (18 cases)
- `simple_agrmt`: Simple subject-verb agreement
- `obj_rel_across_anim/inanim`: Object relative clauses with distant subject
- `obj_rel_within_anim/inanim`: Object relative clauses with local subject
- `obj_rel_no_comp_across_anim/inanim`: Object relatives without complementizer
- `obj_rel_no_comp_within_anim/inanim`: As above, without complementizer
- `subj_rel`: Subject relative clauses
- `prep_anim/inanim`: Prepositional phrase attachment
- `sent_comp`: Sentential complements
- `vp_coord`: VP coordination
- `long_vp_coord`: Long VP coordination
- `reflexives_across`: Reflexive binding across clauses
- `simple_reflexives`: Simple reflexive binding
- `reflexive_sent_comp`: Reflexives in sent. complements

### NPI Tests (4 cases)
- `simple_npi_anim`: Simple NPI licensing with animate subjects
- `simple_npi_inanim`: Simple NPI licensing with inanimate subjects
- `npi_across_anim`: NPI licensing with distant negation (animate)
- `npi_across_inanim`: NPI licensing with distant negation (inanimate)

## Output Files

After running the scripts:

```
gpt2_results.pickle          # Raw scores for all sentences
../results/gpt2/
  ├── analysis_overall.txt   # Summary statistics
  ├── analysis_condensed.txt # Accuracy by construction
  └── analysis_full.txt      # Detailed with examples

comparison.txt               # Side-by-side model comparison
```

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run evaluation (takes ~5-10 minutes depending on GPU)
python gpt2_eval.py --template_dir ../EMNLP2018/templates --device cuda

# 3. Analyze results in detail
python analyze_gpt2_results.py --results_file gpt2_results.pickle --mode full

# 4. Compare with LSTM baseline if available
python compare_models.py \
  --gpt2_results gpt2_results.pickle \
  --lstm_results ../results/lstm/rnn_results.pickle
```

## Expected Results

Based on the paper:
- **N-gram (5-gram)**: ~85-95% accuracy on agreement tasks, ~80-85% on NPI
- **LSTM**: ~75-90% accuracy on agreement, ~70-80% on NPI

**GPT-2** likely performs better than both, potentially achieving:
- Agreement tasks: 95%+
- NPI tasks: 90%+

(Exact numbers will depend on the specific model variant and training data differences)

## Troubleshooting

### Out of Memory
Use a smaller GPT-2 variant or reduce batch size:
```bash
python gpt2_eval.py --model_name gpt2 --batch_size 16
```

### Missing Template Files
Ensure templates have been generated:
```bash
cd ../src
python make_templates.py ../EMNLP2018/templates
```

### Slow Evaluation
GPT-2 evaluation is computationally expensive. Use GPU and the base `gpt2` model for faster results.

## References

- Original paper: [Targeted Syntactic Evaluation of Language Models](https://arxiv.org/abs/1808.09031)
- GPT-2: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

## Notes

- The code expects test sentences to already be generated (see main project README)
- Results are saved in the same pickle format as the original LSTM evaluation for easy comparison
- Token-level scores are included for detailed analysis
