# Multitask LSTM Syntactic Evaluation

This directory contains code to evaluate the trained Multitask LSTM model (jointly trained on language modeling and CCG supertagging) on syntactic constructions as described in "Targeted Syntactic Evaluation of Language Models" (Marvin & Linzen, EMNLP 2018).

## Overview

The evaluation tests whether the Multitask LSTM (trained with both LM and CCG data) can capture specific linguistic phenomena, including:

- Subject-verb agreement in complex syntactic structures
- Negative Polarity Items (NPIs) licensing
- Reflexive binding
- Relative clauses
- And more...

The Multitask LSTM is jointly trained on:
- **LM task**: Standard language modeling on general corpora
- **CCG task**: CCG supertagging on annotated CCG data

This enables the model to learn both generic linguistic patterns and more structured syntactic knowledge.

## Setup

### 1. Prerequisites

- PyTorch (with CUDA support if using GPU)
- dill (for unpickling the Dictionary object from lm_data.bin)
- The trained multitask LSTM model checkpoint: `models/lstm.pt`
- The LM data dictionary file: `models/lm_data.bin`
- Test template files in: `EMNLP2018/templates/`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The `dill` package is essential for loading the Dictionary object stored in `lm_data.bin`. It was used during model training to serialize the custom Dictionary class.

### 3. Model Files

Make sure you have the following trained model files:
- `../models/lstm.pt` - The trained multitask LSTM checkpoint
- `../models/lm_data.bin` - Dictionary and data tensors for faster loading

If these files don't exist, train the model first using:
```bash
cd ../word-language-model
python main.py --lm_data ../data/lm_data --ccg_data ../data/ccg_data \
  --save ../models/lstm.pt --save_lm_data ../models/lm_data.bin
```

## Usage

### Step 1: Generate Test Sentences (if not already done)

```bash
cd ../src
python make_templates.py ../EMNLP2018/templates
```

This generates pickle files containing test sentence pairs for all grammatical constructions.

### Step 2: Evaluate Multitask LSTM on Test Sentences

```bash
python lstm_multitask_eval.py \
  --template_dir ../EMNLP2018/templates \
  --model ../models/lstm.pt \
  --lm_data ../models/lm_data.bin \
  --output_file lstm_multitask_results.pickle \
  --device cuda
```

**Parameters:**
- `--template_dir`: Directory containing template pickle files (default: `../EMNLP2018/templates`)
- `--model`: Path to the trained LSTM checkpoint (default: `../models/lstm.pt`)
- `--lm_data`: Path to LM data file with dictionary (default: `../models/lm_data.bin`)
- `--output_file`: Where to save evaluation results (default: `lstm_multitask_results.pickle`)
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--tests`: Which tests to run (agrmt/npi/all, default: all)
- `--batch_size`: Batch size for evaluation (default: 1)

**Output:** 
- `lstm_multitask_results.pickle` - Scores for all test sentences
- Will be saved in the template directory by default

**Time:** 
- Depends on number of test sentences (typically 20-60 minutes for full evaluation)

### Step 3: Analyze Results

#### Overall Summary
```bash
python analyze_lstm_multitask_results.py \
  --results_file ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --output_dir ../results/lstm_multitask \
  --mode overall
```

#### Grouped by Construction
```bash
python analyze_lstm_multitask_results.py \
  --results_file ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --output_dir ../results/lstm_multitask \
  --mode condensed
```

#### Detailed with Examples
```bash
python analyze_lstm_multitask_results.py \
  --results_file ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --output_dir ../results/lstm_multitask \
  --mode full
```

**Modes:**
- `overall`: Summary accuracy for each test case
- `condensed`: Grouped by construction type (e.g., all object relatives together)
- `full`: Detailed results with sample sentences and token-level scores

**Output:** Analysis results saved to `../results/lstm_multitask/analysis_{mode}.txt`

## Test Cases

The evaluation includes two main categories of syntactic constructions:

### Agreement Tests (18+ cases)
- `simple_agrmt`: Simple subject-verb agreement
- `obj_rel_across_anim/inanim`: Object relative clauses with distant subject
- `obj_rel_within_anim/inanim`: Object relative clauses with local subject
- `obj_rel_no_comp_across_anim/inanim`: Object relatives without complementizer
- `subj_rel`: Subject relative clauses
- `prep_anim/inanim`: Prepositional phrase attachment
- `sent_comp`: Sentential complements
- `vp_coord`: VP coordination
- `long_vp_coord`: Long VP coordination
- `reflexives_across`: Reflexive binding across clauses
- `simple_reflexives`: Simple reflexive binding
- `reflexive_sent_comp`: Reflexives in sentential complements

### NPI (Negative Polarity Item) Tests (4+ cases)
- NPI licensing across various contexts
- Tests proper understanding of licensing conditions for NPIs

## Understanding Results

### Accuracy Metric
For each test case pair (grammatical vs ungrammatical sentence):
- **Correct**: Model assigns higher probability to grammatical sentence
- **Accuracy**: % of cases where grammatical > ungrammatical

### Interpreting Scores

In the output, you'll see:
- `Accuracy`: Percentage of correctly ordered sentence pairs
- `Correct`: Number of correct predictions out of total
- Token-level scores showing log probability for each predicted token

Higher (less negative) log probabilities indicate the model is more confident about a token prediction.

## Comparing with Other Models

To compare results with GPT-2 and N-gram models:

```bash
python ../ngram_eval/compare_ngram_results.py \
  --ngram_results ../EMNLP2018/templates/ngram_results.pickle \
  --gpt2_results ../EMNLP2018/templates/gpt2_results.pickle \
  --lstm_results ../EMNLP2018/templates/lstm_multitask_results.pickle \
  --output_file comparison.txt
```

## Troubleshooting

### Dictionary Loading Errors

If you get errors like `No module named 'data'` or `Can't get attribute 'Dictionary'`:

1. **Install dill**: `pip install dill`
2. **Verify paths**: Check that `lm_data.bin` is in the correct location (`../models/lm_data.bin`)
3. **Recreate lm_data.bin**: If the file is corrupted, retrain the model:
   ```bash
   cd ../word-language-model
   python main.py --lm_data ../data/lm_data --ccg_data ../data/ccg_data \
     --save ../models/lstm.pt --save_lm_data ../models/lm_data.bin
   ```

The `lm_data.bin` file contains a custom `Dictionary` class that requires:
- The `data.py` module to be accessible (added to sys.path automatically)
- `dill` package for proper unpickling

### Model Loading Issues
If you encounter errors loading the model:
1. Verify paths to `lstm.pt` and `lm_data.bin` are correct
2. Check that the model was trained with compatible PyTorch version
3. Try moving model to CPU: use `--device cpu`

### Out of Memory Errors
- Reduce batch size (though default is 1 which is minimal)
- Use CPU instead of GPU
- Score fewer test cases at once by using `--tests agrmt` or `--tests npi`

### Missing Test Files
- Ensure test templates have been generated: `python ../src/make_templates.py ../EMNLP2018/templates`
- Check that pickle files exist in `../EMNLP2018/templates/`

## File Structure

```
lstm_multitask_eval/
├── lstm_multitask_eval.py          # Main evaluation script
├── analyze_lstm_multitask_results.py # Analysis script
├── utils.py                         # Helper functions
├── README.md                        # This file
└── requirements.txt                 # Dependencies (if needed)
```

## Expected Output

When you run the evaluation, you'll see progress bars like:

```
INFO:__main__:Loading multitask LSTM model: ../models/lstm.pt
INFO:__main__:Model loaded on device: cuda
INFO:__main__:Vocabulary size: 10000
INFO:__main__:Starting evaluation...
INFO:__main__:Evaluating simple_agrmt...
simple_agrmt/match_agree: 100%|████████| 48/48 [00:23<00:00, 2.05it/s]
...
INFO:__main__:Results saved to ../EMNLP2018/templates/lstm_multitask_results.pickle
```

Analysis output will show accuracy for each test case.

## Citation

If you use this evaluation code or the multitask LSTM model, please cite:

```
@inproceedings{marvin2018targeted,
  title={Targeted Syntactic Evaluation of Language Models},
  author={Marvin, Rebecca and Linzen, Tal},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={1192--1202},
  year={2018}
}
```

## References

- Original paper: https://aclanthology.org/D18-1151/
- Test templates from: EMNLP2018/templates/

## Contact

For questions about this evaluation code, refer to the main project README.
