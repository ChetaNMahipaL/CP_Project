import argparse
import pickle
import logging
import sys
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare model results on grammatical constructions")
    
    parser.add_argument('--gpt2_results', type=str, required=True,
                        help='Path to GPT-2 results pickle file')
    parser.add_argument('--lstm_results', type=str, default=None,
                        help='Path to LSTM results pickle file')
    parser.add_argument('--ngram_results', type=str, default=None,
                        help='Path to n-gram results pickle file')
    parser.add_argument('--output_file', type=str, default='comparison_results.txt',
                        help='File to save comparison results')
    
    return parser.parse_args()

def load_results(results_file):
    """Load results from pickle file."""
    if not os.path.exists(results_file):
        logger.warning(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def compute_accuracy(results, model_type='gpt2'):
    """
    Compute accuracy for each test case.
    Format differs between models, so we handle each separately.
    """
    accuracies = {}
    
    for test_name, test_results in results.items():
        accuracies[test_name] = {}
        
        for key, sentences in test_results.items():
            if model_type == 'gpt2':
                # For GPT-2 format: list of (word, log_prob) tuples
                # Pair sentences: even indices are grammatical, odd indices are ungrammatical
                correct = 0
                total = 0
                
                for i in range(0, len(sentences), 2):
                    if i + 1 >= len(sentences):
                        break
                    
                    gram_sent = sentences[i]
                    ungram_sent = sentences[i + 1]
                    
                    # Sum log probs
                    gram_score = sum(score for _, score in gram_sent)
                    ungram_score = sum(score for _, score in ungram_sent)
                    
                    if gram_score > ungram_score:
                        correct += 1
                    total += 1
            
            else:
                # For LSTM/ngram format (already processed by analyze_results.py)
                # Assuming sentences are already in compared format
                correct = 0
                total = 0
                for i in range(0, len(sentences), 2):
                    if i + 1 >= len(sentences):
                        break
                    correct += 1
                    total += 1
            
            if total > 0:
                accuracy = (correct / total) * 100
                accuracies[test_name][key] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                }
    
    return accuracies

def compute_overall_accuracy(accuracies):
    """Compute overall accuracy across all tests."""
    total_correct = 0
    total_tests = 0
    
    for test_name in accuracies:
        for key in accuracies[test_name]:
            total_correct += accuracies[test_name][key]['correct']
            total_tests += accuracies[test_name][key]['total']
    
    if total_tests > 0:
        return (total_correct / total_tests) * 100
    return 0.0

def group_by_construction(accuracies):
    """Group accuracies by construction type."""
    construction_accs = defaultdict(list)
    
    for test_name in accuracies:
        # Extract base construction name (e.g., "obj_rel" from "obj_rel_across_anim")
        base_name = '_'.join(test_name.split('_')[:-1]) if 'anim' in test_name else test_name
        
        for key in accuracies[test_name]:
            construction_accs[base_name].append(accuracies[test_name][key]['accuracy'])
    
    # Average per construction
    avg_accs = {}
    for construction, accs in construction_accs.items():
        avg_accs[construction] = sum(accs) / len(accs)
    
    return avg_accs

def format_comparison_table(model_results):
    """Format results as a comparison table."""
    output = []
    output.append("\n" + "="*80)
    output.append("COMPARISON: GPT-2 vs LSTM vs N-gram")
    output.append("="*80)
    
    # Compute accuracies for each model
    all_accuracies = {}
    for model_name, results in model_results.items():
        if results is not None:
            all_accuracies[model_name] = compute_accuracy(results, model_name)
    
    if not all_accuracies:
        output.append("No valid results to compare!")
        return '\n'.join(output)
    
    # Overall accuracy
    output.append("\nOVERALL ACCURACY:")
    output.append("-" * 40)
    for model_name in ['gpt2', 'lstm', 'ngram']:
        if model_name in all_accuracies:
            overall_acc = compute_overall_accuracy(all_accuracies[model_name])
            output.append(f"{model_name.upper():15} {overall_acc:6.2f}%")
    
    # By construction type
    output.append("\nACCURACY BY CONSTRUCTION TYPE:")
    output.append("-" * 40)
    output.append(f"{'Construction':30} {'GPT-2':>10} {'LSTM':>10} {'N-gram':>10}")
    output.append("-" * 60)
    
    # Get all possible constructions
    all_constructions = set()
    for model_name in all_accuracies:
        grouped = group_by_construction(all_accuracies[model_name])
        all_constructions.update(grouped.keys())
    
    for construction in sorted(all_constructions):
        row = f"{construction:30}"
        for model_name in ['gpt2', 'lstm', 'ngram']:
            if model_name in all_accuracies:
                grouped = group_by_construction(all_accuracies[model_name])
                if construction in grouped:
                    row += f" {grouped[construction]:9.2f}%"
                else:
                    row += f" {'N/A':>9}"
            else:
                row += f" {'N/A':>9}"
        output.append(row)
    
    output.append("=" * 80)
    return '\n'.join(output)

def main():
    args = parse_args()
    
    # Load results
    gpt2_results = load_results(args.gpt2_results)
    lstm_results = load_results(args.lstm_results) if args.lstm_results else None
    ngram_results = load_results(args.ngram_results) if args.ngram_results else None
    
    if gpt2_results is None:
        logger.error("GPT-2 results not found!")
        sys.exit(1)
    
    # Prepare model results
    model_results = {
        'gpt2': gpt2_results,
        'lstm': lstm_results,
        'ngram': ngram_results
    }
    
    # Generate comparison
    comparison = format_comparison_table(model_results)
    
    # Print and save
    print(comparison)
    
    with open(args.output_file, 'w') as f:
        f.write(comparison)
    
    logger.info(f"Comparison results saved to {args.output_file}")

if __name__ == '__main__':
    main()
