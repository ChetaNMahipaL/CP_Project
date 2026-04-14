import argparse
import pickle
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare n-gram with other models")
    
    parser.add_argument('--ngram_results', type=str, required=True,
                        help='Path to n-gram results pickle file')
    parser.add_argument('--gpt2_results', type=str, default=None,
                        help='Path to GPT-2 results pickle file')
    parser.add_argument('--lstm_results', type=str, default=None,
                        help='Path to LSTM results pickle file')
    parser.add_argument('--output_file', type=str, default='ngram_comparison.txt',
                        help='File to save comparison results')
    
    return parser.parse_args()

def load_results(results_file):
    """Load results from pickle file."""
    if not results_file or not os.path.exists(results_file):
        logger.warning(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def compute_accuracy(results, model_type='ngram'):
    """Compute accuracy for each test case."""
    accuracies = {}
    
    for test_name, test_results in results.items():
        accuracies[test_name] = {}
        
        for key, items in test_results.items():
            correct = 0
            total = 0
            
            # Process pairs
            for i in range(0, len(items), 2):
                if i + 1 >= len(items):
                    break
                
                item1 = items[i]
                item2 = items[i + 1]
                
                if model_type == 'ngram':
                    # N-gram format: [(word_idx, log_prob), ...]
                    score1 = sum(score for _, score in item1) if item1 else 0
                    score2 = sum(score for _, score in item2) if item2 else 0
                else:
                    # Assume same format for other models
                    score1 = sum(score for _, score in item1) if isinstance(item1, list) and item1 else 0
                    score2 = sum(score for _, score in item2) if isinstance(item2, list) and item2 else 0
                
                if score1 > score2:
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

def format_comparison_table(model_results):
    """Format results as a comparison table."""
    output = []
    output.append("\n" + "="*80)
    output.append("N-GRAM MODEL COMPARISON")
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
    
    model_names = ['ngram', 'gpt2', 'lstm']
    for model_name in model_names:
        if model_name in all_accuracies:
            overall_acc = compute_overall_accuracy(all_accuracies[model_name])
            output.append(f"{model_name.upper():15} {overall_acc:6.2f}%")
    
    # By test case
    output.append("\nACCURACY BY TEST CASE:")
    output.append("-" * 80)
    
    # Header
    header = f"{'Test Case':30}"
    for model_name in model_names:
        if model_name in all_accuracies:
            header += f" {model_name.upper():>12}"
    output.append(header)
    output.append("-" * 80)
    
    # Get all possible test cases
    all_tests = set()
    for model_name in all_accuracies:
        all_tests.update(all_accuracies[model_name].keys())
    
    for test_name in sorted(all_tests):
        row = f"{test_name:30}"
        for model_name in model_names:
            if model_name in all_accuracies and test_name in all_accuracies[model_name]:
                # Average accuracy for this test across all keys
                test_accs = [all_accuracies[model_name][test_name][key]['accuracy'] 
                            for key in all_accuracies[model_name][test_name]]
                avg_acc = sum(test_accs) / len(test_accs) if test_accs else 0
                row += f" {avg_acc:11.2f}%"
            else:
                row += f" {'N/A':>11}"
        output.append(row)
    
    output.append("=" * 80)
    return '\n'.join(output)

def main():
    args = parse_args()
    
    # Load results
    ngram_results = load_results(args.ngram_results)
    gpt2_results = load_results(args.gpt2_results) if args.gpt2_results else None
    lstm_results = load_results(args.lstm_results) if args.lstm_results else None
    
    if ngram_results is None:
        logger.error("N-gram results not found!")
        sys.exit(1)
    
    # Prepare model results
    model_results = {
        'ngram': ngram_results,
        'gpt2': gpt2_results,
        'lstm': lstm_results
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
