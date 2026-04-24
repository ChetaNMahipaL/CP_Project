import argparse
import pickle
import logging
import sys
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze n-gram evaluation results using Critical-Region Surprisal")
    
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to n-gram results pickle file')
    parser.add_argument('--output_dir', type=str, default='../h1_results/ngram',
                        help='Directory to save analysis results')
    parser.add_argument('--mode', type=str, default='overall',
                        help='Level of detail (overall/condensed/full)')
    
    return parser.parse_args()

def load_results(results_file):
    """Load results from pickle file."""
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def compute_critical_region_surprisal(gram_scores, ungram_scores):
    """
    Compute accuracy based on critical-region surprisal difference (Delta S).
    Detects the critical region by looking for diverging token IDs OR diverging probabilities.
    """
    for i, ((gram_idx, gram_prob), (ungram_idx, ungram_prob)) in enumerate(zip(gram_scores, ungram_scores)):
        
        # Ensure we are working with floats
        g_prob = float(gram_prob)
        ug_prob = float(ungram_prob)

        # Detect the critical region if the token IDs differ 
        # OR if the log probabilities diverge by a tiny margin
        if str(gram_idx) != str(ungram_idx) or abs(g_prob - ug_prob) > 1e-7:
            
            # Surprisal is the negative log probability
            s_gram = -g_prob
            s_ungram = -ug_prob
            
            # Delta S = S(ungrammatical) - S(grammatical)
            delta_s = s_ungram - s_gram
            
            # Model correctly prefers the grammatical word if Delta S > 0
            is_correct = delta_s > 0
            
            return is_correct, delta_s, gram_idx, ungram_idx, i
            
    # Fallback if the sentences are somehow completely identical
    return False, 0.0, None, None, -1

def analyze_results(results):
    """
    Analyze n-gram results using Critical-Region Surprisal across all test cases.
    Format: {test_name: {key: [(word_idx, log_prob), ...]}}
    """
    analysis = {}
    
    for test_name, test_data in results.items():
        analysis[test_name] = {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'avg_delta_s': 0.0,
            'correct_pairs': [],
            'incorrect_pairs': []
        }
        
        total_delta_s = 0.0
        
        for key, scores_list in test_data.items():
            # Process sentence pairs (even indices = grammatical, odd = ungrammatical)
            for i in range(0, len(scores_list), 2):
                if i + 1 >= len(scores_list):
                    break
                
                gram_scores = scores_list[i]
                ungram_scores = scores_list[i + 1]
                
                analysis[test_name]['total'] += 1
                
                # Apply the H1 metric
                is_correct, delta_s, g_idx, ug_idx, crit_idx = compute_critical_region_surprisal(gram_scores, ungram_scores)
                total_delta_s += delta_s
                
                # Package the result for output (formatting indices as strings)
                pair_data = {
                    'gram_tokens': [(str(idx), f"{s:.4f}") for idx, s in gram_scores],
                    'ungram_tokens': [(str(idx), f"{s:.4f}") for idx, s in ungram_scores],
                    'critical_info': f"Index {crit_idx}: Token ID '{g_idx}' vs '{ug_idx}' (ΔS = {delta_s:.4f})"
                }
                
                if is_correct:
                    analysis[test_name]['correct'] += 1
                    analysis[test_name]['correct_pairs'].append(pair_data)
                else:
                    analysis[test_name]['incorrect_pairs'].append(pair_data)
        
        if analysis[test_name]['total'] > 0:
            analysis[test_name]['accuracy'] = (
                analysis[test_name]['correct'] / analysis[test_name]['total']
            ) * 100
            analysis[test_name]['avg_delta_s'] = total_delta_s / analysis[test_name]['total']
    
    return analysis

def print_overall_results(analysis):
    """Print overall results summary."""
    output = []
    output.append("\n" + "="*80)
    output.append("N-GRAM EVALUATION RESULTS - OVERALL (CRITICAL-REGION SURPRISAL)")
    output.append("="*80)
    
    total_correct = 0
    total_tests = 0
    
    output.append(f"\n{'Test Case':45} {'Accuracy':>10} {'Avg ΔS':>10} {'Correct':>10}")
    output.append("-" * 80)
    
    for test_name in sorted(analysis.keys()):
        acc = analysis[test_name]
        output.append(f"{test_name:45} {acc['accuracy']:9.2f}% {acc['avg_delta_s']:10.4f} {acc['correct']:>10}/{acc['total']}")
        total_correct += acc['correct']
        total_tests += acc['total']
    
    output.append("-" * 80)
    overall_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
    output.append(f"{'OVERALL':45} {overall_acc:9.2f}% {'-':>10} {total_correct:>10}/{total_tests}")
    output.append("="*80)
    
    return '\n'.join(output)

def print_condensed_results(analysis):
    """Print condensed results by grouping related test cases."""
    output = []
    output.append("\n" + "="*80)
    output.append("N-GRAM EVALUATION RESULTS - CONDENSED (CRITICAL-REGION SURPRISAL)")
    output.append("="*80)
    
    # Group by construction type
    grouped = defaultdict(lambda: {'correct': 0, 'total': 0, 'sum_delta_s': 0.0})
    
    for test_name, acc in analysis.items():
        # Extract base name (remove anim/inanim suffix)
        base_name = '_'.join(test_name.split('_')[:-1]) if 'anim' in test_name else test_name
        grouped[base_name]['correct'] += acc['correct']
        grouped[base_name]['total'] += acc['total']
        grouped[base_name]['sum_delta_s'] += (acc['avg_delta_s'] * acc['total'])
    
    output.append(f"\n{'Construction':40} {'Accuracy':>10} {'Avg ΔS':>10} {'Correct':>10}")
    output.append("-" * 80)
    
    total_correct = 0
    total_tests = 0
    
    for construction in sorted(grouped.keys()):
        correct = grouped[construction]['correct']
        total = grouped[construction]['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_delta_s = (grouped[construction]['sum_delta_s'] / total) if total > 0 else 0.0
        
        output.append(f"{construction:40} {accuracy:9.2f}% {avg_delta_s:10.4f} {correct:>10}/{total}")
        total_correct += correct
        total_tests += total
    
    output.append("-" * 80)
    overall_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
    output.append(f"{'OVERALL':40} {overall_acc:9.2f}% {'-':>10} {total_correct:>10}/{total_tests}")
    output.append("="*80)
    
    return '\n'.join(output)

def print_full_results(analysis):
    """Print detailed results with sample sentences."""
    output = []
    output.append("\n" + "="*90)
    output.append("N-GRAM EVALUATION RESULTS - FULL (CRITICAL-REGION SURPRISAL)")
    output.append("="*90)
    
    for test_name in sorted(analysis.keys()):
        acc = analysis[test_name]
        output.append(f"\n{test_name}: {acc['accuracy']:.2f}% (Avg ΔS: {acc['avg_delta_s']:.4f}) ({acc['correct']}/{acc['total']})")
        output.append("-" * 90)
        
        # Show some correct examples
        if acc['correct_pairs']:
            output.append("CORRECT PREDICTIONS (showing first 3):")
            for i, pair in enumerate(acc['correct_pairs'][:3]):
                output.append(f"\n  Example {i+1}: [{pair['critical_info']}]")
                gram_str = ' '.join([f"ID_{w}({s})" for w, s in pair['gram_tokens']])
                ungram_str = ' '.join([f"ID_{w}({s})" for w, s in pair['ungram_tokens']])
                output.append(f"    Grammatical:   {gram_str}")
                output.append(f"    Ungrammatical: {ungram_str}")
        
        # Show some incorrect examples
        if acc['incorrect_pairs']:
            output.append("\nINCORRECT PREDICTIONS (showing first 3):")
            for i, pair in enumerate(acc['incorrect_pairs'][:3]):
                output.append(f"\n  Example {i+1}: [{pair['critical_info']}]")
                gram_str = ' '.join([f"ID_{w}({s})" for w, s in pair['gram_tokens']])
                ungram_str = ' '.join([f"ID_{w}({s})" for w, s in pair['ungram_tokens']])
                output.append(f"    Grammatical:   {gram_str}")
                output.append(f"    Ungrammatical: {ungram_str}")
    
    output.append("\n" + "="*90)
    return '\n'.join(output)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and analyze results
    results = load_results(args.results_file)
    analysis = analyze_results(results)
    
    # Generate output based on mode
    if args.mode == 'overall':
        output = print_overall_results(analysis)
    elif args.mode == 'condensed':
        output = print_condensed_results(analysis)
    elif args.mode == 'full':
        output = print_full_results(analysis)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return
    
    # Print and save
    print(output)
    
    output_file = os.path.join(args.output_dir, f'analysis_{args.mode}.txt')
    with open(output_file, 'w') as f:
        f.write(output)
    
    logger.info(f"Analysis saved to {output_file}")

if __name__ == '__main__':
    main()