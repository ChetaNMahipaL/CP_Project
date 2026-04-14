import os
import sys
import pickle
import logging
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GPT-2 results in detail")
    
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to GPT-2 results pickle file')
    parser.add_argument('--output_dir', type=str, default='../results/gpt2',
                        help='Directory to save analysis results')
    parser.add_argument('--mode', type=str, default='overall',
                        help='Level of detail (overall/condensed/full)')
    
    return parser.parse_args()

def load_results(results_file):
    """Load results from pickle file."""
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def compute_sentence_accuracy(gram_sent, ungram_sent):
    """
    Compare grammatical vs ungrammatical sentence.
    Returns True if grammatical has higher probability.
    """
    gram_score = sum(score for _, score in gram_sent)
    ungram_score = sum(score for _, score in ungram_sent)
    return gram_score > ungram_score

def analyze_results(results):
    """
    Analyze GPT-2 results across all test cases.
    Format: {test_name: {key: [word_scores]}}
    """
    analysis = {}
    
    for test_name, test_data in results.items():
        analysis[test_name] = {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'correct_pairs': [],
            'incorrect_pairs': []
        }
        
        for key, sentences in test_data.items():
            # Process sentence pairs (even indices = grammatical, odd = ungrammatical)
            for i in range(0, len(sentences), 2):
                if i + 1 >= len(sentences):
                    break
                
                gram_sent = sentences[i]
                ungram_sent = sentences[i + 1]
                
                analysis[test_name]['total'] += 1
                
                if compute_sentence_accuracy(gram_sent, ungram_sent):
                    analysis[test_name]['correct'] += 1
                    analysis[test_name]['correct_pairs'].append({
                        'gram_tokens': [(w, f"{s:.4f}") for w, s in gram_sent],
                        'ungram_tokens': [(w, f"{s:.4f}") for w, s in ungram_sent]
                    })
                else:
                    analysis[test_name]['incorrect_pairs'].append({
                        'gram_tokens': [(w, f"{s:.4f}") for w, s in gram_sent],
                        'ungram_tokens': [(w, f"{s:.4f}") for w, s in ungram_sent]
                    })
        
        if analysis[test_name]['total'] > 0:
            analysis[test_name]['accuracy'] = (
                analysis[test_name]['correct'] / analysis[test_name]['total']
            ) * 100
    
    return analysis

def print_overall_results(analysis):
    """Print overall accuracy summary."""
    output = []
    output.append("\n" + "="*60)
    output.append("GPT-2 EVALUATION RESULTS - OVERALL")
    output.append("="*60)
    
    total_correct = 0
    total_tests = 0
    
    output.append(f"\n{'Test Case':35} {'Accuracy':>10} {'Correct':>10}")
    output.append("-" * 60)
    
    for test_name in sorted(analysis.keys()):
        acc = analysis[test_name]
        output.append(f"{test_name:35} {acc['accuracy']:9.2f}% {acc['correct']:>10}/{acc['total']}")
        total_correct += acc['correct']
        total_tests += acc['total']
    
    output.append("-" * 60)
    overall_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
    output.append(f"{'OVERALL':35} {overall_acc:9.2f}% {total_correct:>10}/{total_tests}")
    output.append("="*60)
    
    return '\n'.join(output)

def print_condensed_results(analysis):
    """Print condensed results by grouping related test cases."""
    output = []
    output.append("\n" + "="*60)
    output.append("GPT-2 EVALUATION RESULTS - CONDENSED")
    output.append("="*60)
    
    # Group by construction type
    grouped = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for test_name, acc in analysis.items():
        # Extract base name (remove anim/inanim suffix)
        base_name = '_'.join(test_name.split('_')[:-1]) if 'anim' in test_name else test_name
        grouped[base_name]['correct'] += acc['correct']
        grouped[base_name]['total'] += acc['total']
    
    output.append(f"\n{'Construction':30} {'Accuracy':>10} {'Correct':>10}")
    output.append("-" * 60)
    
    total_correct = 0
    total_tests = 0
    
    for construction in sorted(grouped.keys()):
        correct = grouped[construction]['correct']
        total = grouped[construction]['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        output.append(f"{construction:30} {accuracy:9.2f}% {correct:>10}/{total}")
        total_correct += correct
        total_tests += total
    
    output.append("-" * 60)
    overall_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
    output.append(f"{'OVERALL':30} {overall_acc:9.2f}% {total_correct:>10}/{total_tests}")
    output.append("="*60)
    
    return '\n'.join(output)

def print_full_results(analysis):
    """Print detailed results with sample sentences."""
    output = []
    output.append("\n" + "="*80)
    output.append("GPT-2 EVALUATION RESULTS - FULL (WITH SAMPLES)")
    output.append("="*80)
    
    for test_name in sorted(analysis.keys()):
        acc = analysis[test_name]
        output.append(f"\n{test_name}: {acc['accuracy']:.2f}% ({acc['correct']}/{acc['total']})")
        output.append("-" * 80)
        
        # Show some correct examples
        if acc['correct_pairs']:
            output.append("CORRECT PREDICTIONS (showing first 3):")
            for i, pair in enumerate(acc['correct_pairs'][:3]):
                output.append(f"\n  Example {i+1}:")
                gram_str = ' '.join([f"{w}({s})" for w, s in pair['gram_tokens']])
                ungram_str = ' '.join([f"{w}({s})" for w, s in pair['ungram_tokens']])
                output.append(f"    Grammatical:   {gram_str}")
                output.append(f"    Ungrammatical: {ungram_str}")
        
        # Show some incorrect examples
        if acc['incorrect_pairs']:
            output.append("\nINCORRECT PREDICTIONS (showing first 3):")
            for i, pair in enumerate(acc['incorrect_pairs'][:3]):
                output.append(f"\n  Example {i+1}:")
                gram_str = ' '.join([f"{w}({s})" for w, s in pair['gram_tokens']])
                ungram_str = ' '.join([f"{w}({s})" for w, s in pair['ungram_tokens']])
                output.append(f"    Grammatical:   {gram_str}")
                output.append(f"    Ungrammatical: {ungram_str}")
    
    output.append("\n" + "="*80)
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
