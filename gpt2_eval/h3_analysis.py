import os
import sys
import pickle
import logging
import argparse
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze H3: Reflexive Asymmetry using Critical-Region Surprisal")
    
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to results pickle file')
    parser.add_argument('--output_dir', type=str, default='../h3_results',
                        help='Directory to save H3 analysis results')
    
    return parser.parse_args()

def load_results(results_file):
    """Load results from pickle file."""
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def compute_critical_region_surprisal(gram_sent, ungram_sent):
    """
    Compute accuracy based on critical-region surprisal difference (Delta S).
    S(w) = -log P(w)
    """
    for i, ((gram_w, gram_score), (ungram_w, ungram_score)) in enumerate(zip(gram_sent, ungram_sent)):
        if gram_w != ungram_w:
            # Surprisal is the negative log probability
            s_gram = -float(gram_score)
            s_ungram = -float(ungram_score)
            
            # Delta S = S(ungrammatical) - S(grammatical)
            delta_s = s_ungram - s_gram
            
            is_correct = delta_s > 0
            
            # CRITICAL FIX FOR GPT-2: Remove BPE 'Ġ', whitespace, and punctuation
            clean_g_word = gram_w.replace('Ġ', '').strip('.,!?;: \t\n\r').lower()
            clean_ug_word = ungram_w.replace('Ġ', '').strip('.,!?;: \t\n\r').lower()
            
            return is_correct, delta_s, clean_g_word, clean_ug_word, i
            
    return False, 0.0, None, None, -1

def analyze_h3_reflexives(results):
    """
    Isolate reflexive pairs and group them by the grammatical target:
    Group A: 'herself' (and subword 'her' just in case of weird tokenization)
    Group B: 'himself' or 'themselves' (and subwords 'him', 'them')
    """
    analysis = {
        'herself': {
            'deltas': [],
            'correct': 0,
            'total': 0
        },
        'himself_themselves': {
            'deltas': [],
            'correct': 0,
            'total': 0
        }
    }
    
    # Added the root prefixes just in case GPT-2 split the token (e.g., 'Ġhim' + 'self')
    herself_targets = {'herself', 'her'}
    others_targets = {'himself', 'themselves', 'him', 'them'}
    
    for test_name, test_data in results.items():
        for key, sentences in test_data.items():
            for i in range(0, len(sentences), 2):
                if i + 1 >= len(sentences):
                    break
                
                gram_sent = sentences[i]
                ungram_sent = sentences[i + 1]
                
                is_correct, delta_s, g_word, ug_word, crit_idx = compute_critical_region_surprisal(gram_sent, ungram_sent)
                
                # Categorize based on the cleaned grammatical word
                if g_word in herself_targets:
                    group = 'herself'
                    analysis[group]['total'] += 1
                    analysis[group]['deltas'].append(delta_s)
                    if is_correct:
                        analysis[group]['correct'] += 1
                        
                elif g_word in others_targets:
                    group = 'himself_themselves'
                    analysis[group]['total'] += 1
                    analysis[group]['deltas'].append(delta_s)
                    if is_correct:
                        analysis[group]['correct'] += 1
                        
    return analysis

def print_h3_report(analysis):
    """Format and print the comparative report for H3."""
    output = []
    output.append("\n" + "="*80)
    output.append("H3: REFLEXIVE ASYMMETRY ANALYSIS (CRITICAL-REGION SURPRISAL)")
    output.append("="*80)
    
    output.append(f"\n{'Target Pronoun Group':25} {'Accuracy':>10} {'Mean ΔS':>10} {'Std Dev':>10} {'N':>10}")
    output.append("-" * 80)
    
    for group in sorted(analysis.keys()):
        data = analysis[group]
        total = data['total']
        
        if total == 0:
            output.append(f"{group:25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>10}")
            continue
            
        acc = (data['correct'] / total) * 100
        mean_delta_s = sum(data['deltas']) / total
        std_delta_s = statistics.stdev(data['deltas']) if total > 1 else 0.0
        
        output.append(f"{group:25} {acc:9.2f}% {mean_delta_s:10.4f} {std_delta_s:10.4f} {total:>10}")
        
    output.append("-" * 80)
    
    # Calculate the difference to directly address the hypothesis
    if analysis['herself']['total'] > 0 and analysis['himself_themselves']['total'] > 0:
        mean_herself = sum(analysis['herself']['deltas']) / analysis['herself']['total']
        mean_others = sum(analysis['himself_themselves']['deltas']) / analysis['himself_themselves']['total']
        diff = mean_others - mean_herself
        
        output.append("\nHYPOTHESIS CHECK:")
        output.append(f"Difference in Mean ΔS (Others - Herself): {diff:.4f}")
        if diff > 0:
            output.append("-> CONCLUSION: 'herself' has a LOWER surprisal magnitude than himself/themselves.")
            output.append("-> H3 is SUPPORTED by this data.")
        else:
            output.append("-> CONCLUSION: 'herself' has an EQUAL or HIGHER surprisal magnitude.")
            output.append("-> H3 is NOT supported by this data.")
            
    output.append("="*80)
    return '\n'.join(output)

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = load_results(args.results_file)
    analysis = analyze_h3_reflexives(results)
    output = print_h3_report(analysis)
    
    print(output)
    
    output_file = os.path.join(args.output_dir, 'h3_report.txt')
    with open(output_file, 'w') as f:
        f.write(output)
    
    logger.info(f"H3 Analysis saved to {output_file}")

if __name__ == '__main__':
    main()