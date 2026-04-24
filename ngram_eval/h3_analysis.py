import os
import sys
import pickle
import logging
import argparse
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze H3 N-gram using LSTM data as a Rosetta Stone")
    
    parser.add_argument('--ngram_file', type=str, required=True,
                        help='Path to n-gram results pickle file (the numbers)')
    parser.add_argument('--lstm_file', type=str, required=True,
                        help='Path to LSTM results pickle file (the text strings)')
    parser.add_argument('--output_dir', type=str, default='../h3_results',
                        help='Directory to save H3 analysis results')
    
    return parser.parse_args()

def load_data(filepath, desc="data"):
    logger.info(f"Loading {desc} from {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def compute_ngram_delta_s(gram_scores, ungram_scores):
    """
    Compute Delta S for the n-gram using probabilities.
    Finds the critical region where probabilities or token IDs diverge.
    """
    for i, ((g_idx, g_prob), (ug_idx, ug_prob)) in enumerate(zip(gram_scores, ungram_scores)):
        g_prob = float(g_prob)
        ug_prob = float(ug_prob)
        
        if str(g_idx) != str(ug_idx) or abs(g_prob - ug_prob) > 1e-7:
            s_gram = -g_prob
            s_ungram = -ug_prob
            delta_s = s_ungram - s_gram
            is_correct = delta_s > 0
            return is_correct, delta_s
            
    return False, 0.0

def extract_target_from_lstm(lstm_gram_sent, lstm_ungram_sent):
    """Find the critical grammatical word using the LSTM text data."""
    for (g_word, _), (ug_word, _) in zip(lstm_gram_sent, lstm_ungram_sent):
        if g_word != ug_word:
            return g_word.strip('.,!?;: \t\n\r').lower()
    return None

def analyze_aligned_h3(ngram_results, lstm_results):
    """
    Align the n-gram probabilities with the LSTM text to evaluate H3.
    """
    analysis = {
        'herself': {'deltas': [], 'correct': 0, 'total': 0},
        'himself_themselves': {'deltas': [], 'correct': 0, 'total': 0}
    }
    
    herself_targets = {'herself'}
    others_targets = {'himself', 'themselves'}
    
    for test_name, ngram_test_data in ngram_results.items():
        # Only process reflexive tests to save time
        if 'reflexive' not in test_name.lower():
            continue
            
        if test_name not in lstm_results:
            logger.warning(f"Test {test_name} found in n-gram but missing in LSTM. Skipping.")
            continue
            
        lstm_test_data = lstm_results[test_name]
        
        for key, ngram_scores in ngram_test_data.items():
            if key not in lstm_test_data:
                continue
                
            lstm_sentences = lstm_test_data[key]
            
            # Ensure the lengths match before iterating
            if len(ngram_scores) != len(lstm_sentences):
                logger.warning(f"Mismatch in pair counts for {test_name} -> {key}. Skipping.")
                continue
                
            for i in range(0, len(ngram_scores), 2):
                if i + 1 >= len(ngram_scores): break
                
                # 1. Figure out WHICH pronoun it is using the LSTM text
                target_word = extract_target_from_lstm(lstm_sentences[i], lstm_sentences[i+1])
                
                if target_word in herself_targets:
                    group = 'herself'
                elif target_word in others_targets:
                    group = 'himself_themselves'
                else:
                    continue # Not a target pronoun
                
                # 2. Do the actual math using the N-GRAM probabilities
                is_correct, delta_s = compute_ngram_delta_s(ngram_scores[i], ngram_scores[i+1])
                
                analysis[group]['total'] += 1
                analysis[group]['deltas'].append(delta_s)
                if is_correct:
                    analysis[group]['correct'] += 1
                        
    return analysis

def print_h3_report(analysis):
    output = []
    output.append("\n" + "="*80)
    output.append("H3: REFLEXIVE ASYMMETRY ANALYSIS - N-GRAM (Rosetta Stone Method)")
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
    
    if analysis['herself']['total'] > 0 and analysis['himself_themselves']['total'] > 0:
        mean_herself = sum(analysis['herself']['deltas']) / analysis['herself']['total']
        mean_others = sum(analysis['himself_themselves']['deltas']) / analysis['himself_themselves']['total']
        diff = mean_others - mean_herself
        
        output.append("\nHYPOTHESIS CHECK:")
        output.append(f"Difference in Mean ΔS (Others - Herself): {diff:.4f}")
        if diff > 0:
            output.append("-> CONCLUSION: 'herself' has a LOWER surprisal magnitude than himself/themselves.")
        else:
            output.append("-> CONCLUSION: 'herself' has an EQUAL or HIGHER surprisal magnitude.")
            
    output.append("="*80)
    return '\n'.join(output)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    ngram_results = load_data(args.ngram_file, "n-gram results")
    lstm_results = load_data(args.lstm_file, "LSTM results")
    
    analysis = analyze_aligned_h3(ngram_results, lstm_results)
    output = print_h3_report(analysis)
    
    print(output)
    
    output_file = os.path.join(args.output_dir, 'h3_ngram_report.txt')
    with open(output_file, 'w') as f:
        f.write(output)

if __name__ == '__main__':
    main()