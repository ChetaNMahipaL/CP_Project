import os
import sys
import pickle
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze NPI surprisal for H2")
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the model results pickle file')
    return parser.parse_args()

def load_results(results_file):
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def extract_npi_surprisal(results):
    npi_analysis = {
        'simple_npi': {'licensed_s': [], 'unlicensed_s': [], 'delta_s': []},
        'rc_npi': {'licensed_s': [], 'unlicensed_s': [], 'delta_s': []}
    }
    
    # Isolate only the NPI tests
    npi_tests = {k: v for k, v in results.items() if 'npi' in k.lower()}
    
    if not npi_tests:
        print("Warning: No NPI test cases found in this pickle file.")
        return npi_analysis

    for test_name, test_data in npi_tests.items():
        category = 'simple_npi' if 'simple' in test_name.lower() else 'rc_npi'
        
        for key, sentences in test_data.items():
            for i in range(0, len(sentences), 2):
                if i + 1 >= len(sentences):
                    break
                
                gram_sent = sentences[i]
                ungram_sent = sentences[i + 1]
                
                # Find the log probability for the word 'ever'
                gram_ever_logp = next((score for word, score in gram_sent if word.lower().strip(".,!?") == 'ever'), None)
                ungram_ever_logp = next((score for word, score in ungram_sent if word.lower().strip(".,!?") == 'ever'), None)
                
                if gram_ever_logp is not None and ungram_ever_logp is not None:
                    # Surprisal = -log_prob
                    s_licensed = -gram_ever_logp
                    s_unlicensed = -ungram_ever_logp
                    delta_s = s_unlicensed - s_licensed
                    
                    npi_analysis[category]['licensed_s'].append(s_licensed)
                    npi_analysis[category]['unlicensed_s'].append(s_unlicensed)
                    npi_analysis[category]['delta_s'].append(delta_s)

    return npi_analysis

def main():
    args = parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: File {args.results_file} not found.")
        sys.exit(1)
        
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    npi_analysis = extract_npi_surprisal(results)
    
    print("\n" + "="*80)
    print(f"H2: NPI LICENSING SURPRISAL ANALYSIS ({os.path.basename(args.results_file)})")
    print("="*80)
    print(f"{'Condition':<20} | {'Mean S (Licensed)':<17} | {'Mean S (Unlicensed)':<20} | {'Mean ΔS':<10}")
    print("-" * 80)
    
    overall_delta_s = []

    for category, data in npi_analysis.items():
        if data['delta_s']:
            mean_lic = np.mean(data['licensed_s'])
            mean_unlic = np.mean(data['unlicensed_s'])
            mean_delta = np.mean(data['delta_s'])
            overall_delta_s.extend(data['delta_s'])
            
            print(f"{category:<20} | {mean_lic:<17.4f} | {mean_unlic:<20.4f} | {mean_delta:<10.4f}")
        else:
            print(f"{category:<20} | {'N/A':<17} | {'N/A':<20} | {'N/A':<10}")

    print("-" * 80)
    if overall_delta_s:
        print(f"{'OVERALL NPI':<20} | {'-':<17} | {'-':<20} | {np.mean(overall_delta_s):<10.4f}")
    print("="*80)

if __name__ == '__main__':
    main()