import os
import sys
import pickle
import logging
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze H4: GPT-2 Entropy Reduction")
    parser.add_argument('--results_file', type=str, default='./gpt2_entropy_results.pickle',
                        help='Path to GPT-2 entropy results pickle file')
    parser.add_argument('--output_dir', type=str, default='../h4_results/gpt2',
                        help='Directory to save analysis text output')
    return parser.parse_args()

def load_results(results_file):
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def find_critical_index(gram_sent, ungram_sent, test_name):
    """Find the BPE token index of the critical region."""
    if 'npi' in test_name.lower():
        for i, (word, _, _) in enumerate(gram_sent):
            # GPT-2 tokenizes words with a leading 'Ġ' for spaces. Strip it for matching.
            clean_word = word.replace('Ġ', '').replace('ġ', '').lower().strip()
            if clean_word == 'ever':
                return i
                
    for i in range(min(len(gram_sent), len(ungram_sent))):
        if gram_sent[i][0] != ungram_sent[i][0]:
            return i
            
    return -1

def calculate_pair_er(gram_sent, ungram_sent, test_name):
    n = find_critical_index(gram_sent, ungram_sent, test_name)
    
    if n < 0 or n + 1 >= len(gram_sent) or n + 1 >= len(ungram_sent):
        return None
        
    # H_before: Entropy BEFORE reading the critical word.
    # We grab this independently for gram and ungram to account for NPI prefixes
    # differing ("No" vs "Most"), meaning the models approach the critical region in different states.
    H_before_gram   = gram_sent[n][2]
    H_before_ungram = ungram_sent[n][2]
    
    # H_after: Entropy AFTER reading the critical word (diverges here)
    H_after_gram   = gram_sent[n + 1][2]
    H_after_ungram = ungram_sent[n + 1][2]
    
    # ER(w_n) = max(H_{n-1} - H_n, 0)
    ER_gram   = max(H_before_gram - H_after_gram,   0)
    ER_ungram = max(H_before_ungram - H_after_ungram, 0)

    # H1 cross-reference: surprisal at critical word
    s_gram   = -gram_sent[n][1]
    s_ungram = -ungram_sent[n][1]
    delta_s  = s_ungram - s_gram

    return ER_gram, ER_ungram, delta_s

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_file)
    
    er_stats = defaultdict(lambda: {
        'sum_gram'  : 0.0,
        'sum_ungram': 0.0,
        'sum_delta_s': 0.0,
        'h4_correct': 0,    
        'h1_correct': 0,    
        'count'     : 0,
    })
    
    for test_name, test_data in results.items():
        base_name = '_'.join(test_name.split('_')[:-1]) if 'anim' in test_name else test_name
        
        for key, sentences in test_data.items():
            for i in range(0, len(sentences), 2):
                if i + 1 >= len(sentences):
                    break
                
                gram_sent   = sentences[i]
                ungram_sent = sentences[i + 1]
                
                result = calculate_pair_er(gram_sent, ungram_sent, test_name)
                if result is None:
                    continue

                er_gram, er_ungram, delta_s = result

                er_stats[base_name]['sum_gram']   += er_gram
                er_stats[base_name]['sum_ungram'] += er_ungram
                er_stats[base_name]['sum_delta_s']+= delta_s
                er_stats[base_name]['count']      += 1

                if er_gram > er_ungram:
                    er_stats[base_name]['h4_correct'] += 1
                if delta_s > 0:
                    er_stats[base_name]['h1_correct'] += 1

    # Print results
    W = 105
    lines = []
    lines.append("\n" + "=" * W)
    lines.append("HYPOTHESIS 4: GPT-2 ENTROPY REDUCTION AT CRITICAL REGION")
    lines.append("=" * W)
    lines.append(
        "  Mean ER(gram)  : average entropy reduction when the GRAMMATICAL word is read\n"
        "  Mean ER(ungram): average entropy reduction when the UNGRAMMATICAL word is read\n"
        "  H4 Acc         : % of pairs where ER(gram) > ER(ungram)  [primary H4 metric]\n"
        "  H1 Acc         : % of pairs where S(ungram) > S(gram)    [cross-reference]\n"
        "  H4 Supported?  : YES if Mean ER(gram) > Mean ER(ungram)"
    )
    lines.append("=" * W)

    header = (
        f"\n{'Construction':<32} | "
        f"{'Mean ER(gram)':>14} | "
        f"{'Mean ER(ungram)':>15} | "
        f"{'H4 Acc':>8} | "
        f"{'H1 Acc':>8} | "
        f"{'H4 Supp?':>10} | "
        f"{'N':>6}"
    )
    lines.append(header)
    lines.append("-" * W)

    total_h4 = total_h1 = total_n = 0
    total_sum_gram = total_sum_ungram = 0.0

    for const, stats in sorted(er_stats.items()):
        n = stats['count']
        if n == 0:
            continue

        mean_gram   = stats['sum_gram']   / n
        mean_ungram = stats['sum_ungram'] / n
        h4_acc      = stats['h4_correct'] / n * 100
        h1_acc      = stats['h1_correct'] / n * 100
        supported   = "YES ✓" if mean_gram > mean_ungram else "NO  ✗"

        lines.append(
            f"{const:<32} | "
            f"{mean_gram:>14.4f} | "
            f"{mean_ungram:>15.4f} | "
            f"{h4_acc:>7.2f}% | "
            f"{h1_acc:>7.2f}% | "
            f"{supported:>10} | "
            f"{n:>6}"
        )

        total_h4          += stats['h4_correct']
        total_h1          += stats['h1_correct']
        total_n           += n
        total_sum_gram    += stats['sum_gram']
        total_sum_ungram  += stats['sum_ungram']

    lines.append("-" * W)
    if total_n > 0:
        lines.append(
            f"{'OVERALL':<32} | "
            f"{total_sum_gram/total_n:>14.4f} | "
            f"{total_sum_ungram/total_n:>15.4f} | "
            f"{total_h4/total_n*100:>7.2f}% | "
            f"{total_h1/total_n*100:>7.2f}% | "
            f"{'—':>10} | "
            f"{total_n:>6}"
        )
    lines.append("=" * W)

    output = '\n'.join(lines)
    print(output)

    out_file = os.path.join(args.output_dir, 'gpt2_h4_entropy_analysis.txt')
    with open(out_file, 'w') as f:
        f.write(output)
    logger.info(f"H4 analysis saved to {out_file}")

if __name__ == '__main__':
    main()