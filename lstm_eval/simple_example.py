#!/usr/bin/env python3
"""
Simple example of evaluating LSTM LM on syntactic constructions.
This demonstrates the basic workflow for testing the model.
"""

import os
import sys
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Paths
    MODEL_FILE = '../models/lstm_lm.pt'
    LM_DATA_FILE = '../models/lstm_lm.bin'
    TEMPLATE_DIR = '../EMNLP2018/templates'
    RESULTS_FILE = f'{TEMPLATE_DIR}/lstm_results.pickle'
    OUTPUT_DIR = '../results/lstm'
    
    print("="*80)
    print("LSTM Language Model - Syntactic Evaluation Example")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print(f"\nERROR: Model file not found: {MODEL_FILE}")
        print("Please train the model first:")
        print("  cd ../word-language-model")
        print("  python main.py --lm_data ../data/lm_data \\")
        print("    --save ../models/lstm_lm.pt --save_lm_data ../models/lstm_lm.bin")
        return
    
    if not os.path.exists(LM_DATA_FILE):
        print(f"\nERROR: LM data file not found: {LM_DATA_FILE}")
        print("Please train the model first (see instructions above)")
        return
    
    print("\n✓ Model files found")
    print(f"  - Model: {MODEL_FILE}")
    print(f"  - LM Data: {LM_DATA_FILE}")
    
    # Check if templates exist
    if not os.path.exists(TEMPLATE_DIR):
        print(f"\nWARNING: Template directory not found: {TEMPLATE_DIR}")
        print("Generating templates...")
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        sys.path.insert(0, '../src')
        from make_templates import make_all_templates
        # Note: Adjust this import based on actual implementation
    
    print(f"✓ Template directory found: {TEMPLATE_DIR}")
    
    # Check if results already exist
    if os.path.exists(RESULTS_FILE):
        print(f"\n✓ Results file found: {RESULTS_FILE}")
        print("\nSkipping evaluation (already done)")
        print("To re-run evaluation, delete:", RESULTS_FILE)
    else:
        print("\nStep 1: Evaluating LSTM LM on syntactic constructions...")
        print("-"*80)
        print("This will test the model on:")
        print("  • Subject-verb agreement")
        print("  • Relative clauses")
        print("  • NPI licensing")
        print("  • And more...")
        print()
        
        import subprocess
        cmd = [
            'python', 'lstm_eval.py',
            '--template_dir', TEMPLATE_DIR,
            '--model', MODEL_FILE,
            '--lm_data', LM_DATA_FILE,
            '--output_file', 'lstm_results.pickle',
            '--device', 'cuda'
        ]
        
        print("Running:", ' '.join(cmd))
        print()
        
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print("ERROR: Evaluation failed!")
            return
        
        print("\n✓ Evaluation complete")
    
    # Analyze results
    print("\n" + "="*80)
    print("Step 2: Analyzing results...")
    print("-"*80)
    
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        return
    
    import subprocess
    cmd = [
        'python', 'analyze_lstm_results.py',
        '--results_file', RESULTS_FILE,
        '--output_dir', OUTPUT_DIR,
        '--mode', 'overall'
    ]
    
    print("Running:", ' '.join(cmd))
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode == 0:
        print("\n✓ Analysis complete")
        print(f"Results saved to: {OUTPUT_DIR}")
    
    # Show quick summary
    print("\n" + "="*80)
    print("Quick Summary")
    print("="*80)
    
    try:
        with open(RESULTS_FILE, 'rb') as f:
            results = pickle.load(f)
        
        total_correct = 0
        total_tests = 0
        
        print(f"\n{'Test Case':35} {'Correct':>10}")
        print("-" * 60)
        
        for test_name in sorted(results.keys()):
            test_data = results[test_name]
            correct = 0
            total = 0
            
            for key, scores_list in test_data.items():
                for i in range(0, len(scores_list), 2):
                    if i + 1 >= len(scores_list):
                        break
                    
                    gram_sent = scores_list[i]
                    ungram_sent = scores_list[i + 1]
                    
                    gram_score = sum(score for _, score in gram_sent)
                    ungram_score = sum(score for _, score in ungram_sent)
                    
                    total += 1
                    if gram_score > ungram_score:
                        correct += 1
            
            if total > 0:
                accuracy = (correct / total) * 100
                print(f"{test_name:35} {correct:>10}/{total}")
                total_correct += correct
                total_tests += total
        
        print("-" * 60)
        overall_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
        print(f"{'OVERALL':35} {overall_acc:>9.1f}% {total_correct}/{total_tests}")
        
    except Exception as e:
        logger.warning(f"Could not display summary: {e}")
    
    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. View detailed analysis: cat ../results/lstm/analysis_overall.txt")
    print("  2. Compare with other models (multitask LSTM, GPT-2, N-gram)")
    print("  3. Check relative clause results: ../results/lstm/analysis_full.txt")

if __name__ == '__main__':
    main()
