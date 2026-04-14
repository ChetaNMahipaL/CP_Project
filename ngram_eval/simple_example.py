"""
Quick example script for n-gram evaluation.
Demonstrates how to score sentences with SRILM n-gram model.
"""

import sys
import os

sys.path.insert(0, '.')

from utils import compare_two_sentences, get_sentence_score

def main():
    SRILM_BIN = './ngram'
    MODEL_FILE = '../models/ngram_model.lm'
    VOCAB_FILE = '../models/vocab.txt'
    
    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        print("Please train the model first:")
        print("  bash train_ngram.sh")
        return
    
    print("N-gram Language Model - Quick Examples")
    print("="*80)
    
    # Example 1: Simple agreement
    print("\nExample 1: Simple Agreement")
    print("-"*80)
    sent_gram = "The dog walks."
    sent_ungram = "The dog walk."
    compare_two_sentences(SRILM_BIN, MODEL_FILE, VOCAB_FILE, sent_gram, sent_ungram)
    
    # Example 2: Complex agreement (object relative)
    print("\n\nExample 2: Object Relative Clause (with attractor)")
    print("-"*80)
    sent_gram = "The dog that the cat chased walks."
    sent_ungram = "The dog that the cat chased walk."
    compare_two_sentences(SRILM_BIN, MODEL_FILE, VOCAB_FILE, sent_gram, sent_ungram)
    
    # Example 3: NPI
    print("\n\nExample 3: Negative Polarity Item (NPI)")
    print("-"*80)
    sent_gram = "No one has ever seen a unicorn."
    sent_ungram = "Someone has ever seen a unicorn."
    compare_two_sentences(SRILM_BIN, MODEL_FILE, VOCAB_FILE, sent_gram, sent_ungram)
    
    # Example 4: Reflexive
    print("\n\nExample 4: Reflexive Binding")
    print("-"*80)
    sent_gram = "The boy said that the girl likes herself."
    sent_ungram = "The boy said that the girl likes himself."
    compare_two_sentences(SRILM_BIN, MODEL_FILE, VOCAB_FILE, sent_gram, sent_ungram)
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)

if __name__ == '__main__':
    main()
