"""
Simple example script demonstrating GPT-2 evaluation.
Shows how to score individual sentences and minimal pairs.
"""

import sys
sys.path.insert(0, '..')

from utils import load_gpt2_model, compare_sentences

def main():
    print("Loading GPT-2 model...")
    model, tokenizer = load_gpt2_model('gpt2', device='cuda')
    
    # Example 1: Simple agreement
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Agreement")
    print("="*80)
    sent_gram = "The dog walks."
    sent_ungram = "The dog walk."
    compare_sentences(model, tokenizer, sent_gram, sent_ungram)
    
    # Example 2: Complex agreement (object relative clause)
    print("\n" + "="*80)
    print("EXAMPLE 2: Object Relative Clause (with attractor)")
    print("="*80)
    sent_gram = "The dog that the cat chased walks."
    sent_ungram = "The dog that the cat chased walk."
    compare_sentences(model, tokenizer, sent_gram, sent_ungram)
    
    # Example 3: Negative Polarity Item (NPI)
    print("\n" + "="*80)
    print("EXAMPLE 3: Negative Polarity Item (NPI)")
    print("="*80)
    sent_gram = "No one has ever seen a unicorn."
    sent_ungram = "Someone has ever seen a unicorn."
    compare_sentences(model, tokenizer, sent_gram, sent_ungram)
    
    # Example 4: Reflexive binding
    print("\n" + "="*80)
    print("EXAMPLE 4: Reflexive Binding")
    print("="*80)
    sent_gram = "The boy said that the girl likes herself."
    sent_ungram = "The boy said that the girl likes himself."
    compare_sentences(model, tokenizer, sent_gram, sent_ungram)
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)

if __name__ == '__main__':
    main()
