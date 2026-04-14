"""
Utility functions for GPT-2 evaluation
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt2_model(model_name='gpt2', device='cuda'):
    """Load GPT-2 model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def score_sentence(model, tokenizer, sentence, device='cuda'):
    """
    Score a sentence using GPT-2.
    Returns total log probability and per-token scores.
    """
    if not sentence.strip():
        return 0.0, []
    
    # Ensure sentence ends with period
    if not sentence.strip().endswith('.'):
        sentence = sentence.strip() + ' .'
    else:
        sentence = sentence.strip()
    
    tokens = tokenizer.encode(sentence, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        logits = outputs.logits
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    token_log_probs = []
    for i in range(len(tokens[0]) - 1):
        token_id = tokens[0, i + 1]
        token_log_prob = log_probs[0, i, token_id].item()
        token_log_probs.append(token_log_prob)
    
    total_score = sum(token_log_probs)
    return total_score, token_log_probs

def format_token_scores(tokens, scores):
    """
    Format token-level scores for display.
    Returns a formatted string showing each token with its score.
    """
    output = []
    for token, score in zip(tokens, scores):
        output.append(f"{token:15} {score:8.4f}")
    return '\n'.join(output)

def compare_sentences(model, tokenizer, sent1, sent2, device='cuda', verbose=True):
    """
    Compare two sentences and return which one has higher probability.
    """
    score1, tokens1 = score_sentence(model, tokenizer, sent1, device)
    score2, tokens2 = score_sentence(model, tokenizer, sent2, device)
    
    if verbose:
        print(f"\nSentence 1: {sent1}")
        print(f"Total Score: {score1:.4f}")
        print(format_token_scores(tokenizer.convert_ids_to_tokens(
            tokenizer.encode(sent1)), tokens1))
        
        print(f"\nSentence 2: {sent2}")
        print(f"Total Score: {score2:.4f}")
        print(format_token_scores(tokenizer.convert_ids_to_tokens(
            tokenizer.encode(sent2)), tokens2))
        
        print(f"\nPreferred: {'Sentence 1' if score1 > score2 else 'Sentence 2'}")
    
    return score1 > score2, score1, score2

def batch_score_sentences(model, tokenizer, sentences, device='cuda', batch_size=32):
    """
    Score multiple sentences efficiently in batches.
    Returns list of (total_score, token_scores) tuples.
    """
    results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # Tokenize batch
        encoded = [tokenizer.encode(s.strip() if s.strip().endswith('.') else s.strip() + ' .') 
                   for s in batch]
        
        # Pad to same length
        max_len = max(len(e) for e in encoded)
        padded = []
        for e in encoded:
            padded.append(e + [tokenizer.eos_token_id] * (max_len - len(e)))
        
        # Score
        tokens = torch.tensor(padded).to(device)
        
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        for j, tokens_seq in enumerate(padded):
            token_scores = []
            for k in range(len(tokens_seq) - 1):
                token_id = tokens_seq[k + 1]
                token_log_prob = log_probs[j, k, token_id].item()
                token_scores.append(token_log_prob)
            
            total_score = sum(token_scores)
            results.append((total_score, token_scores))
    
    return results
