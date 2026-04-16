import torch
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lstm_model(model_path, lm_data_path, device='cuda'):
    """Load the trained multitask LSTM model and dictionary."""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(lm_data_path):
        raise FileNotFoundError(f"LM data file not found: {lm_data_path}")
    
    # Load LM data (contains dictionary)
    lm_data = torch.load(lm_data_path, map_location='cpu')
    dictionary = lm_data
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Import model class
    sys.path.insert(0, '../word-language-model')
    from model import RNNModel
    
    # Reconstruct model
    if isinstance(checkpoint, dict):
        model_args = checkpoint.get('args', None)
        if model_args is None:
            ntokens = len(dictionary)
            model = RNNModel('LSTM', ntokens, 400, 650, 2, dropout=0.2, tie_weights=False)
        else:
            ntokens = len(dictionary)
            model = RNNModel(model_args.model, ntokens, model_args.emsize, 
                            model_args.nhid, model_args.nlayers, 
                            dropout=model_args.dropout, tie_weights=model_args.tied)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    
    return model, dictionary

def sentence_to_indices(sentence, dictionary):
    """Convert sentence string to token indices using dictionary."""
    words = sentence.strip().lower().split()
    indices = []
    for word in words:
        if word in dictionary:
            indices.append(dictionary[word])
        else:
            indices.append(dictionary.get('<unk>', 0))
    return indices

def score_sentence(model, dictionary, sentence, device='cuda'):
    """Score a sentence using the LSTM model."""
    
    if not sentence.strip():
        return 0.0, []
    
    if not sentence.strip().endswith('.'):
        sentence = sentence.strip() + ' .'
    else:
        sentence = sentence.strip()
    
    try:
        indices = sentence_to_indices(sentence, dictionary)
        if not indices:
            return 0.0, []
        
        data = torch.LongTensor(indices).unsqueeze(1).to(device)
        
        with torch.no_grad():
            hidden = model.init_hidden(1)
            output, hidden = model(data, hidden)
            
            vocab_size = output.size(-1)
            output_flat = output.view(-1, vocab_size)
            log_probs = torch.nn.functional.log_softmax(output_flat, dim=-1)
            
            token_scores = []
            total_score = 0.0
            
            for i in range(len(indices) - 1):
                next_token_idx = indices[i + 1]
                token_log_prob = log_probs[i, next_token_idx].item()
                token_scores.append(token_log_prob)
                total_score += token_log_prob
            
            return total_score, token_scores
    
    except Exception as e:
        logger.warning(f"Error scoring sentence: {e}")
        return 0.0, []

def get_sentence_word_scores(model, dictionary, sentence, device='cuda'):
    """Get word-level scores for a sentence."""
    
    if not sentence.strip():
        return []
    
    if not sentence.strip().endswith('.'):
        sentence = sentence.strip() + ' .'
    else:
        sentence = sentence.strip()
    
    try:
        words = sentence.lower().split()
        indices = sentence_to_indices(sentence, dictionary)
        
        if not indices:
            return []
        
        data = torch.LongTensor(indices).unsqueeze(1).to(device)
        
        with torch.no_grad():
            hidden = model.init_hidden(1)
            output, hidden = model(data, hidden)
            
            vocab_size = output.size(-1)
            output_flat = output.view(-1, vocab_size)
            log_probs = torch.nn.functional.log_softmax(output_flat, dim=-1)
            
            word_scores = []
            for i in range(len(indices) - 1):
                next_token_idx = indices[i + 1]
                token_log_prob = log_probs[i, next_token_idx].item()
                word = words[i + 1] if i + 1 < len(words) else '<UNK>'
                word_scores.append((word, token_log_prob))
            
            return word_scores
    
    except Exception as e:
        logger.warning(f"Error getting word scores: {e}")
        return []

def compare_sentences(model, dictionary, sent1, sent2, device='cuda', verbose=True):
    """Compare two sentences and return which one the model prefers."""
    
    score1, tokens1 = score_sentence(model, dictionary, sent1, device)
    score2, tokens2 = score_sentence(model, dictionary, sent2, device)
    
    if verbose:
        print(f"\nSentence 1: {sent1}")
        print(f"Total Score: {score1:.4f}")
        print()
        print(f"Sentence 2: {sent2}")
        print(f"Total Score: {score2:.4f}")
        print()
        print(f"Preferred: {'Sentence 1' if score1 > score2 else 'Sentence 2'}")
    
    return score1 > score2, score1, score2

def batch_score_sentences(model, dictionary, sentences, device='cuda', batch_size=1):
    """Score multiple sentences."""
    results = []
    
    for sentence in sentences:
        score, token_scores = score_sentence(model, dictionary, sentence, device)
        results.append((score, token_scores))
    
    return results
