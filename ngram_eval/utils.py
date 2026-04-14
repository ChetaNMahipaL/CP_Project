"""
Utility functions for n-gram evaluation using SRILM
"""
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

def run_ngram_command(ngram_bin, model_file, vocab_file, input_file, order=5, debug=2):
    """
    Run SRILM ngram command and return output.
    """
    cmd = [
        ngram_bin,
        '-lm', model_file,
        '-order', str(order),
        '-vocab', vocab_file,
        '-unk',
        '-debug', str(debug)
    ]
    
    try:
        with open(input_file, 'r') as f:
            result = subprocess.run(cmd, stdin=f, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"SRILM command failed: {result.stderr}")
            return None
        
        return result.stdout
    except Exception as e:
        logger.error(f"Error running ngram command: {e}")
        return None

def parse_srilm_output(output):
    """
    Parse SRILM output to extract log probabilities.
    Returns list of (word, log_prob) tuples.
    """
    word_scores = []
    
    for line in output.split('\n'):
        if 'p( ' in line:
            try:
                # Format: "p( word | context ) [ log_prob ]"
                parts = line.split('[')
                if len(parts) >= 2:
                    log_prob = float(parts[-1].strip().rstrip(']'))
                    # Extract word
                    word_part = line.split('p( ')[1].split(' |')[0]
                    word_scores.append((word_part, log_prob))
            except (ValueError, IndexError):
                continue
    
    return word_scores

def get_sentence_score(ngram_bin, model_file, vocab_file, sentence, order=5):
    """
    Score a single sentence.
    Returns: (total_log_prob, word_scores, count)
    """
    if not sentence.strip():
        return 0.0, [], 0
    
    # Format for SRILM
    sentence = sentence.strip()
    if not sentence.endswith('.'):
        sentence = sentence + ' .'
    
    sentence = f"<s> {sentence} </s>"
    
    # Use temporary file for SRILM input
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(sentence)
        temp_file = f.name
    
    try:
        output = run_ngram_command(ngram_bin, model_file, vocab_file, temp_file, order)
        
        if output is None:
            return 0.0, [], 0
        
        word_scores = parse_srilm_output(output)
        total_score = sum(score for _, score in word_scores)
        
        return total_score, word_scores, len(word_scores)
    
    finally:
        os.unlink(temp_file)

def compare_two_sentences(ngram_bin, model_file, vocab_file, sent1, sent2, order=5, verbose=True):
    """
    Compare two sentences and return which one has higher probability.
    """
    score1, words1, count1 = get_sentence_score(ngram_bin, model_file, vocab_file, sent1, order)
    score2, words2, count2 = get_sentence_score(ngram_bin, model_file, vocab_file, sent2, order)
    
    if verbose:
        print(f"\nSentence 1: {sent1}")
        print(f"Total Score: {score1:.4f} ({count1} words)")
        
        print(f"\nSentence 2: {sent2}")
        print(f"Total Score: {score2:.4f} ({count2} words)")
        
        print(f"\nPreferred: {'Sentence 1' if score1 > score2 else 'Sentence 2'}")
        print(f"Difference: {abs(score1 - score2):.4f} log-prob units")
    
    return score1 > score2, score1, score2

def batch_score_sentences(ngram_bin, model_file, vocab_file, sentences, order=5):
    """
    Score multiple sentences efficiently.
    Returns list of (total_score, word_count) tuples.
    """
    results = []
    
    for sent in sentences:
        score, words, count = get_sentence_score(ngram_bin, model_file, vocab_file, sent, order)
        results.append((score, count))
    
    return results
