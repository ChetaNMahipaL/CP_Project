import argparse
import os
import subprocess
import pickle
import logging
import sys
import tempfile
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate n-gram model on grammatical constructions")
    
    parser.add_argument('--template_dir', type=str, default='../EMNLP2018/templates',
                        help='Location of the template files')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained SRILM n-gram model (.lm file)')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Vocabulary file used for training')
    parser.add_argument('--output_file', type=str, default='ngram_results.pickle',
                        help='File to store the results')
    parser.add_argument('--srilm_bin', type=str, default='./ngram',
                        help='Path to SRILM ngram binary')
    parser.add_argument('--tests', type=str, default='all',
                        help='Which constructions to test (agrmt/npi/all)')
    parser.add_argument('--order', type=int, default=5,
                        help='Order of n-gram model')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for scoring (larger = faster but more memory)')
    
    return parser.parse_args()

class NGramEvaluator:
    def __init__(self, model_path, vocab_path, srilm_bin='./ngram', order=5, batch_size=100):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.srilm_bin = srilm_bin
        self.order = order
        self.batch_size = batch_size
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        if not os.path.exists(srilm_bin):
            raise FileNotFoundError(f"SRILM ngram binary not found: {srilm_bin}")
        
        logger.info(f"Loaded n-gram model: {model_path}")
        logger.info(f"Model order: {order}")
        logger.info(f"Batch size for scoring: {batch_size}")
    
    def score_batch(self, sentences):
        """
        Score multiple sentences efficiently using single SRILM process.
        This is 10-100x faster than scoring each sentence individually.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of [(word_idx, log_prob), ...] for each sentence
        """
        if not sentences:
            return []
        
        # Ensure sentences are properly formatted
        formatted_sents = []
        for sent in sentences:
            if not sent.strip():
                formatted_sents.append("")
                continue
            
            sent = sent.strip()
            if not sent.endswith('.'):
                sent = sent + ' .'
            formatted_sents.append(sent)
        
        # Write to temporary file (SRILM works better with files)
        temp_fd, temp_file = tempfile.mkstemp(suffix='.txt', text=True)
        
        try:
            # Write sentences to temp file
            with os.fdopen(temp_fd, 'w') as f:
                for sent in formatted_sents:
                    f.write(sent + "\n")
            
            # Launch SRILM process with -ppl flag (which outputs word-level probabilities)
            cmd = [
                self.srilm_bin,
                '-lm', self.model_path,
                '-order', str(self.order),
                '-vocab', self.vocab_path,
                '-unk',
                '-ppl', temp_file,
                '-debug', '2'
            ]
            
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate(timeout=600)
            
            if proc.returncode != 0:
                logger.warning(f"SRILM error in batch processing")
                logger.warning(f"stderr: {stderr}")
                return [[] for _ in sentences]
            
            # Parse output: extract word-level log probabilities
            # Format is like: "p( word | context ) = [ngram_type] prob [ log_prob ]"
            # with TAB indent for probability lines
            all_word_scores = []
            current_sent_scores = []
            
            for line in stdout.split('\n'):
                stripped = line.strip()
                
                # Skip empty lines
                if not stripped:
                    continue
                
                # Check for end-of-file summary (contains "sentences," in plain text, not tabbed)
                if 'file' in stripped and 'sentences,' in stripped:
                    break
                
                # Probability lines start with tab/spaces and contain 'p( '
                if line.startswith(('\t', ' ')) and 'p( ' in line:
                    try:
                        # Extract log probability from "[ log_prob ]"
                        bracket_start = line.rfind('[')
                        bracket_end = line.rfind(']')
                        if bracket_start >= 0 and bracket_end > bracket_start:
                            log_prob_str = line[bracket_start+1:bracket_end].strip()
                            log_prob = float(log_prob_str)
                            current_sent_scores.append(log_prob)
                    except (ValueError, IndexError):
                        continue
                elif not line.startswith((' ', '\t')) and stripped:
                    # Non-indented line that's not empty = new sentence or summary
                    if 'sentences,' in stripped or 'OOVs' in stripped or 'zeroprobs' in stripped:
                        # Summary line - save current sentence scores
                        if current_sent_scores:
                            all_word_scores.append(current_sent_scores)
                            current_sent_scores = []
                    elif stripped and not any(c.isdigit() for c in stripped[0]):
                        # Looks like a sentence (starts with word, not digit)
                        if current_sent_scores:
                            all_word_scores.append(current_sent_scores)
                            current_sent_scores = []
            
            # Don't forget last sentence
            if current_sent_scores:
                all_word_scores.append(current_sent_scores)
            
            # Pad with empty lists if needed
            while len(all_word_scores) < len(sentences):
                all_word_scores.append([])
            
            # Convert to required format: [(word_idx, log_prob), ...]
            results = []
            for word_scores in all_word_scores[:len(sentences)]:
                results.append([(i, score) for i, score in enumerate(word_scores)])
            
            return results
        
        except subprocess.TimeoutExpired:
            logger.error("SRILM batch processing timeout")
            return [[] for _ in sentences]
        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            return [[] for _ in sentences]
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

def load_test_sentences(template_dir, tests='all'):
    """Load test sentences from pickle files."""
    logger.info("Loading test sentences...")
    
    sys.path.insert(0, '../src')
    from template.TestCases import TestCase
    
    testcase = TestCase()
    if tests == 'agrmt':
        test_list = testcase.agrmt_cases
    elif tests == 'npi':
        test_list = testcase.npi_cases
    else:
        test_list = testcase.all_cases
    
    all_test_sents = {}
    for test_name in test_list:
        try:
            test_file = os.path.join(template_dir, f"{test_name}.pickle")
            with open(test_file, 'rb') as f:
                test_sents = pickle.load(f)
                all_test_sents[test_name] = test_sents
                total_sents = sum(len(v) for v in test_sents.values())
                logger.info(f"Loaded {test_name}: {total_sents} sentences")
        except FileNotFoundError:
            logger.warning(f"Test file not found: {test_file}")
    
    return all_test_sents

def evaluate_ngram(evaluator, all_test_sents):
    """Evaluate n-gram model on all test sentences using batch scoring."""
    logger.info("Starting evaluation (batch mode)...")
    
    results = {}
    total_sentences = sum(sum(len(v) for v in test_sents.values()) 
                         for test_sents in all_test_sents.values())
    processed = 0
    
    for test_name, test_sents in all_test_sents.items():
        logger.info(f"Evaluating {test_name}...")
        results[test_name] = {}
        
        for key, sentence_pairs in test_sents.items():
            results[test_name][key] = []
            
            # Collect all sentences in this group
            all_sents = []
            for pair in sentence_pairs:
                if isinstance(pair, (list, tuple)):
                    all_sents.extend(pair)
                else:
                    all_sents.append(pair)
            
            # Score in batches
            batch_results = []
            for batch_start in tqdm(range(0, len(all_sents), evaluator.batch_size),
                                   desc=f"{test_name}/{key}"):
                batch_end = min(batch_start + evaluator.batch_size, len(all_sents))
                batch = all_sents[batch_start:batch_end]
                
                # Score entire batch at once
                batch_scores = evaluator.score_batch(batch)
                batch_results.extend(batch_scores)
            
            results[test_name][key] = batch_results
            processed += len(all_sents)
    
    return results

def main():
    args = parse_args()
    
    # Create evaluator with batch size
    evaluator = NGramEvaluator(
        model_path=args.model,
        vocab_path=args.vocab,
        srilm_bin=args.srilm_bin,
        order=args.order,
        batch_size=args.batch_size
    )
    
    # Load test sentences
    all_test_sents = load_test_sentences(args.template_dir, args.tests)
    
    # Evaluate
    results = evaluate_ngram(evaluator, all_test_sents)
    
    # Save results
    output_path = os.path.join(args.template_dir, args.output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
