import argparse
import pickle
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
import sys
import dill

# Add word-language-model to path BEFORE any imports that might unpickle
_script_dir = os.path.dirname(os.path.abspath(__file__))
_lm_dir = os.path.join(_script_dir, '../word-language-model')
if _lm_dir not in sys.path:
    sys.path.insert(0, _lm_dir)

# Import Dictionary class so it's available when unpickling
from data import Dictionary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Multitask LSTM on grammatical constructions")
    
    parser.add_argument('--template_dir', type=str, default='../EMNLP2018/templates',
                        help='Location of the template files')
    parser.add_argument('--output_file', type=str, default='lstm_multitask_results.pickle',
                        help='File to store the results')
    parser.add_argument('--model', type=str, default='../models/lstm.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--lm_data', type=str, default='../models/lm_data.bin',
                        help='The model .bin file for faster loading (contains dictionary)')
    parser.add_argument('--tests', type=str, default='all',
                        help='Which constructions to test (agrmt/npi/all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation (LSTM scores sentences individually)')
    
    return parser.parse_args()

class MultitaskLSTMEvaluator:
    def __init__(self, model_path, lm_data_path, device='cuda'):
        self.device = device
        logger.info(f"Loading multitask LSTM model: {model_path}")
        
        # Load the checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(lm_data_path):
            raise FileNotFoundError(f"LM data file not found: {lm_data_path}")
        
        # Load saved data (contains Dictionary object, saved with dill)
        try:
            self.dictionary = torch.load(lm_data_path, map_location='cpu', pickle_module=dill)
            logger.info(f"Loaded Dictionary from {lm_data_path}")
            logger.info(f"Vocabulary size: {len(self.dictionary)}")
        except Exception as e:
            logger.error(f"Failed to load LM data: {e}")
            raise
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            logger.info(f"Loaded checkpoint from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Import model class
        from model import RNNModel
        
        # Reconstruct model from checkpoint
        if isinstance(checkpoint, dict):
            # Checkpoint is a state dict
            model_args = checkpoint.get('args', None)
            if model_args is None:
                # Try to infer from checkpoint keys
                logger.warning("Model args not found in checkpoint, using defaults")
                ntokens = len(self.dictionary)
                model = RNNModel('LSTM', ntokens, 400, 650, 2, dropout=0.2, tie_weights=False)
            else:
                ntokens = len(self.dictionary)
                model = RNNModel(model_args.model, ntokens, model_args.emsize, 
                                model_args.nhid, model_args.nlayers, 
                                dropout=model_args.dropout, tie_weights=model_args.tied)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        else:
            # Checkpoint is the model itself
            model = checkpoint
        
        self.model = model
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded on device: {device}")
    
    def sentence_to_indices(self, sentence):
        """Convert sentence string to token indices."""
        words = sentence.strip().lower().split()
        indices = []
        for word in words:
            if word in self.dictionary.word2idx:
                indices.append(self.dictionary.word2idx[word])
            else:
                # Use <unk> token if word not in vocabulary
                unk_idx = self.dictionary.word2idx.get('<unk>', 0)
                indices.append(unk_idx)
        return indices
    
    def get_sentence_score(self, sentence):
        """
        Score a sentence using the LSTM.
        Returns log probability and per-token scores.
        """
        sentence = str(sentence).strip()
        
        if not sentence:
            return 0.0, []
        
        if not sentence.endswith('.'):
            sentence = sentence + ' .'
        
        try:
            indices = self.sentence_to_indices(sentence)
            if not indices:
                return 0.0, []
            
            # Convert to tensor
            data = torch.LongTensor(indices).unsqueeze(1).to(self.device)
            
            with torch.no_grad():
                hidden = self.model.init_hidden(1)
                output, hidden = self.model(data, hidden)
                
                # Get log probabilities
                vocab_size = output.size(-1)
                output_flat = output.view(-1, vocab_size)
                
                # Compute log softmax
                log_probs = torch.nn.functional.log_softmax(output_flat, dim=-1)
                
                # Get scores for predicted tokens
                token_scores = []
                total_score = 0.0
                
                for i in range(len(indices) - 1):
                    next_token_idx = indices[i + 1]
                    token_log_prob = log_probs[i, next_token_idx].item()
                    token_scores.append(token_log_prob)
                    total_score += token_log_prob
                
                # Also add score for predicting end token or last token
                if len(indices) > 0:
                    last_output_idx = len(indices) - 1
                    if last_output_idx < log_probs.size(0):
                        # Add negative of end-of-sequence penalty (use average of scores as approximation)
                        pass
                
                return total_score, token_scores
        
        except Exception as e:
            logger.warning(f"Error scoring sentence '{sentence}': {e}")
            return 0.0, []
    
    def get_sentence_word_scores(self, sentence):
        """
        Score a sentence and return word-level scores.
        Returns list of (word, log_prob) tuples.
        """
        sentence = str(sentence).strip()
        
        if not sentence:
            return []
        
        if not sentence.endswith('.'):
            sentence = sentence + ' .'
        
        try:
            words = sentence.lower().split()
            indices = self.sentence_to_indices(sentence)
            
            if not indices:
                return []
            
            data = torch.LongTensor(indices).unsqueeze(1).to(self.device)
            
            with torch.no_grad():
                hidden = self.model.init_hidden(1)
                output, hidden = self.model(data, hidden)
                
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
            logger.warning(f"Error getting word scores for '{sentence}': {e}")
            return []


def load_test_sentences(template_dir, tests='all'):
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
                logger.info(f"Loaded {test_name}: {sum(len(v) for v in test_sents.values())} sentences")
        except FileNotFoundError:
            logger.warning(f"Test file not found: {test_file}")
    
    return all_test_sents


def evaluate_lstm_multitask(evaluator, all_test_sents):
    logger.info("Starting evaluation...")
    
    results = {}
    
    for test_name, test_sents in all_test_sents.items():
        logger.info(f"Evaluating {test_name}...")
        results[test_name] = {}
        
        for key, sentence_tuples in test_sents.items():
            results[test_name][key] = []
            
            for i, sent_tuple in enumerate(tqdm(sentence_tuples, desc=f"{test_name}/{key}")):
                try:
                    # Handle both 2-tuple (agreement) and 3-tuple (NPI) formats
                    if len(sent_tuple) == 2:
                        gram_sent, ungram_sent = sent_tuple
                        scores = []
                        scores.append(evaluator.get_sentence_word_scores(gram_sent))
                        scores.append(evaluator.get_sentence_word_scores(ungram_sent))
                    elif len(sent_tuple) == 3:
                        gram_sent, intr_sent, ungram_sent = sent_tuple
                        scores = []
                        scores.append(evaluator.get_sentence_word_scores(gram_sent))
                        scores.append(evaluator.get_sentence_word_scores(intr_sent))
                        scores.append(evaluator.get_sentence_word_scores(ungram_sent))
                    else:
                        logger.error(f"Unexpected tuple length: {len(sent_tuple)}")
                        raise ValueError(f"Expected 2 or 3 sentences per tuple, got {len(sent_tuple)}")
                    
                    results[test_name][key].extend(scores)
                
                except Exception as e:
                    logger.error(f"Error processing tuple {i} in {test_name}/{key}: {e}")
                    logger.error(f"Tuple type: {type(sent_tuple)}, Length: {len(sent_tuple) if isinstance(sent_tuple, tuple) else 'N/A'}")
                    raise
    
    return results


def main():
    args = parse_args()
    
    evaluator = MultitaskLSTMEvaluator(model_path=args.model, lm_data_path=args.lm_data, 
                                      device=args.device)
    
    all_test_sents = load_test_sentences(args.template_dir, args.tests)
    
    results = evaluate_lstm_multitask(evaluator, all_test_sents)
    
    output_path = os.path.join(args.template_dir, args.output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
