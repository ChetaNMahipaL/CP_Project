import argparse
import pickle
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test GPT-2 on grammatical constructions")
    
    parser.add_argument('--template_dir', type=str, default='../EMNLP2018/templates',
                        help='Location of the template files')
    parser.add_argument('--output_file', type=str, default='gpt2_results.pickle',
                        help='File to store the results')
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='GPT-2 model variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
    parser.add_argument('--tests', type=str, default='all',
                        help='Which constructions to test (agrmt/npi/all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    
    return parser.parse_args()

class GPT2Evaluator:
    def __init__(self, model_name='gpt2', device='cuda'):
        self.device = device
        logger.info(f"Loading GPT-2 model: {model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded on device: {device}")
    
    def get_sentence_score(self, sentence):
        
        sentence = str(sentence).strip()
        
        if not sentence:
            return 0.0, []
        
        if not sentence.endswith('.'):
            sentence = sentence + ' .'
        
        tokens = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            logits = outputs.logits
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        token_log_probs = []
        for i in range(len(tokens[0]) - 1):
            token_id = tokens[0, i + 1]
            token_log_prob = log_probs[0, i, token_id].item()
            token_log_probs.append(token_log_prob)
        
        total_score = sum(token_log_probs)
        
        return total_score, token_log_probs
    
    def get_sentence_word_scores(self, sentence):
        
        sentence = str(sentence).strip()
        
        if not sentence:
            return []
        
        if not sentence.endswith('.'):
            sentence = sentence + ' .'
        
        tokens = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
        token_ids = tokens[0].tolist()
        
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            logits = outputs.logits
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        word_scores = []
        words = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        for i in range(len(token_ids) - 1):
            token_id = token_ids[i + 1]
            token_log_prob = log_probs[0, i, token_id].item()
            word = words[i + 1]
            word_scores.append((word, token_log_prob))
        
        return word_scores

def load_test_sentences(template_dir, tests='all'):
    logger.info("Loading test sentences...")
    
    import sys
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

def evaluate_gpt2(evaluator, all_test_sents):
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
                    # Agreement: (grammatical, ungrammatical)
                    # NPI: (grammatical, intrusive, ungrammatical)
                    
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
    
    evaluator = GPT2Evaluator(model_name=args.model_name, device=args.device)
    
    all_test_sents = load_test_sentences(args.template_dir, args.tests)
    
    results = evaluate_gpt2(evaluator, all_test_sents)
    
    output_path = os.path.join(args.template_dir, args.output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
