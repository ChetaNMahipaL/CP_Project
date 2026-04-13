import os
import sys
import torch
import dill
import gzip
from tqdm import tqdm

class convertvocab(object):
    def __init__(self, load_from, save_to):
        self.dictionary = Dictionary()
        self.loadme = self.load_dict(load_from)
        self.save_to = self.save_dict(save_to)

    def save_dict(self, path):
        with open(path, 'wb') as f:
            torch.save(self.dictionary, f, pickle_module=dill)

    def load_dict(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            for line in f:
                self.dictionary.add_word(line.strip())

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = []
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag) - 1
        return self.tag2idx[tag]

    def __len__(self):
        return len(self.idx2word)

class SentenceCorpus(object):
    def __init__(self, lm_path, ccg_path=None, save_to='lm_data.bin', testflag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt'):
        if not testflag:
            # Check if data paths exist
            if not os.path.exists(lm_path):
                sys.stderr.write(f'ERROR: Language model data path not found: {lm_path}\n')
                raise FileNotFoundError(f'Language model data path not found: {lm_path}')
            sys.stderr.write(f'Loading language model data from {lm_path}\n')
            
            self.dictionary = Dictionary()
            self.train_lm = self.tokenize(os.path.join(lm_path, trainfname))
            self.valid_lm = self.tokenize_with_unks(os.path.join(lm_path, validfname))
            if ccg_path:
                if not os.path.exists(ccg_path):
                    sys.stderr.write(f'ERROR: CCG data path not found: {ccg_path}\n')
                    raise FileNotFoundError(f'CCG data path not found: {ccg_path}')
                sys.stderr.write(f'Loading CCG data from {ccg_path}\n')
                self.train_ccg = self.tokenize_ccg(os.path.join(ccg_path, trainfname))
                self.valid_ccg = self.tokenize_ccg(os.path.join(ccg_path, validfname))
            else:
                self.train_ccg = self.valid_ccg = torch.LongTensor(0)
            self.save_to = self.save_dict(save_to)
        else:
            self.dictionary = self.load_dict(save_to)
            if not os.path.exists(lm_path):
                sys.stderr.write(f'ERROR: Language model data path not found: {lm_path}\n')
                raise FileNotFoundError(f'Language model data path not found: {lm_path}')
            sys.stderr.write(f'Loading test data from {lm_path}\n')
            self.test_lm = self.sent_tokenize_with_unks(os.path.join(lm_path,testfname))
            if ccg_path:
                if not os.path.exists(ccg_path):
                    sys.stderr.write(f'ERROR: CCG data path not found: {ccg_path}\n')
                    raise FileNotFoundError(f'CCG data path not found: {ccg_path}')
                self.test_ccg = self.sent_tokenize_ccg_with_unks(os.path.join(ccg_path, testfname))
            else:
                self.test_ccg = torch.LongTensor(0)

    def save_dict(self, path):
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sys.stderr.write(f'Saving dictionary to {path}\n')
        with open(path, 'wb') as f:
            torch.save(self.dictionary, f, pickle_module=dill)
        sys.stderr.write(f'Dictionary saved successfully\n')

    def load_dict(self, path):
        #assert os.path.exists(path)
        if not os.path.exists(path):
            sys.stderr.write(f'ERROR: Dictionary file not found at {path}\n')
            sys.stderr.write(f'Please run training first (without --test flag) to generate the dictionary\n')
            sys.stderr.write(f'Example: python main.py --lm_data data/lm_data --save_lm_data {path}\n')
            raise FileNotFoundError(f'Dictionary file not found: {path}. Run training first to create it.')
        sys.stderr.write(f'Loading dictionary from {path}\n')
        with open(path, 'rb') as f:
            fdata = torch.load(f, pickle_module=dill)
            if type(fdata) == type(()):
                # compatibility with old pytorch LM saving
                sys.stderr.write(f'Dictionary loaded successfully\n')
                return(fdata[3])
            sys.stderr.write(f'Dictionary loaded successfully\n')
            return(fdata)

    def tokenize_ccg(self, path):
        """Tokenizes and gets CCG tags for a text file."""
        assert os.path.exists(path)

        # Add words and tags to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            FIRST = True
            for line in tqdm(f, desc='Counting tokens (CCG train)', unit=' lines'):
                if line.strip() != "":
                    (word, tag) = line.strip().split("\t")
                    if FIRST:
                        self.dictionary.add_word("<eos>")
                        self.dictionary.add_tag("<EOS>")
                        tokens += 1
                        FIRST = False
                    self.dictionary.add_word(word)
                    self.dictionary.add_tag(tag)
                    tokens += 1
                    if word == '.':
                        FIRST = True
                        self.dictionary.add_word("<eos>")
                        self.dictionary.add_tag("<EOS>")
                        tokens += 1
        # Tokenize file content
        with open(path, 'r') as f:
            word_ids = torch.LongTensor(tokens)
            tag_ids = torch.LongTensor(tokens)
            token = 0
            FIRST = True
            for line in tqdm(f, desc='Tokenizing (CCG train)', total=tokens, unit=' tokens'):
                if line.strip() != "":
                    if FIRST:
                        word_ids[token] = self.dictionary.word2idx["<eos>"]
                        tag_ids[token] = self.dictionary.tag2idx["<EOS>"]
                        token += 1
                        FIRST = False
                    (word, tag) = line.strip().split("\t")
                    word_ids[token] = self.dictionary.word2idx[word]
                    tag_ids[token] = self.dictionary.tag2idx[tag]
                    token += 1
                    if word == ".":
                        word_ids[token] = self.dictionary.word2idx["<eos>"]
                        tag_ids[token] = self.dictionary.tag2idx["<EOS>"]
                        token += 1
                        FIRST = True
        return word_ids, tag_ids

    def tokenize_ccg_with_unks(self, path):
        """Tokenizes and gets CCG tags for a text file."""
        assert os.path.exists(path)

        # Add words and tags to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            FIRST = True
            for line in tqdm(f, desc='Counting tokens (CCG valid)', unit=' lines'):
                if line.strip() != "":
                    (word, tag) = line.strip().split("\t")
                    if FIRST:
                        tokens += 1
                        FIRST = False
                    tokens += 1
                    if word == '.':
                        FIRST = True
                        tokens += 1
        # Tokenize file content
        with open(path, 'r') as f:
            word_ids = torch.LongTensor(tokens)
            tag_ids = torch.LongTensor(tokens)
            token = 0
            FIRST = True
            for line in tqdm(f, desc='Tokenizing (CCG valid)', unit=' lines'):
                if line.strip() != "":
                    if FIRST:
                        word_ids[token] = self.dictionary.word2idx["<eos>"]
                        tag_ids[token] = self.dictionary.tag2idx["<EOS>"]
                        token += 1
                        FIRST = False
                    (word, tag) = line.strip().split("\t")
                    if word not in self.dictionary.word2idx:
                        word_ids[token] = self.dictionary.add_word("<unk>")
                    else:
                        word_ids[token] = self.dictionary.word2idx[word]
                    if tag not in self.dictionary.tag2idx:
                        tag_ids[token] = self.dictionary.add_tag("<UNK>")
                    else:
                        tag_ids[token] = self.dictionary.tag2idx[tag]
                    
                    token += 1
                    if word == ".":
                        word_ids[token] = self.dictionary.word2idx["<eos>"]
                        tag_ids[token] = self.dictionary.tag2idx["<EOS>"]
                        token += 1
                        FIRST = True
        return word_ids, tag_ids

    def sent_tokenize_ccg_with_unks(self, path):
        """Tokenizes and gets CCG tags for a text file."""
        assert os.path.exists(path)
        sents = []
        # Add words and tags to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            FIRST = True
            for line in tqdm(f, desc='Counting tokens (CCG test)', unit=' lines'):
                if line.strip() != "":
                    (word, tag) = line.strip().split("\t")
                    if FIRST:
                        tokens += 1
                        FIRST = False
                    tokens += 1
                    if word == '.':
                        FIRST = True
                        tokens += 1
        # Tokenize file content
        with open(path, 'r') as f:
            word_ids = torch.LongTensor(tokens)
            tag_ids = torch.LongTensor(tokens)
            token = 0
            sent = []
            FIRST = True
            for line in tqdm(f, desc='Tokenizing (CCG test)', unit=' lines'):
                if line.strip() != "":
                    if FIRST:
                        word_ids[token] = self.dictionary.word2idx["<eos>"]
                        tag_ids[token] = self.dictionary.tag2idx["<EOS>"]
                        token += 1
                        FIRST = False
                        sent.append(("<eos>", "<EOS>"))
                    (word, tag) = line.strip().split("\t")
                    sent.append((word, tag))
                    if word not in self.dictionary.word2idx:
                        word_ids[token] = self.dictionary.add_word("<unk>")
                    else:
                        word_ids[token] = self.dictionary.word2idx[word]
                    if tag not in self.dictionary.tag2idx:
                        tag_ids[token] = self.dictionary.add_tag("<UNK>")
                    else:
                        tag_ids[token] = self.dictionary.tag2idx[tag]
                    
                    token += 1
                    if word == ".":
                        word_ids[token] = self.dictionary.word2idx["<eos>"]
                        tag_ids[token] = self.dictionary.tag2idx["<EOS>"]
                        token += 1
                        FIRST = True
                        sent.append(("<eos>", "<EOS>"))
                        sents.append(sent)
                        sent = []
        return sents, word_ids, tag_ids
                    
        

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb', encoding="utf-8") as f:
                tokens = 0
                for line in tqdm(f.readlines(), desc='Counting tokens (LM train)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with gzip.open(path, 'rb', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in tqdm(f.readlines(), desc='Tokenizing (LM train)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
        else:
            with open(path, 'r+', encoding="utf-8") as f:
                tokens = 0
                for line in tqdm(f, desc='Counting tokens (LM train)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r+', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in tqdm(f, desc='Tokenizing (LM train)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
        return ids

    def tokenize_with_unks(self, path):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        if path[-2:] == 'gz':
            # Add words to the dictionary
            with gzip.open(path, 'rb', encoding="utf-8") as f:
                tokens = 0
                for line in tqdm(f, desc='Counting tokens (LM valid)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    tokens += len(words)

            # Tokenize file content
            with gzip.open(path, 'rb', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in tqdm(f, desc='Tokenizing (LM valid)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    for word in words:
                        if word not in self.dictionary.word2idx:
                            ids[token] = self.dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.dictionary.word2idx[word]
                        token += 1
        else:
            # Add words to the dictionary
            with open(path, 'r+', encoding="utf-8") as f:
                tokens = 0
                for line in tqdm(f, desc='Counting tokens (LM valid)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    tokens += len(words)

            # Tokenize file content
            with open(path, 'r+', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in tqdm(f, desc='Tokenizing (LM valid)', unit=' lines'):
                    words = ['<eos>'] + line.split()
                    for word in words:
                        if word not in self.dictionary.word2idx:
                            ids[token] = self.dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.dictionary.word2idx[word]
                        token += 1
        return ids

    def sent_tokenize_with_unks(self, path):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        if path [-2:] == 'gz':
            with gzip.open(path, 'rb', encoding="utf-8") as f:
                for line in tqdm(f, desc='Tokenizing sentences (LM test)', unit=' sentences'):
                    sents.append(line.strip())
                    words = ['<eos>'] + line.split()
                    tokens = len(words)

                    # tokenize file content
                    ids = torch.LongTensor(tokens)
                    token = 0
                    for word in words:
                        if word not in self.dictionary.word2idx:
                            ids[token] = self.dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    all_ids.append(ids)
        else:
            with open(path, 'r+', encoding="utf-8") as f:
                for line in tqdm(f, desc='Tokenizing sentences (LM test)', unit=' sentences'):
                    sents.append(line.strip())
                    words = ['<eos>'] + line.split()
                    tokens = len(words)
                    
                    # tokenize file content
                    ids = torch.LongTensor(tokens)
                    token = 0
                    for word in words:
                        if word not in self.dictionary.word2idx:
                            ids[token] = self.dictionary.add_word("<unk>")
                        else:
                            ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    all_ids.append(ids)                
        return (sents, all_ids)
