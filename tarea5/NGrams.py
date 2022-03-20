# os
import random

# NLP and numpy
import nltk 
import numpy as np
import nltk
from nltk.probability import FreqDist
from nltk import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd

# pytorch
from torch import nn
from torch.nn import Module

punctuation = ['.', '...', ',', '!', '¡', '¿', '?', ':', ';', '"', '|', '[', ']', '°', '(', ')', '*', '+', '/', '-', '^', '<', '>', '\'', '&', '@usuario', '<url>']

class NGramBuilder:
    def __init__(self, tokenizer=None, embeddings=None, d_model=256, sos='<s>', eos='</s>', unk='<unk>', punctuation=punctuation, postprocess=None):
        self.tokenizer = self.default_tokenizer() if tokenizer == None else tokenizer
        self.embeddings = embeddings
        self.d_model = d_model if embeddings is None else embeddings.d_model
        # special symbols
        self.SOS = sos
        self.EOS = eos
        self.UNK = unk
        # vocabulary 2 id and viceversa
        self.word2id  = None
        self.id2word  = None
        self.voc_size = 0
        # post tokenization functions
        self.punctuation = set(punctuation) if punctuation != None else None
        self.postprocess = postprocess if postprocess is not None else lambda x : x
        
    def default_tokenizer(doc):
        return TweetTokenizer().tokenize
    
    def get_vocabulary(self):
        return set(self.word2id.keys())
    
    def remove_punct(self, tokenized_documents):
        if self.punctuation == None:
            return tokenized_documents
        else:
            return [[token for token in doc if token not in self.punctuation] for doc in tokenized_documents]
        
    def get_ids(self, words:list):
        # transform list of words to list of ids
        unk_id = self.word2id.get(self.UNK, 0)
        ids = [self.word2id.get(word, unk_id) for word in words]
        return ids
    
    def __transform(self, tokenized_docs, start_padding:bool, end_padding:bool):
        N = self.N
        # docs and labels lists
        ngram_docs, ngram_targs = [], []
        # traverse each doc
        for doc in tokenized_docs:
            # add padding
            doc = ([self.SOS]*(N - 1) if start_padding else []) + \
                    doc + ([self.EOS] if end_padding else [])
            # get ids    
            ids = self.get_ids(doc)
            # traverse each word as center and build ngrams
            for i in range(N-1, len(doc)):    
                ngram_docs.append(ids[i-(N-1): i])
                ngram_targs.append(ids[i])
                
        return np.array(ngram_docs), np.array(ngram_targs)
    
    def _tokenize(self, documents):
        tokenized_docs = [self.tokenizer(doc.lower()) for doc in documents]
        tokenized_docs = self.remove_punct(tokenized_docs)
        tokenized_docs = self.postprocess(tokenized_docs)
        return tokenized_docs
    
    def build_emb_matrix(self):
        dim_v = len(self.word2id)
        if self.embeddings is None:
            self.emb_matrix = np.random.rand(dim_v, self.d_model)
        else:
            self.emb_matrix = np.random.rand(dim_v, self.d_model)
            for word in self.word2id.keys():
                if self.embeddings[word] is not None:
                    pos = self.word2id[word]
                    self.emb_matrix[pos] = self.embeddings[word]
                
    def fit(self, documents, N, t=10000):
        self.N = N
        # tokenize documents
        tokenized_docs = self._tokenize(documents)
        
        # get vocabulary and word2id and ids2word dicts
        vocabulary = get_vocabulary(tokenized_docs, t-3)
        self.word2id, self.id2word = word2ids(vocabulary)
        self.voc_size = len(self.word2id)
        self.build_emb_matrix()
        
        return self.__transform(tokenized_docs, start_padding=True, end_padding=True)
    
    def transform(self, documents: list[list or str], start_padding=True, end_padding=True):
        # list of documents as strings
        if type(documents[0]) is str:
            # tokenize documents
            tokenized_docs = self._tokenize(documents)
            return self.__transform(tokenized_docs, start_padding, end_padding)
        
        # list of documents as list of tokens
        elif type(documents[0]) is list:
            return self.__transform(documents, start_padding, end_padding)
        
        print('[ERR]: documents should be list of strings or list of lists of tokens')
        return None
    
    def inverse(self, docs_as_ids):
        # empty list
        if len(docs_as_ids) == 0:
            return None
        
        # multiple docs
        if type(docs_as_ids[0]) in (list, np.ndarray):
            return [[self.id2word.get(tok_id) for tok_id in doc] 
                    for doc in docs_as_ids ]
        # single doc
        return [self.id2word.get(tok_id) for tok_id in docs_as_ids]
    
    

def sample(probs):
    acc = np.cumsum(probs)       # build cumulative probability
    val = np.random.uniform()    # get random number between [0, 1]
    pos = np.argmax((val < acc)) # get the index of the word to sample
    return pos

    
class NGramNeuralModel:
    def __init__(self, NGram: NGramBuilder, neuralModel:nn.Module):
        self.model = neuralModel
        self.NGram = NGram
        self.model.eval()
    
    def predict(self, context:list, use_gpu=False):
        context = self.NGram.get_ids(context)
        context = torch.tensor([context])
        if use_gpu:
            context = context.cuda()
            
        logits = self.model(context)
        cond_probs = get_probs(logits)
        index = sample(cond_probs)
        return self.NGram.inverse([index])[0]
    
    def estimate_prob(self, sequence:str, use_gpu=False, ret_probs=False, start_padding=False, end_padding=False):
        # feed model and get probs
        ngrams, targets = self.NGram.transform([sequence], start_padding, end_padding)
        ngrams = torch.tensor(ngrams)
        if use_gpu:
            ngrams = ngrams.cuda()
            
        logits = self.model(ngrams)
        probs  = get_probs(logits)
        
        # get prob for each context and target
        num_target = [i for i in range(len(targets))]
        cond_probs = probs[num_target, targets]
        log_prob = np.sum(np.log(cond_probs))
        return np.exp(log_prob) if ret_probs else log_prob
        
    def generate_sequence(self, use_gpu=False, max_length=100):
        sequence = ['<s>']*(self.NGram.N - 1)
        context = [token for token in sequence]
        while sequence[-1] != '</s>' and len(sequence) < max_length:
            word = self.predict(context, use_gpu)
            context.pop(0)
            context.append(word)
            sequence.append(word)
            
        return sequence
    
    def perplexity(self, test_set, use_gpu=False):
        ngrams, targets = self.NGram.transform(test_set)
        ngrams = torch.tensor(ngrams)
        if use_gpu:
            ngrams = ngrams.cuda()
        logits = self.model(ngrams)
        probs = get_probs(logits)
        
        # get cond probs and perplexity
        num_target = [i for i in range(len(targets))]
        cond_probs = probs[num_target, targets]
        log_perp = np.sum(-np.log(cond_probs))     # log(1/cond_probs) = log(1) - log(cond_probs) = -log(cond_probs)
        perp = np.exp(1/len(targets) * log_perp)   # 1/N = 1/len(targets)
        return perp