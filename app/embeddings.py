import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os
import sys
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, trainers, models, pre_tokenizers

# To have a progress bar when running .apply
tqdm.pandas()


# Training
DATA_DIR = '../data'
ARTICLES_FILE = f'{DATA_DIR}/articles.csv'
ARTICLES_TEXT_FILE = f'{DATA_DIR}/articles.txt'
EMBEDDINGS_BERT_FILE = f'{DATA_DIR}/embeddings_bert.npy'
EMBEDDINGS_SPARSE_FILE = f'{DATA_DIR}/embeddings_sparse.npz'

# Inference
BERT_TOKENIZER = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
BERT_MODEL = AutoModel.from_pretrained("dbmdz/bert-base-german-uncased")
SPARSE_TOKENIZER_FILE = "tokenizer.json"
try:
    SPARSE_TOKENIZER = PreTrainedTokenizerFast(tokenizer_file=SPARSE_TOKENIZER_FILE)
except:
    print('tokenizer not yet computed')



def get_texts() -> pd.Series:
    '''
    Combine title and subtitles as text and convert to lower case
    '''
    df = pd.read_csv(ARTICLES_FILE)
    texts = (df['title'].fillna('').astype(str) + '. '+ df['subtitle'].fillna('').astype(str)).str.lower()
    return texts


def compute_bert_embeddings():
    '''
    Compute bert embeddings with 768 dims
    '''
    texts = get_texts()
    vecs = texts.progress_apply(lambda x: get_vec(x))
    vecs = np.stack(vecs.values)
    np.save(EMBEDDINGS_BERT_FILE, vecs)


def get_bert_vec(text: str) -> np.ndarray:
    '''
    Generate BERT embedding with 768 dims for given text
    '''
    # Get hidding state as text embedding
    text = text.lower()
    inputs = BERT_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = BERT_MODEL(**inputs)
    last_hidden_states = outputs.last_hidden_state
    #vec = last_hidden_states[:, 0, :].detach().numpy() # Get embedding of [CLS] token
    vec = torch.mean(last_hidden_states, dim=1).detach().numpy() # Get average of last layer
    vec = normalize(vec)[0]
    return vec
    

def compute_sparse_tokenizer():
    '''
    Compute tokenizer
    '''
    texts = get_texts()
    texts.to_csv(ARTICLES_TEXT_FILE,index=False) # Store texts in file for tokenizer
    SPARSE_TOKENIZER = Tokenizer(models.BPE())
    SPARSE_TOKENIZER.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=10)
    SPARSE_TOKENIZER.train(files=[ARTICLES_TEXT_FILE], trainer=trainer)
    SPARSE_TOKENIZER.save(SPARSE_TOKENIZER_FILE)

    
def compute_sparse_embeddings():
    '''
    Compute sparse embeddings
    '''
    texts = get_texts()
    tokens = texts.progress_apply(lambda x: SPARSE_TOKENIZER.encode(x)).tolist()
    # Turn tokens into sparse CSR matrix
    rows = [i for i, row in enumerate(tokens) for _ in row]
    cols = [col for row in tokens for col in row]
    data = np.ones(len(rows))
    vecs_sp = sp.coo_matrix((data, (rows, cols))).tocsr()
    sp.save_npz(EMBEDDINGS_SPARSE_FILE,vecs_sp)


def get_sparse_vec(text: str) -> np.ndarray:
    '''
    Generate sparse embedding for given text
    '''
    text = text.lower()
    tokens = SPARSE_TOKENIZER.encode(text)
    vec = np.zeros(SPARSE_TOKENIZER.vocab_size)
    vec[tokens] = 1
    return vec 


def get_vec(text: str) -> np.ndarray:
    '''
    Generate combined embedding for given text
    '''
    text.lower()
    vec_bert = get_bert_vec(text)
    vec_sparse = get_sparse_vec(text)
    vec = np.hstack([vec_bert,vec_sparse])
    return vec


if __name__ == "__main__":
    #compute_bert_embeddings()
    #print(get_bert_vec('test'))
    #compute_sparse_tokenizer()
    #compute_sparse_embeddings()
    #print(get_sparse_vec('test'))
    print(get_vec('test'))

