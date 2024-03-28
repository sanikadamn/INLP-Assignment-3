import torch
import torch.nn as nn
import numpy as np
import csv
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re
from torchtext.vocab import build_vocab_from_iterator
# import scipy sparse matrix to use scipy.sparse.linalg.svds
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from tqdm import tqdm
import pickle
from preprocess import Preprocess


# load textual data from csv file as a list of strings
def load_data(file_path):
    # load only the second column of the csv file
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row[1])
    # delete the first element of the list (header)
    del data[0]
    return data

# build a co-occurrence matrix
def build_co_occurrence_matrix(corpus, vocab, window_size=2):
    co_occurrence_matrix = sp.lil_matrix((len(vocab), len(vocab)), dtype=np.float32)
    for sentence in tqdm(corpus, desc='Building co-occurrence matrix'):
        # skip the first and last window_size words
        for i in range(window_size, len(sentence) - window_size):
            for j in range(i - window_size, i + window_size + 1):
                if i != j:
                    co_occurrence_matrix[sentence[i], sentence[j]] += 1
        
    return co_occurrence_matrix

def perform_svd(co_occurrence_matrix, k=300):
    U, S, V = linalg.svds(co_occurrence_matrix, k=k)
    word_vectors = get_word_vectors(U, S, k)
    return word_vectors

# Select word vectors
def get_word_vectors(U, S, k):
    word_vectors = U[:, :k]
    return word_vectors


if __name__ == '__main__':
    data = load_data('./ANLP-2/train.csv')
    window_size = int(input("Enter window size: "))
    indexed_data, vocab, tokenized_data = Preprocess(data, train=True, window_size=window_size)()
    co_occurrence_matrix = build_co_occurrence_matrix(indexed_data, vocab, window_size=window_size)
    print("Co-occurrence matrix built")
    word_vectors = perform_svd(co_occurrence_matrix)
    print("SVD performed")
    # save word vectors and vocab

    torch.save(word_vectors, f'./models/svd-word-vectors_{str(window_size)}.pt')

    torch.save(vocab, f'./models/svd-vocab_{str(window_size)}.pt')

    print("Word vectors and vocab saved")

    