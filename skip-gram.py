import torch
import torch.nn as nn
import numpy as np
import csv
import nltk
import pickle
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re
from torchtext.vocab import build_vocab_from_iterator
# import scipy sparse matrix to use scipy.sparse.linalg.svds
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from preprocess import Preprocess
from skip_gram_model import Skip_Gram

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


# make pairs of positive and negative samples
def make_pairs(data, window_size):
    pairs = []
    for sentence in tqdm(data):
        for i in range(len(sentence)):
            for j in range(i - window_size, i + window_size + 1):
                if j < 0 or j >= len(sentence) or i == j:
                    continue
                pairs.append((sentence[i], sentence[j]))
    return pairs

def make_negative_pairs(data, vocab, num_negative_samples):
    pairs = []
    for sentence in tqdm(data):
        for i in range(len(sentence)):
            for j in range(num_negative_samples):
                rand_index = np.random.randint(len(vocab))
                pairs.append((sentence[i], rand_index))
    return pairs

# concatenate and add output labels
def make_dataset(data, vocab, window_size, num_negative_samples):
    positive_pairs = make_pairs(data, window_size)
    # remove duplicates
    positive_pairs = list(set(positive_pairs))
    negative_pairs = make_negative_pairs(data, vocab, num_negative_samples)
    # remove duplicates
    negative_pairs = list(set(negative_pairs))
    # remove pairs from negative pairs that are in positive pairs
    dataset = {}
    for pair in positive_pairs:
        dataset[pair] = 1
    for pair in tqdm(negative_pairs):
        if pair not in dataset:
            dataset[pair] = 0

    # pick negative pairs out (value = 0)
    negative_pairs = []
    for pair in dataset:
        if dataset[pair] == 0:
            negative_pairs.append(pair)

    # randomly sample negative pairs so that the number of negative pairs is equal to the number of positive pairs
    # get indices of negative pairs
    if len(negative_pairs) >= 3*len(positive_pairs):
        indices = np.random.choice(len(negative_pairs), len(positive_pairs)*3, replace=False)
        negative_pairs = [negative_pairs[i] for i in indices]
    
    return positive_pairs, negative_pairs

# create a dataset and dataloader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
        self.pairs = self.positive_pairs + self.negative_pairs
        self.labels = [1 for _ in range(len(self.positive_pairs))] + [0 for _ in range(len(self.negative_pairs))]
        # convert all the pairs and labels to tensors

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index], self.labels[index]
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    window_size = int(input("Enter window size: "))
    data = load_data('./ANLP-2/train.csv')
    indexed_data, vocab, tokenized_data = Preprocess(data, True, 1)()    
    positive_pairs, negative_pairs = make_dataset(indexed_data, vocab, window_size=window_size, num_negative_samples=5)
    
    dataset = Dataset(positive_pairs, negative_pairs)

    loaders = {
        'train': torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    }

    model = Skip_Gram(len(vocab), 300)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    n_epochs = 10

    model.fit(loaders, optimizer, criterion, n_epochs, device)

    embeddings_dict = {}
    for idx, embedding in enumerate(model.target_embedding.weight):
        embeddings_dict[idx] = embedding.to('cpu').detach().numpy()

    torch.save(embeddings_dict, f'./models/skip-gram-word-vectors_{str(window_size)}.pt')

    torch.save(vocab, f'./models/skip-gram-vocab_{str(window_size)}.pt')
    print("Embeddings and vocab saved")