import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import csv
import nltk
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator, Vocab
import re
import pickle
from typing import List, Tuple, Optional
from torch import swapaxes
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(13)
from dataset_model import NEWSDataset, RNN_News_Classification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row[1])
            labels.append(row[0])
    # delete the first element of the list (header)
    del data[0]
    del labels[0]
    return data, labels

def tokenize_data(data):
    tokenized_data = []
    for text in data:
        text = re.sub(r'\\', ' ', text)
        text = re.sub(r'\"', ' ', text)
        text = re.sub(r'\d+', '<NUMBER>', text)
        text = text.lower()
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@^_`{|}~]', ' ', text)
        tokenized_data.append(word_tokenize(text))
    return tokenized_data

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    choice = input("Do you want to use a pretrained model? (yes/no): ")
    window_size = int(input("Enter window size: "))
    window_size = 2
    test_data, test_labels = load_data('./ANLP-2/test.csv')
    test_data = tokenize_data(test_data)
    vocab = torch.load(f'./models/svd-vocab_{str(window_size)}.pt')

    word_vectors = torch.load(f'./models/svd-word-vectors_{str(window_size)}.pt')
    test_dataset = NEWSDataset(test_data, test_labels, vocab)
    
    test = test_dataset.format(test_dataset, word_vectors)
    print("Test data preprocessed")

    if choice == "yes":
        model = RNN_News_Classification(len(vocab), 300, 128, 4, 2)
        model = torch.load(f'./models/svd-classification-model_{str(window_size)}.pt', map_location=device)
        loaders = {
            'test': DataLoader(test, batch_size=128, shuffle=False),
        }
        correct, predicted = model.evaluate(loaders, device)
        cm = confusion_matrix(correct, predicted)
        plt.matshow(cm)
        plt.title(f'Confusion matrix SVD {str(window_size)}')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig(f'confusion_matrix_svd_{str(window_size)}.png')

    else:
        train_data, train_labels = load_data('./ANLP-2/train.csv')
        train_data = tokenize_data(train_data)

        train_dataset = NEWSDataset(train_data, train_labels, vocab)
        train = train_dataset.format(train_dataset, word_vectors)

        print("Train data preprocessed")

        # split train into train and validation
        train_size = int(0.8 * len(train))
        val_size = len(train) - train_size
        train, val = torch.utils.data.random_split(train, [train_size, val_size])

        loaders = {
            'train': DataLoader(train, batch_size=128, shuffle=True),
            'test': DataLoader(test, batch_size=128, shuffle=False),
            'val': DataLoader(val, batch_size=128, shuffle=False)
        }

        model = RNN_News_Classification(len(vocab), 300, 128, len(train_dataset.labels_vocab), 2)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()

        model.fit(loaders, optimizer, criterion, 5, device)
        print("Done Training")
        model.evaluate(loaders, device)
        # save weights of the model
        torch.save(model, f'./models/svd-classification-model_{str(window_size)}.pt')
        print(f"Model saved as ./models/svd-classification-model_{str(window_size)}.pt")