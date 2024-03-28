import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch import swapaxes
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

OUT_OF_VOCAB = '<OOV>'
PAD_TAG = '<PAD>'
START_TAG = '<BOS>'
END_TAG = '<EOS>'

class NEWSDataset(Dataset):
    def __init__(self, sentences, labels, vocabulary: Optional[Vocab] = None):
        self.sentences = sentences
        self.labels = labels
        if vocabulary is None:
            self.vocab = build_vocab_from_iterator(self.sentences, specials=[OUT_OF_VOCAB, PAD_TAG])
            self.vocab.set_default_index(self.vocab[OUT_OF_VOCAB])
        else:
            self.vocab = vocabulary

        # set default index
        self.vocab.set_default_index(self.vocab[OUT_OF_VOCAB])

        # Extract unique labels
        self.labels_vocab = list(set(self.labels))
        # sort
        self.labels_vocab.sort()

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.vocab.lookup_indices(self.sentences[idx])), torch.tensor(self.labels_vocab.index(self.labels[idx]))

    def format(self, batch, encodings) -> Tuple[torch.Tensor, torch.Tensor]:
        # add <BOS> and <EOS> to the sentences
        sentences, labels = zip(*batch)
        sentences = list(sentences)
        # add <BOS> and <EOS> to the sentences
        sentences = [torch.tensor([self.vocab[START_TAG]] + list(s) + [self.vocab[END_TAG]]) for s in sentences]

        sentences = pad_sequence(sentences, padding_value=self.vocab[PAD_TAG])
        sentences = swapaxes(sentences, 0, 1)

        # cut sentences off at length 50
        sentences = sentences[:, :40]

        # one hot encode the labels
        labels = [torch.nn.functional.one_hot(torch.tensor(l), num_classes=len(self.labels_vocab)) for l in labels]

        # encodings is a dictionary with each index corresponding to a word encoding. Map those to the sentences
        # convert sentences to a list of list of tensors
        sentences = [[encodings[int(i)] for i in s] for s in sentences]

        # zip and return
        return list(zip(sentences, labels))

        
class RNN_News_Classification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, text):
        output, (hidden, cell) = self.rnn(text)
        final_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(final_output)
        return output

    def fit(self, loaders, optimizer, criterion, n_epochs, device):
        self.to(device)
        self.train()
        for epoch in range(n_epochs):
            self.train()
            total_loss = 0  
            for i, (sentences, labels) in enumerate(loaders['train']):
                # convert sentences to tensors
                # sentences = sentences.to(device)
                sentences = torch.stack(sentences).to(device)
                sentences = sentences.permute(1, 0, 2)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = self(sentences)
                labels = labels.float()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}') 
            print(f'Loss: {total_loss/len(loaders["train"])}')
            
            # evaluate
            self.eval()
            correct = 0
            total = 0
            total_loss = 0
            with torch.no_grad():
                for i, (sentences, labels) in enumerate(loaders['val']):
                    # convert to tensors
                    # sentences = sentences.to(device)
                    sentences = torch.stack(sentences).to(device)
                    sentences = sentences.permute(1, 0, 2)
                    labels = labels.to(device)
                    output = self(sentences)
                    predicted = torch.argmax(output, dim=1)
                    total += labels.size(0)
                    original = torch.argmax(labels, dim=1)
                    correct += (predicted == original).sum().item()
                    labels = labels.float()
                    loss = criterion(output, labels)
                    total_loss += loss.item()
            print(f'Validation Loss: {total_loss/len(loaders["val"])}')
            print(f'Validation Accuracy: {correct/total}')

            
    def evaluate(self, loaders, device):
        self.to(device)
        self.eval()
        correct = 0
        total = 0
        predicted_labels = []
        correct_labels = []
        with torch.no_grad():
            for i, (sentences, labels) in enumerate(loaders['test']):
                # convert to tensors
                sentences = torch.stack(sentences).to(device)
                sentences = sentences.permute(1, 0, 2)
                labels = labels.to(device)
                output = self(sentences)
                predicted = torch.argmax(output, dim=1)
                labels = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_labels.append(predicted)
                correct_labels.append(labels)
        print(f'Test Accuracy: {correct/total}')
        # conver correct_labels, predicted_labels to cpu and numpy
        correct_labels = torch.cat(correct_labels).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels).cpu().numpy()
        # print recall, f1 score, precision 
        print(f'Precision: {precision_score(correct_labels, predicted_labels, average="macro")}')
        print(f'Recall: {recall_score(correct_labels, predicted_labels, average="macro")}')
        print(f'F1 Score: {f1_score(correct_labels, predicted_labels, average="macro")}')
        # save the confusion matrix to a file (matplotlib)
        return correct_labels, predicted_labels


