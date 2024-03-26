import re
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from torchtext.vocab import build_vocab_from_iterator


class Preprocess():
    def __init__ (self, data, train=True, window_size=5):
        self.data = data
        self.train = train
        self.window_size = window_size

    def tokenize(self, data):
        tokenized_data = []
        for text in data:
            text = re.sub(r'\\', ' ', text)
            text = re.sub(r'\"', ' ', text)
            text = re.sub(r'\d+', '<NUMBER>', text)
            text = text.lower()
            text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@^_`{|}~]', ' ', text)
            tokenized_data.append(word_tokenize(text))
        return tokenized_data

    def convert_to_outofvocab(self, data):
        # make a dictionary of frequencies, words that appear less than 3 times are converted to <OOV>
        freq_dict = {}
        for sentence in data:
            for word in sentence:
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1

        for i in range(len(data)):
            for j in range(len(data[i])):
                if freq_dict[data[i][j]] < 2:
                    data[i][j] = '<OOV>'
        return data
    
    def build_vocab(self, tokenized_data):
        OUT_OF_VOCAB = '<OOV>'
        PAD_TAG = '<PAD>'
        START_TAG = '<BOS>'
        END_TAG = '<EOS>'
        vocab = build_vocab_from_iterator(tokenized_data, specials=[OUT_OF_VOCAB, PAD_TAG, START_TAG, END_TAG])
        return vocab
    
    def text_to_indices(self, tokenized_data, vocab):
        indexed_data = []
        for sentence in tokenized_data:
            indexed_data.append([vocab[token] for token in sentence])
        return indexed_data
    
    def add_eos_bos(self, tokenized_data):
        # add eos, bos tags based on the window size
        for i in range(len(tokenized_data)):
            tokenized_data[i] = ['<BOS>'] * (self.window_size) + tokenized_data[i] + ['<EOS>'] * (self.window_size)

        return tokenized_data
    
    
    def __call__(self):
        tokenized_data = self.tokenize(self.data)
        if self.train:
            tokenized_data = self.convert_to_outofvocab(tokenized_data)
        vocab = self.build_vocab(tokenized_data)
        tokenized_data = self.add_eos_bos(tokenized_data)
        indexed_data = self.text_to_indices(tokenized_data, vocab)
        return indexed_data, vocab, tokenized_data 