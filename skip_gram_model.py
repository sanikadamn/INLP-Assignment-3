import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# okay so till now i have the indexed data, i have the vocab and the tokenized data
# now, for getting embeddings using skipgram, i need to make pairs of negative and positive samples

class Skip_Gram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Skip_Gram, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, target, context):
        target_embed = self.target_embedding(target)
        context_embed = self.context_embedding(context)
        # concatenate the embeddings
        embed = torch.cat((target_embed, context_embed), 1)
        out = self.fc(embed)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    def fit(self, loaders, optimizer, criterion, n_epochs, device):
        self.to(device)
        self.train()

        for epoch in range(n_epochs):
            self.train()
            total_loss = 0
            correct = 0
            for i, (data, labels) in enumerate(loaders['train']):
                optimizer.zero_grad()
                # convert to tensors
                target = data[0]
                context = data[1]
                target = target.to(device)
                context = context.to(device)
                output = self(target, context)
                # squeeze the output
                output = output.squeeze()
                # convert to float
                labels = labels.to(device, dtype=torch.float32)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += torch.sum((output > 0.5) == labels).item()
            print(f'Epoch: {epoch+1}/{n_epochs}, Loss: {total_loss/len(loaders["train"])}')
            print(f'Accuracy: {correct/len(loaders["train"].dataset)}')
                