import torch
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader
from WordEmbedding import WordEmbeddingWithLinear, plot_embeddings

corpus = """India is my country. 
            Australia is a country. 
            United States is a large country. 
            The capital of India is New Delhi.
            Sydney is a city in Australia. 
            New York is a city in the United States.
            People in India speak Hindi.
            People in Australia speak English.
            People in the United States speak English."""

tokens = corpus.split()

vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in enumerate(vocab)}

encode = lambda text: [stoi[word] for word in text]
decode = lambda nums: ' '.join([itos[index] for index in nums])

data = encode(tokens)

def get_data(data):
    inputs = torch.tensor(data[:-1])
    labels = torch.tensor(data[1:])
    return inputs, labels

inputs, labels = get_data(data)
inputs_one_hot = F.one_hot(inputs, num_classes=vocab_size).float()
labels_one_hot = F.one_hot(labels, num_classes=vocab_size).float()

dataset = TensorDataset(inputs_one_hot, labels_one_hot)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

modelLinear = WordEmbeddingWithLinear(vocab_size)

plot_embeddings(modelLinear, "Initial Embeddings", itos, vocab_size)
modelLinear.train_model(data_loader, epochs=100)
plot_embeddings(modelLinear, "Embeddings After Training", itos, vocab_size)