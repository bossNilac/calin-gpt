import random

import numpy as np
import torch

from Week1.Day3.main import split_data, create_batch
from Week1.day2.main import load_vocabulary, tokenize


def random_weights(size):
    weights = []
    for i in range(size):
        weights.append(random.random())

    return weights

def initialize_embeddings(vocab_size, embedding_dim):
    e = [random_weights(embedding_dim) for _ in range(vocab_size)]
    e = np.array(e)
    e = torch.from_numpy(e).type(torch.FloatTensor)

    return e

if __name__ == "__main__":
    vocabulary = load_vocabulary()
    f = open('text.txt','r').read()
    tokens = tokenize(vocabulary,f)
    print(tokens)

    training_set, test_set, validation_set = split_data(0.9, 0.1, 0, tokens)

    B = 3; T = 10; C = 8; V = len(vocabulary)

    X, Y = create_batch(B, T, training_set)

    X = torch.tensor(X);Y = torch.tensor(Y)
    embeddings_table = initialize_embeddings(V, C)
    embedded_batch = embeddings_table[X].type(torch.FloatTensor)

    weights = np.random.rand(C, V)

    weight_tensor = torch.from_numpy(weights).type(torch.FloatTensor)

    logits = torch.matmul(embedded_batch, weight_tensor)

    probs = torch.softmax(logits, dim=-1)

    logits_flat = logits.reshape(B * T, V)
    Y_flat = Y.reshape(B * T)

    loss = torch.nn.functional.cross_entropy(logits_flat, Y_flat)


    print(X.shape)
    print(embeddings_table.shape)
    print(embedded_batch.shape)
    print(logits.shape)
    print(probs.shape)
    print(probs[0, 0].sum())
    print(logits_flat.shape)
    print(Y_flat.shape)
    print(loss)


