import numpy as np
import torch

from Week1.Day3.main import split_data, create_batch
from Week1.Day4.main import initialize_embeddings
from Week1.day2.main import load_vocabulary, tokenize

class MyModule(torch.nn.Module):
    def __init__(self,table,weight):
        super().__init__()
        self.embedding_table = torch.nn.Parameter(table)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x, y):
        embedded = self.embedding_table[x]
        logits = embedded @ self.weight
        b, t, v = logits.shape
        logits_flat = logits.reshape(b * t, v)
        y_flat = y.reshape(b * t)
        loss = torch.nn.functional.cross_entropy(logits_flat, y_flat)

        return logits, loss


if __name__ == "__main__":
    vocabulary = load_vocabulary()
    f = open('text.txt','r').read()
    tokens = tokenize(vocabulary,f)
    print(tokens)

    training_set, test_set, validation_set = split_data(0.9, 0, 0.1, tokens)

    B=32; T=16; C = 8; V = len(vocabulary)
    TRAINING_STEPS = 1000


    embeddings_table = initialize_embeddings(V, C).type(torch.FloatTensor)
    weights = np.random.rand(C, V)
    weights = torch.from_numpy(weights)

    model = MyModule(embeddings_table, weights).type(torch.FloatTensor)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(TRAINING_STEPS):
        X, Y = create_batch(B, T, training_set)

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        optimizer.zero_grad()
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('At step ', step,' the loss is: ', loss)

        if step % 100 == 0:
            with torch.no_grad():
                X_val, Y_val = create_batch(B, T, validation_set)

                X_val = torch.tensor(X_val)
                Y_val = torch.tensor(Y_val)

                logits, val_loss = model(X_val, Y_val)

                print('At step ', step, ' validation loss: ', val_loss)
