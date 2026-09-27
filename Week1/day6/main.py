import numpy as np
import torch

from Week1.Day3.main import split_data, create_batch
from Week1.Day4.main import initialize_embeddings
from Week1.day2.main import load_vocabulary, tokenize, de_tokenize, reverse_vocabulary


class MyModule(torch.nn.Module):
    def __init__(self,table,weight):
        super().__init__()
        self.embedding_table = torch.nn.Parameter(table)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x, y=None):
        embedded = self.embedding_table[x]
        logits = embedded @ self.weight

        if y is None:
            return logits, None

        b, t, v = logits.shape
        logits_flat = logits.reshape(b * t, v)
        y_flat = y.reshape(b * t)
        loss = torch.nn.functional.cross_entropy(logits_flat, y_flat)

        return logits, loss

def train_model(b, t, c, v, steps, training_set_, validation_set_):
    embeddings_table = initialize_embeddings(v, c).type(torch.FloatTensor)
    weights = np.random.rand(c, v)
    weights = torch.from_numpy(weights)

    model = MyModule(embeddings_table, weights).type(torch.FloatTensor)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(steps):
        x, y = create_batch(b, t, training_set_)

        x = torch.tensor(x)
        y = torch.tensor(y)
        optimizer.zero_grad()
        logits_, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('At step ', step, ' the loss is: ', loss)

        if step % 100 == 0:
            with torch.no_grad():
                x_val, y_val = create_batch(b, t, validation_set_)

                x_val = torch.tensor(x_val)
                y_val = torch.tensor(y_val)

                logits_, val_loss = model(x_val, y_val)

                print('At step ', step, ' validation loss: ', val_loss)

    return model

def test_model(steps,greedy,rev_voc):
    first = input('Your first letter:')
    output = first = tokenize(vocabulary, first)
    first = torch.tensor([first])

    for i in range(steps):
        logits, _ = model(first)
        next_logits = logits[0, -1]
        if greedy:
            next_token = torch.argmax(next_logits)
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        first = next_token.reshape(1, 1)
        output.append(next_token.item())

    print(output)
    print(de_tokenize(rev_voc, output),'\n')

if __name__ == "__main__":
    vocabulary = load_vocabulary(); reverse_vocabulary = reverse_vocabulary(vocabulary)
    f = open('text.txt','r').read()
    tokens = tokenize(vocabulary,f)
    print(tokens)

    training_set, test_set, validation_set = split_data(0.9, 0, 0.1, tokens)

    B=32; T=16; C = 8; V = len(vocabulary)
    TRAINING_STEPS = 3000
    TEST_STEPS = 1000
    GREEDY = False

    model = train_model(B, T, C, V, TRAINING_STEPS,training_set,validation_set)
    test_model(TEST_STEPS,GREEDY,reverse_vocabulary)


