import json
import random
import string

import torch

class Day7Model(torch.nn.Module):
    def __init__(self, table, weights):
        super().__init__()
        self.embedding_table = torch.nn.Parameter(table)
        self.weights = torch.nn.Parameter(weights)

    def forward(self,x,y=None):
        embedded = self.embedding_table[x]
        logits = embedded @ self.weights
        if y is None:
            return logits
        b, t, v = logits.shape
        logits_flat = logits.reshape(b * t, v)
        y_flat = y.reshape(b * t)
        loss = torch.nn.functional.cross_entropy(logits_flat, y_flat)
        return logits,loss

def load_vocabulary():
    with open('vocabulary.json') as json_file:
        voc = json.load(json_file)
        reverse_voc = {}
        for key, value in voc.items():
            reverse_voc[value] = key
        return voc,reverse_voc

def tokenise(sentence,vocab,rev):
    output = []
    for char in sentence:
        if char not in vocab:
            raise Exception('Character not in vocabulary')
        output.append(vocab[char])
    if rev:
        return ''.join(output)
    return output

def split_dataset(input_, train_ratio, test_ratio, valid_ratio):
    if abs(train_ratio + test_ratio + valid_ratio - 1) > 1e-9:
        raise ValueError('Ratios do not add to 1')

    if train_ratio == 0:
        raise ValueError('train_ratio is 0')

    size = len(input_)

    train_end = int(size * train_ratio)
    test_end = train_end + int(size * test_ratio)

    train_set_ = input_[:train_end]

    if valid_ratio == 0:
        test_set_ = input_[train_end:]
        return train_set_, test_set_, None

    if test_ratio == 0:
        valid_set_ = input_[train_end:]
        return train_set_, None, valid_set_

    test_set_ = input_[train_end:test_end]
    valid_set_ = input_[test_end:]

    return train_set_, test_set, valid_set_

def x_y_batches(in_, b, t):
    last_index = len(in_) - 1 - t
    output_x = []
    output_y = []

    for i in range(b):
        random_stop = random.randint(0,last_index)
        output_x.append(in_[random_stop:random_stop + t])
        output_y.append(in_[random_stop + 1:random_stop + 1 + t])

    return torch.tensor(output_x),torch.tensor(output_y)

def create_table(voc, c):
    tensor = torch.rand(size=(len(voc), c))
    return tensor

def create_weights(c,v):
    tensor = torch.rand((c,v))
    return tensor

def embed_dataset(dataset,embd):
    return embd[dataset].type(torch.FloatTensor)

def train_loop(model_, optimizer_, steps, tr_set, val_set, b, t):
    for epoch in range(steps):
        x, y = x_y_batches(tr_set, b, t)
        optimizer_.zero_grad()
        logits,loss = model_(x, y)
        loss.backward()
        optimizer_.step()
        print('[',epoch+1,']: ',loss.item())

        if epoch % 100 == 0:
            x_, y_ = x_y_batches(val_set, b, t)
            logits_, loss_ = model_(x_, y_)
            print('validation has loss: ',loss_)

    return model_

def test_model(model_,chars,greedy,vocab,rev):
    output = first = tokenise(random.choice(string.ascii_letters),vocab,False)
    first = torch.tensor([first])

    for i in range(chars):
        logits = model_(first)
        next_logits = logits[0, -1]
        if greedy:
            next_token = torch.argmax(next_logits)
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        first = next_token.reshape(1, 1)
        output.append(next_token.item())

    print('\n========Test Done========')
    return ''.join(tokenise(output,rev,True))

if __name__ == '__main__':
    vocabulary, reverse_vocabulary = load_vocabulary()
    print(reverse_vocabulary)
    print(vocabulary)

    INPUT = open('text.txt').read()
    C = 8

    tokenized = tokenise(INPUT,vocabulary,False)
    train_set,test_set,valid_set = split_dataset(tokenized,0.8,0,0.2)
    model = Day7Model(create_table(vocabulary,C),create_weights(C,len(vocabulary)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trained_model = train_loop(model,optimizer,steps=1000,tr_set=train_set,val_set=valid_set,b=32,t=16)
    print(test_model(trained_model,200,False,vocabulary,reverse_vocabulary))





