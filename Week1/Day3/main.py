import random
import torch

from Week1.day2.main import load_vocabulary, tokenize

INPUT = 'helloworldguys'

def get_last_index(t, input_):
    if t>= len(input_) or (t <= 0):
        raise ValueError('T bigger than input')
    else:
        return len(input_) - t - 1

def get_random_index(limit):
    return random.randint(0,limit)

def create_batch(b, t, input_):
    last_index = get_last_index(t, input_)
    x_batch = []
    y_batch = []

    for i in range(b):
        start_index = get_random_index(last_index)
        arr = input_[start_index:start_index + t]
        y_arr = input_[start_index + 1:start_index + 1 + t]
        x_batch.append(arr)
        y_batch.append(y_arr)

    return x_batch,y_batch

def split_data(train_r,test_r,val_r,input_):
    if  abs(train_r + test_r + val_r - 1.0) > 1e-9:
        raise ValueError('train_r + test_r + val_r > 1')

    size = len(input_)

    train_i = int(train_r * size)

    if val_r != 0:
        test_i = train_i + int(test_r * size)

        train_dataset = input_[:train_i]
        test_dataset = input_[train_i:test_i]
        validation_dataset = input_[test_i:]
    else:
        train_dataset = input_[:train_i]
        test_dataset = input_[train_i:]
        print ('Validation set is empty')
        return train_dataset, test_dataset, None

    return train_dataset, test_dataset, validation_dataset

if __name__ == '__main__':
    vocabulary = load_vocabulary()

    tokens = tokenize(vocabulary,INPUT)
    print(tokens)

    training_set,test_set, validation_set = split_data(0.9,0.1,0,tokens)

    B= int(input('(B) batch size:\n'))
    T= int(input('(T) token size:\n'))

    X,Y = create_batch(B,T,training_set)


    print(X)
    print(Y)

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
    print(X.dtype)