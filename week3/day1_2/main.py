import torch


def matrix_times_transpose(matrix):
    transpose = torch.transpose(matrix, 0, 1)
    return torch.matmul(matrix, transpose)

def calculate_z(matrix,vector):
    return torch.matmul(vector, matrix)

def matrix_aggregate(matrix, weights):
    return weights @ matrix

# each token is converted into a vector using your embedding table
#
# X = [x_cat,x_dog,x_sleep]
#

# W is a table of weights (0..1) pretty much saying the weight based of information of before tokens

# Z = is the scores based of the multiplication of W and X which gets you a formula
# z_sleeps = 0.1 x_1 +0.7 x_2 + 0.2 x_3

def calculate_qkv(x, wq, wk, wv):
    Q = torch.matmul(x, wq)
    K = torch.matmul(x, wk)
    V = torch.matmul(x, wv)

    return Q, K, V

# Q: What information am I looking for?
# K: What information can I be matched on?
# V: What information do I provide?
