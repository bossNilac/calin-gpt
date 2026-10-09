from math import sqrt

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


#  Z = softmax(QK^T/ sqrt(D)) * V
# (QK^T) Calculate compatibility scores between tokens.
# Divide by (sqrt D): Prevent scores from becoming too large, which could make softmax overly concentrated and gradients small.
# Softmax: Convert scores into attention weights that sum to 1 across each row.
# Multiply by V: Create contextualized token representations.


def scaled_dot_product_attention(q, k, v):
    D = q.shape[-1]
    k_transpose = k.transpose(-2, -1)
    return torch.softmax(torch.matmul(q, k_transpose,).div(sqrt(D)),dim=-1).matmul(v)