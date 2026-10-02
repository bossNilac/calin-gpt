from week2.day1.main import encode, text_to_bytes, bytes_to_token_ids
import time



def init_vocab():
    v = {}

    for i in range(256):
        v[i] = bytes([i])

    return v

def select_best_pair(tokens_):
    pair_count = {}
    max_count = 0
    key = None

    for i in range(len(tokens_) - 1):
        pair_ = (tokens_[i], tokens_[i + 1])
        if pair_ not in pair_count:
            pair_count[pair_] = 1
        else:
            pair_count[pair_] += 1
        if pair_count[pair_] > max_count:
            max_count = pair_count[pair_]
            key = pair_

    if max_count == 0:
        return None

    return key

def merge_pair(tokens_, pair_, new_token):
    output = []
    i = 0

    while i < len(tokens_):

        if (
            i < len(tokens_) - 1
            and tokens_[i] == pair_[0]
            and tokens_[i + 1] == pair_[1]
        ):
            output.append(new_token)
            i += 2

        else:
            output.append(tokens_[i])
            i += 1

    return output


def extend_vocab(text_, size, vocab_):
    tokens_ = encode(text_)
    merges = {}

    current_token_value = 256

    while current_token_value < size:

        best_pair = select_best_pair(tokens_)

        if best_pair is None:
            break

        tokens_ = merge_pair(
            tokens_,
            best_pair,
            current_token_value
        )

        vocab_[current_token_value] = vocab_[best_pair[0]] + vocab_[best_pair[1]]
        merges[best_pair] = current_token_value

        current_token_value += 1

    return tokens_, merges

def training_loop(sizes):
    text = open('bpe_benchmark_corpus_utf8.srt', 'r',encoding="utf-8").read()
    # Initial byte tokens
    byte_data = text_to_bytes(text)
    tokens = bytes_to_token_ids(byte_data)

    for size in sizes:
        vocab = init_vocab()
        old_ts = time.time()
        print("\nTraining BPE...")
        final_tokens,merge_map = extend_vocab(text, size,vocab)

        print("\nLearned merge rules:")
        for pair, token_id in merge_map.items():
            print(pair, "->", token_id)

        print("\nInitial token count:", len(tokens))
        print("Final token count:", len(final_tokens))
        print("Number of learned tokens:", len(vocab))

        res = (time.time() - old_ts)/3600
        print("\nTraining time:", res, ' for vocab size: ', size)
        print('++++++++++++++++++++++++++++++++++++++++++=')


if __name__ == '__main__':
    desired_size = [300,500,1000]
    training_loop(desired_size)

# Initial token count: 23613149
# Final token count: 15301149
# Number of learned tokens: 300
#
# Training time: 0.09008264833026462  for vocab size:  300
# ++++++++++++++++++++++++++++++++++++++++++