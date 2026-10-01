import operator

from week2.day1.main import encode, text_to_bytes, bytes_to_token_ids


def split_pairs(tokens):
    pairs = []

    if len(tokens) < 2:
        return pairs

    first_token = tokens[0]

    for token in tokens[1:]:
        pair = (first_token, token)
        pairs.append(pair)
        first_token = token

    return pairs


def count_pairs(pairs):
    pair_count = {}

    for pair in pairs:
        if pair not in pair_count:
            pair_count[pair] = 1
        else:
            pair_count[pair] += 1

    pair_count = sorted(
        pair_count.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    return pair_count


def select_best_pair(tokens):
    pairs = split_pairs(tokens)
    pair_count = count_pairs(pairs)

    if len(pair_count) == 0:
        return None

    return pair_count[0][0]


def merge_pair(tokens, pair, new_token):
    output = []
    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            output.append(new_token)
            i += 2

        else:
            output.append(tokens[i])
            i += 1

    return output


def extend_vocab(text, size):
    tokens = encode(text)

    new_vocab = {}
    current_token_value = 256

    while current_token_value < size:

        best_pair = select_best_pair(tokens)

        if best_pair is None:
            break

        new_vocab[best_pair] = current_token_value

        tokens = merge_pair(
            tokens,
            best_pair,
            current_token_value
        )

        current_token_value += 1

    return new_vocab, tokens



if __name__ == '__main__':
    text = open('text.txt', 'r').read()
    vocab_size = 598
    # Initial byte tokens
    byte_data = text_to_bytes(text)
    tokens = bytes_to_token_ids(byte_data)

    print("Original text:")
    print(text)
    print("\nInitial token ids:")
    print(tokens)
    print("\nInitial pair counts:")
    counted_pairs = count_pairs(split_pairs(tokens))
    print(counted_pairs)

    print("\nTraining BPE...")
    new_vocab, final_tokens = extend_vocab(text, vocab_size)

    print("\nLearned merge rules:")
    for pair, token_id in new_vocab.items():
        print(pair, "->", token_id)

    print("\nFinal token sequence:")
    print(final_tokens)

    print("\nInitial token count:", len(tokens))
    print("Final token count:", len(final_tokens))
    print("Number of learned tokens:", len(new_vocab))
