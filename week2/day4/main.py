from week2.day1.main import encode
from week2.day3.main import extend_vocab, init_vocab, merge_pair

with open('text.txt', 'r', encoding = 'utf-8') as f:
    train_text =  f.read()
    vocab = init_vocab()
    tokens, merges = extend_vocab(train_text, 300, vocab)


import heapq

from week2.day1.main import encode


def encode_bpe(text_, merges_):
    tokens_ = encode(text_)

    if len(tokens_) < 2:
        return tokens_

    # pair -> (rank, resulting token)
    merge_ranks = {}

    for rank, (pair_, new_token) in enumerate(merges_.items()):
        merge_ranks[pair_] = (rank, new_token)

    n = len(tokens_)

    # We keep the original positions, but link only the currently alive tokens.
    previous = [i - 1 for i in range(n)]
    next_ = [i + 1 for i in range(n)]
    next_[-1] = -1

    alive = [True] * n

    # Heap entries:
    #
    # (merge_rank, left_position, right_position, new_token)
    heap = []

    def add_pair(left):
        if left == -1 or not alive[left]:
            return

        right = next_[left]

        if right == -1 or not alive[right]:
            return

        pair_ = (tokens_[left], tokens_[right])

        if pair_ in merge_ranks:
            rank, new_token = merge_ranks[pair_]

            heapq.heappush(
                heap,
                (rank, left, right, new_token)
            )

    # Find initially mergeable adjacent pairs.
    for i in range(n - 1):
        add_pair(i)

    while heap:
        rank, left, right, new_token = heapq.heappop(heap)

        # This heap entry may be stale because an earlier merge
        # could have changed the sequence.
        if not alive[left] or not alive[right]:
            continue

        if next_[left] != right:
            continue

        pair_ = (tokens_[left], tokens_[right])

        if pair_ not in merge_ranks:
            continue

        current_rank, current_new_token = merge_ranks[pair_]

        if current_rank != rank:
            continue

        tokens_[left] = new_token
        alive[right] = False

        after_right = next_[right]

        next_[left] = after_right

        if after_right != -1:
            previous[after_right] = left

        # Only adjacency around the newly-created token changed.
        before_left = previous[left]

        if before_left != -1:
            add_pair(before_left)

        add_pair(left)

    # Reconstruct compact token sequence.
    output = []

    current = 0

    while current != -1:
        if alive[current]:
            output.append(tokens_[current])

        current = next_[current]

    return output

if __name__ == "__main__":
    merges = {
        (97, 98): 256
    }

    assert encode_bpe("ab", merges) == [256]
    assert encode_bpe("abab", merges) == [256, 256]


    merges = {
        (97, 98): 256,
        (256, 256): 257
    }

    assert encode_bpe("abab", merges) == [257]
    assert encode_bpe("ababab", merges) == [257, 256]


    assert encode_bpe("abxabab", merges) == [256, 120, 257]


    assert encode_bpe("xyz", merges) == [120, 121, 122]


    assert encode_bpe("", merges) == []
    assert encode_bpe("a", merges) == [97]


    unicode_tokens = encode_bpe("🙂", merges)
    assert unicode_tokens == encode("🙂")


    print("All encode_bpe tests passed.")

