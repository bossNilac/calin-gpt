import random
import string


def main():
    key_input = open(file='sample', mode='r')
    key = key_input.read()
    arr = list(key)

    print(arr)

    probabilities_of_char_given_prev_char = find_all_pairs(arr)

    print("enter 1 for only one letter to text of preffered size")
    print("enter 2 for only one letter at a time (more controlled text generation)\n")

    path = input()

    if path == '2':
        text = letter = input("enter letter\n")

        while letter != '\n':
            next_letter, score = predict_next_letter(probabilities_of_char_given_prev_char, letter)
            text += next_letter
            print('Predicted next letter: ', next_letter, ' with score: ', score)
            print('Whole text: ', text)
            letter = input("enter letter\n")
            text += letter

        print('Final text: ', text)

    elif path == '1':
        text_size = int(input("enter text size:\n"))
        text = next_letter = input("enter a letter\n")

        while len(text) < text_size:
            next_letter, score = predict_next_letter(probabilities_of_char_given_prev_char, next_letter)
            text += next_letter

        print('Final text: ', text)



def predict_next_letter(prob, letter):
    score = -1
    output = ''
    for pair in prob.keys():
        if pair[0] == letter and prob[pair] > score :
            score = prob[pair]
            output = pair[1]

    if score == -1:
        return random.choice(string.ascii_letters),score

    return output, score


def find_all_pairs(arr: list[str]) -> dict[tuple[str, str], float]:
    pair_map: dict[tuple[str, str], int] = {}

    total_pairs = len(arr) - 1

    if total_pairs <= 0:
        return {}

    for i in range(total_pairs):
        pair = (arr[i], arr[i + 1])

        pair_map[pair] = pair_map.get(pair, 0) + 1

    # Pass the actual total number of pairs to the normalizer
    return normalise(pair_map)

def normalise(pair_map):
    # P(b|a) = count(a -> b) / total transitions starting with a

    counts_firsts: dict[str, int] = {}

    # Count how many transitions start with each character
    for pair in pair_map:
        first = pair[0]
        pair_count = pair_map[pair]

        counts_firsts[first] = counts_firsts.get(first, 0) + pair_count

    # Convert pair counts into conditional probabilities
    for pair in pair_map:
        first = pair[0]
        pair_map[pair] = pair_map[pair] / counts_firsts[first]

    return pair_map

if __name__ == '__main__':
    main()