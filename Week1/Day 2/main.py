import json

def load_vocabulary():
    with open('vocabulary.json', 'r') as file:
        return json.load(file)

def check_error(voc, char):
    if char not in voc:
        error_message = f"The character {char} not in the vocabulary."
        raise ValueError(error_message)
    return False

def tokenize(voc, str_):
    token = []
    for char in str_:
        if not check_error(voc, char):
            token.append(voc[char])
    return token

def reverse_vocabulary(voc):
    rev_voc = {}
    for key in voc.keys():
        rev_voc.update({voc[key] : key})
    return rev_voc

def de_tokenize(rev_voc, token):
    string = ''
    for integer in token:
        if integer not in rev_voc.keys():
            error_message = f"The integer {integer} not in the vocabulary."
            raise ValueError(error_message)
        else:
            string += rev_voc[integer]
    return string


if __name__ == '__main__':
    vocabulary = load_vocabulary()
    print(vocabulary)
    reverse_voc = reverse_vocabulary(vocabulary)
    print(tokenize(vocabulary, 'hey'))
    print(de_tokenize(reverse_voc, [7,4,24]))
    print(de_tokenize(reverse_voc,tokenize(vocabulary, 'hey')) == 'hey')
    print(tokenize(vocabulary, 'hey!'))