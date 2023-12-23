def read_dataset(path='names.txt'):
    words = open(path, 'r').read().splitlines()
    return words

def make_bigram(words):
    b = {}

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get((ch1, ch2), 0) + 1
    return b

def create_char_mapping(words):
    chars = list(set(''.join(words)))
    chars = sorted(chars)
    stoi = {s:i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0

    itos = {i:s for s, i in stoi.items()}
    return stoi, itos