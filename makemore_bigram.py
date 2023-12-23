import torch
from tqdm import tqdm
from utils import *
g = torch.Generator().manual_seed(2147483647)


def create_count_tensor(words, stoi):
    N = torch.zeros(len(stoi), len(stoi))
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1, ix2 = stoi[ch1], stoi[ch2]
            N[ix1, ix2] += 1 
    return N

def create_prob_tensor(N):
    N = N + 1e-9
    P = N / N.sum(dim=1, keepdim=True)
    return P

def sample_next_char(P, itos, ch, stoi):
    ix = stoi[ch]
    p = P[ix]
    next_ix = torch.multinomial(p, 1, generator=g, replacement=True).item()
    ch = itos[next_ix]
    return ch

def sample_word(P, itos, stoi):
    new_word = ''
    ch = '.'
    while True:
        ch = sample_next_char(P, itos, ch, stoi)
        if ch == '.':
            break
        new_word += ch
    
    return new_word

def calculate_log_prob(P, word, stoi):
    chs = ['.'] + list(word) + ['.']
    log_prob = 0
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        log_prob += torch.log(P[ix1, ix2])
    return log_prob

def calculate_dataset_log_prob(P, words, stoi):
    log_prob = 0
    for w in tqdm(words, desc='Calculating log prob'):
        log_prob += calculate_log_prob(P, w, stoi)
    # calculate number of characters in dataset
    n_chars = sum([len(w) + 1 for w in words])
    normalized_log_prob = log_prob / n_chars
    return normalized_log_prob

if __name__ == '__main__':
    words = read_dataset()
    b = make_bigram(words)
    stoi, itos = create_char_mapping(words)
    N = create_count_tensor(words, stoi)
    P = create_prob_tensor(N)

    for _ in range(10):
        word = sample_word(P, itos, stoi)
        print(word)

    log_prop = calculate_dataset_log_prob(P, words, stoi)

    print(f'Log prob: {log_prop:.4f}')





