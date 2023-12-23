import torch
from tqdm import tqdm
from utils import *

g = torch.Generator().manual_seed(2147483647)


def create_simple_nn(b, stoi):
    n_chars = len(stoi)
    W = torch.randn(n_chars, n_chars, requires_grad=True)
    return W

def create_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1, ix2 = stoi[ch1], stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xs = torch.nn.functional.one_hot(xs, num_classes=len(stoi)).float()
    return xs, ys

def forward_pass(xs, ys, W):
    o1 = torch.matmul(xs, W)
    o1_exp = o1.exp()
    o1_exp_sum = o1_exp.sum(dim=1, keepdim=True)
    o2 = o1_exp / o1_exp_sum # softmax
    loss = -o2[torch.arange(len(ys)), ys].log().mean()
    return loss, o2

def backward_pass(loss, W, lr=1e-3):
    loss.backward()
    with torch.no_grad():
        W -= W.grad * lr
        W.grad.zero_()
    return W

def train(xs, ys, W, n_epochs=1000, lr=1e-3):
    print('Training...')
    for i in tqdm(range(n_epochs), desc='Training...'):
        loss, _ = forward_pass(xs, ys, W)
        if i % 10 == 0:
            print(f'Loss: {loss.item():.4f}')
        W = backward_pass(loss, W, lr=lr)
    return W

def sample_next_char(ch, stoi, itos, W):
    ix = stoi[ch]
    # encode ch
    x = torch.zeros(len(stoi))
    x[ix] = 1
    # forward pass
    o1 = torch.matmul(x, W)
    o1_exp = o1.exp()
    o1_exp_sum = o1_exp.sum()
    o2 = o1_exp / o1_exp_sum
    # sample next character
    next_ix = torch.multinomial(o2, 1, generator=g, replacement=True).item()
    ch = itos[next_ix]
    return ch

def sample_word(stoi, itos, W):
    new_word = ''
    ch = '.'
    while True:
        ch = sample_next_char(ch, stoi, itos, W)
        if ch == '.':
            break
        new_word += ch
    return new_word





if __name__ == '__main__':
    words = read_dataset()
    b = make_bigram(words)
    stoi, itos = create_char_mapping(words)
    W = create_simple_nn(b, stoi)
    xs, ys = create_dataset(words, stoi)
    W = train(xs, ys, W, n_epochs=100, lr=50)
    for _ in range(10):
        word = sample_word(stoi, itos, W)
        print(word)

