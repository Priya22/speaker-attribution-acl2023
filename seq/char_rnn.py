import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pickle as pkl
import random

import basic_stats
import mod_funcs

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

novel_df = pd.read_csv('/h/vkpriya/data/pdnc/ListOfNovels.txt')
novels = sorted(novel_df['folder'])

RANDOM_SEED = 22

HYPERS = {
    'EXCLUDE_IN_QUOTE': True,
    'USE_ONLY_EXPLICIT': False,
    'USE_PREV_SPEAKERS': True,
    'USE_IN_CUR_QUOTE': True,
    'LOOKAHEAD_WINDOW': 1,
    'PREV_WINDOW': 5,
    'MODEL_TYPE': 'char'
}

SAVE_PATH = '/h/vkpriya/bookNLP/booknlp-en/seq/outputs/crnn_a'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,batch_first=True,
                          bidirectional=False)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = self.encoder(input)
        output, hidden = self.gru(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

def get_context_token(tok, char2let):
    if mod_funcs.is_tok_spk(tok):
        id_ = tok.split("_")[1]
        return 'S_' + char2let[id_]
#     else:
    return 'M_' + char2let[tok]
    
def get_in_quote_token(tok, char2let):
    return 'M_' + char2let[tok]

def get_speaker_token(tok, char2let):
    id_ = tok.split("_")[1]
    return 'S_' + char2let[id_]

def get_char2let(cont, spk):
#     cont, men, spk = seq
    chars = [mod_funcs.get_seq_token(x, mode='token') for x in cont+[spk]]
    char2let = {}
    ind = 0
    for c in chars:
        if c not in char2let:
            char2let[c] = mod_funcs.IND2LETTER[ind]
            ind += 1
    
    return char2let
            
def make_rnnlm_seqs(og_seqs):
    rnn_seqs = []
    for cont, men, spk in og_seqs:
        ct = [s for s in cont if s!='P']
        char2let = get_char2let(ct, spk)
        mn = list(set([s for s in men if s in char2let]))
        
        cseq = [get_context_token(tok, char2let) for tok in ct]
        mseq = [get_in_quote_token(tok, char2let) for tok in mn]
        sseq = [get_speaker_token(spk, char2let)]
        
        seq = ['<B>'] + cseq + ['<Q>'] + mseq + ['</Q>'] + sseq + ['</B>']
        rnn_seqs.append(" ".join(seq))
    
    return rnn_seqs

def get_lm_seqs():
    og_seqs = []
    for novel in novels:
        _, _, char_seqs = mod_funcs.get_mention_seqs(novel, HYPERS)
        og_seqs.extend(char_seqs)
    
    return og_seqs

SEQ_LEN = 50
BATCH_SIZE = 32
lr = 0.001
NUM_EPOCHS = 5

def predict(model, test_x, test_y, c2i, i2c, gen_len = 20):
    model.eval()

    out_preds = []
    for sample_tx, sample_ty in zip(test_x, test_y):
        sample_x = [c2i[x] for x in sample_tx]
        sample_y = [c2i[y] for y in sample_ty]

        sample_x = torch.tensor(sample_x).view(1, -1).to(device)
        sample_y = torch.tensor(sample_y).view(1, -1).to(device)

        hidden = model.init_hidden(1).to(device)
        for step in range(sample_x.size(1)-1):
            _, hidden = model(sample_x[:, step].view(1, -1), hidden)

        #generate
        inp = sample_x[:, -1]
        outs = []
        for _ in range(sample_y.size(-1)):
            out, hidden = model(inp.view(1, -1), hidden)
            pred_logits = out[0, -1]
            p = torch.nn.functional.softmax(pred_logits, dim=0).detach().cpu().numpy()
            argmax = np.argmax(p)
            outs.append(i2c[argmax])
            inp = torch.tensor(argmax).view(1, -1).to(device)

        out_preds.append(''.join(outs))

    
    return test_y, out_preds

def get_top_k_preds(model, test_x, test_y, c2i, i2c, gen_len = 20, k=2):
    #this should probably be beam search but I hate implementing beam search
    model.eval()

    out_sequences = []
    out_probs = []

    for sample_tx, sample_ty in zip(test_x, test_y):

        sample_x = [c2i[x] for x in sample_tx]
        sample_y = [c2i[y] for y in sample_ty]

        sample_x = torch.tensor(sample_x).view(1, -1).to(device)
        sample_y = torch.tensor(sample_y).view(1, -1).to(device)

        gen_len = sample_y.size(-1)

        hidden = model.init_hidden(1).to(device)
        for step in range(sample_x.size(1)-1):
            _, hidden = model(sample_x[:, step].view(1, -1), hidden)

        inp = sample_x[:, -1]
        out, init_hidden = model(inp.view(1, -1), hidden)

        pred_logits = out[0, -1]
        p = torch.nn.functional.softmax(pred_logits, dim=0).detach().cpu().numpy()
        arg_k = np.argsort(p)[::-1][:k]
        gen_seqs = [[i2c[i]] for i in arg_k]
        gen_probs = [[p[i]] for i in arg_k]
        
        for ind, cand in enumerate(arg_k):
            inp = torch.tensor(cand).view(1, -1).to(device)
            hidden = init_hidden
            for _ in range(sample_y.size(-1)-1):
                out, hidden = model(inp.view(1, -1), hidden)
                pred_logits = out[0, -1]
                p = torch.nn.functional.softmax(pred_logits, dim=0).detach().cpu().numpy()
                argmax = np.argmax(p)
                gen_seqs[ind].append(i2c[argmax])
                gen_probs[ind].append(p[argmax])
                inp = torch.tensor(argmax).view(1, -1).to(device)

        out_sequences.append(gen_seqs)
        out_probs.append(gen_probs)

    return out_sequences, out_probs

def train_crnn():
    og_seqs = get_lm_seqs()
    rnnlm_seqs = make_rnnlm_seqs(og_seqs)

    uniq_toks = set()
    for seq in rnnlm_seqs:
        toks = seq.split()
        uniq_toks.update(toks)

    #kfold later
    train_seqs, test_seqs = train_test_split(rnnlm_seqs, test_size=0.2, random_state=RANDOM_SEED)

    test_ip, test_op = [], []
    for ts in test_seqs:
        ip, op = ts.split("</Q>")
        ip += '</Q> S_'
        test_ip.append(ip)
        test_op.append(op[3:])  

    # train_seqs = sorted(train_seqs, key=lambda x:len(x))
    train_text = ' '.join(train_seqs)

    train_x = []
    train_y = []
    for i in range(0, len(train_text)-SEQ_LEN):
        train_x.append(train_text[i:i+SEQ_LEN])
        train_y.append(train_text[i+1:i+SEQ_LEN+1])

    i2c = list(set(train_text))
    c2i = {c:i for i,c in enumerate(i2c)}

    INPUT_SIZE = len(i2c)
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = len(i2c)

    def get_ind_seq(seq):
        return [c2i[c] for c in seq]

    data_x = [get_ind_seq(x) for x in train_x]
    data_y = [get_ind_seq(x) for x in train_y]

    data_x = torch.tensor(data_x).to(device)
    data_y = torch.Tensor(data_y).to(device)

    train_dataset = TensorDataset(data_x, data_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    rnn_net = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(rnn_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    rnn_net.to(device)

    rnn_net.train()
    for epoch in range(NUM_EPOCHS):
        ep_losses = []
        for batch_num, (X, y) in tqdm(enumerate(train_loader)):
            batch_size = X.size(0)
            hidden = rnn_net.init_hidden(batch_size).to(device)
            optimizer.zero_grad()
            
            y_pred, hidden = rnn_net(X, hidden)
            loss = criterion(y_pred.transpose(1,2), y.type(torch.LongTensor).to(device))
            
            hidden.detach()
            
            loss.backward()
            optimizer.step()
            
            if batch_num %100 == 0:
                print("Epoch: {}; Batch: {}; loss: {}".format(epoch,batch_num, loss.item()))

            ep_losses.append(loss.item())
        
        print("Epoch: {}; loss: {}".format(epoch, np.mean(ep_losses)))

    
    print("Finished training")
    torch.save(rnn_net.state_dict(), os.path.join(SAVE_PATH, 'model.dict'))
    pkl.dump(i2c, open(os.path.join(SAVE_PATH, 'i2c.pkl'), 'wb'))
    pkl.dump(c2i, open(os.path.join(SAVE_PATH, 'c2i.pkl'), 'wb'))


    gold_y, pred_y = predict(rnn_net, test_ip, test_op, c2i, i2c)

    with open(os.path.join(SAVE_PATH, 'test_op.csv'), 'w') as f:
        writer = csv.writer(f)
        for ip, op, pred in zip(test_ip, gold_y, pred_y):
            writer.writerow([ip, op, pred])

    print("DOne!")

if __name__ == '__main__':
    train_crnn()



    