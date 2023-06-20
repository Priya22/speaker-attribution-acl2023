import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import random

import basic_stats
import mod_funcs

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import pickle as pkl
import argparse
import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, metavar='FOLDER', help='path to folder with train/val/test files')
parser.add_argument('--output_folder', type=str, metavar='FOLDER', help='path to output folder')
parser.add_argument('--hyper_int', type=int, help='index of the hyperparameter comf')

novel_df = pd.read_csv('/h/vkpriya/data/pdnc/ListOfNovels.txt')
novels = sorted(novel_df['folder'])

RANDOM_SEED = 22

HYPER_COMBS = create_dataset.HYPER_COMBS

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
        # self.fc = nn.Linear(2*hidden_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.dropout =nn.Dropout(0.5)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = self.encoder(input)
        output, hidden = self.gru(input, hidden)
        # output = self.dropout(self.relu(self.fc(output)))
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
            
def make_rnnlm_seqs(og_seqs, HYPER_COMB):
    rnn_seqs = []
    for prev_cont, next_context, mseq, spk in og_seqs:
        seq = ['<B>'] + prev_cont + ['<P>'] + next_context  
        if HYPER_COMB['USE_IN_CUR_QUOTE']:
            seq = seq + ['<Q>'] + mseq 
        seq = seq + ['</Q>'] + [spk] + ['</B>']
        rnn_seqs.append(" ".join(seq))
    
    return rnn_seqs

def make_rnnlm_seqs_b(og_seqs, HYPER_COMB):
    rnn_seqs = []
    for prev_cont, next_context, mseq, spk in og_seqs:
        seq = ['<B>'] + prev_cont + next_context  
        if HYPER_COMB['USE_IN_CUR_QUOTE']:
            seq = seq + mseq 
        seq = seq + ['</Q>'] + [spk] + ['</B>']
        rnn_seqs.append(" ".join(seq))
    
    return rnn_seqs

def make_array(s):
    if isinstance(s, list):
        return s
    if isinstance(s, str) and isinstance(eval(s), list):
        return eval(s)
    return [s]

#data
def read_file(ip_file):
    seqs = []
    with open(ip_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            prev_context = make_array(row[0])
            next_context = make_array(row[1])
            mens_in_quote = make_array(row[2])
            spkr = row[3]

            # prev_context = [x.split("_")[1] for x in prev_context]
            # next_context = [x.split("_")[1] for x in next_context]
            # mens_in_quote = [x.split("_")[1] for x in mens_in_quote]
            # spkr = spkr.split("_")[1]

            seqs.append([prev_context, next_context, mens_in_quote, spkr])

    return seqs

def get_seqs(input_folder):
    '''
    Format of each datapoint: prev_context, next_context, in_quote_mentions as arrays, speaker as a single character.
    '''

    train_seqs = read_file(os.path.join(input_folder, 'train_local_seqs.csv'))
    val_seqs = read_file(os.path.join(input_folder, 'val_local_seqs.csv'))
    test_seqs = read_file(os.path.join(input_folder, 'test_local_seqs.csv'))
    
    return train_seqs, val_seqs, test_seqs

SEQ_LEN = 25
BATCH_SIZE = 32
lr = 0.001
NUM_EPOCHS = 15

def predict(model, test_x, test_y, c2i, i2c, gen_len = 20):
    model.eval()

    out_preds = []
    for sample_tx, sample_ty in zip(test_x, test_y):
        sample_x = [c2i[x] if x in c2i else c2i['UNK'] for x in sample_tx]
        sample_y = [c2i[y] if y in c2i else c2i['UNK'] for y in sample_ty]

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

        out_preds.append(' '.join(outs))

    gold_y = [' '.join(x) for x in test_y]
    return gold_y, out_preds

def get_top_k_preds(model, test_x, test_y, c2i, i2c, gen_len = 20, k = 2):
    model.eval()

    out_sequences = []
    out_probs = []

    for sample_tx, sample_ty in zip(test_x, test_y):

        sample_x = [c2i[x] if x in c2i else c2i['UNK'] for x in sample_tx]
        sample_y = [c2i[y] if y in c2i else c2i['UNK'] for y in sample_ty]

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

def train_crnn(input_folder, output_folder, hyper_int):
    HYPER_COMB = HYPER_COMBS[hyper_int]

    # og_seqs = get_lm_seqs()
    train_seqs, val_seqs, test_seqs = get_seqs(input_folder)

    train_lm_seqs = make_rnnlm_seqs(train_seqs, HYPER_COMB)
    val_lm_seqs = make_rnnlm_seqs(val_seqs, HYPER_COMB)
    test_lm_seqs = make_rnnlm_seqs(test_seqs, HYPER_COMB)

    seq_lens = [len(x.split(" ")) for x in train_lm_seqs]
    SEQ_LEN = int(np.quantile(seq_lens, 0.90))

    uniq_toks = set()
    for seq in train_lm_seqs:
        toks = seq.split(" ")
        uniq_toks.update(toks)

    print("{} unique tokens: {}".format(len(uniq_toks), uniq_toks))
    #kfold later
    # train_seqs, test_seqs = train_test_split(rnnlm_seqs, test_size=0.2, random_state=RANDOM_SEED)

    val_ip, val_op = [], []
    for ts in val_lm_seqs:
        ip, op = ts.split("</Q>")
        ip += '</Q>'
        val_ip.append(ip.split(" "))
        val_op.append(op.strip().split(" "))
    
    test_ip, test_op = [], []
    for ts in test_lm_seqs:
        ip, op = ts.split("</Q>")
        ip += '</Q>'
        test_ip.append(ip.split(" "))
        test_op.append(op.strip().split(" "))

    # train_seqs = sorted(train_seqs, key=lambda x:len(x))
    train_text = ' '.join(train_lm_seqs).split(" ")
    val_text = ' '.join(val_lm_seqs).split(" ")

    #modify seq len depending on length of ip sequences
    train_x = []
    train_y = []
    for i in range(0, len(train_text)-SEQ_LEN):
        train_x.append(train_text[i:i+SEQ_LEN])
        train_y.append(train_text[i+1:i+SEQ_LEN+1])

    val_x = []
    val_y = []
    for i in range(0, len(val_text)-SEQ_LEN):
        val_x.append(val_text[i:i+SEQ_LEN])
        val_y.append(val_text[i+1:i+SEQ_LEN+1])


    i2c = list(set(train_text))
    i2c = ['UNK'] + i2c
    c2i = {c:i for i,c in enumerate(i2c)}

    INPUT_SIZE = len(i2c)
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = len(i2c)

    def get_ind_seq(seq):
        return [c2i[c] if c in c2i else c2i['UNK'] for c in seq]

    train_data_x = [get_ind_seq(x) for x in train_x]
    train_data_y = [get_ind_seq(x) for x in train_y]

    val_data_x = [get_ind_seq(x) for x in val_x]
    val_data_y = [get_ind_seq(x) for x in val_y]

    train_data_x = torch.tensor(train_data_x).to(device)
    train_data_y = torch.Tensor(train_data_y).to(device)

    val_data_x = torch.tensor(val_data_x).to(device)
    val_data_y = torch.Tensor(val_data_y).to(device)

    train_dataset = TensorDataset(train_data_x, train_data_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset = TensorDataset(val_data_x, val_data_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    rnn_net = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(rnn_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    rnn_net.to(device)

    # rnn_net.train()
    min_val_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        rnn_net.train()
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
            
            # if batch_num %100 == 0:
            #     print("Epoch: {}; Batch: {}; loss: {}".format(epoch,batch_num, loss.item()))

            ep_losses.append(loss.item())
        
        #val loss -- accuracy?
        val_losses = []
        rnn_net.eval()
        with torch.no_grad():
            for batch_num, (X, y) in tqdm(enumerate(val_loader)):
                batch_size = X.size(0)
                hidden = rnn_net.init_hidden(batch_size).to(device)
                # optimizer.zero_grad()
                
                y_pred, hidden = rnn_net(X, hidden)
                loss = criterion(y_pred.transpose(1,2), y.type(torch.LongTensor).to(device))
                
                hidden.detach()
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            if val_loss<min_val_loss:
                min_val_loss = val_loss
                print("Saving best model: ")
                torch.save(rnn_net.state_dict(), os.path.join(output_folder, 'best_model.dict'))

        print("Epoch: {}; train loss: {}; val loss: {}".format(epoch, np.mean(ep_losses), val_loss))

    
    print("Finished training")
    torch.save(rnn_net.state_dict(), os.path.join(output_folder, 'latest_model.dict'))
    pkl.dump(i2c, open(os.path.join(output_folder, 'i2c.pkl'), 'wb'))
    pkl.dump(c2i, open(os.path.join(output_folder, 'c2i.pkl'), 'wb'))


    # test_gold_y, test_pred_y = predict(rnn_net, test_ip, test_op, c2i, i2c)
    #load best model
    rnn_net = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    rnn_net.load_state_dict(torch.load(os.path.join(output_folder, 'best_model.dict')))
    rnn_net.to(device)
    rnn_net.eval()
    val_preds, val_pred_probs = get_top_k_preds(rnn_net, val_ip, val_op, c2i, i2c, gen_len = 20, k = 10)
    test_preds, test_pred_probs = get_top_k_preds(rnn_net, test_ip, test_op, c2i, i2c, gen_len = 20, k = 10)

    with open(os.path.join(output_folder, 'val_op.csv'), 'w') as f:
        writer = csv.writer(f)
        for ip, gold, op, probs in zip(val_ip, val_op, val_preds, val_pred_probs):
            writer.writerow([ip, gold, op, probs])
    
    with open(os.path.join(output_folder, 'test_op.csv'), 'w') as f:
        writer = csv.writer(f)
        for ip, gold, op, probs in zip(test_ip, test_op, test_preds, test_pred_probs):
            writer.writerow([ip, gold, op, probs])

    print("DOne!")

if __name__ == '__main__':
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    hyper_int = int(args.hyper_int)
    train_crnn(input_folder, output_folder, hyper_int)































































































































































































































































































































