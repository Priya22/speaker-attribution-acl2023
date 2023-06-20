import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', metavar='FOLDER', type=str, help='path to folder with train/test/val files')
parser.add_argument('--output_folder', metavar='FOLDER', type=str, help='path to folder with train/test/val files')

novel_df = pd.read_csv('/h/vkpriya/data/pdnc/ListOfNovels.txt')
novels = sorted(novel_df['folder'])
RANDOM_SEED = 22

def make_lm_seq(seq):
    prev_context, next_context, in_quote, spkr = seq
    in_quote_set = sorted(list(set(in_quote)))

    all_cont = prev_context[-4:] + next_context[:4] + in_quote_set + [spkr]
    return all_cont


def get_train_probs(train_seqs):
    context_counts = {}
    for seq in train_seqs:
        cont, spkr = seq[:-1], seq[-1] 

        all_cont = " ".join(cont)
        if all_cont not in context_counts:
            context_counts[all_cont] = defaultdict(int)

        context_counts[all_cont][spkr] += 1
        
    context_to_spkr_probs = {}
    for cont, sp_counts in context_counts.items():
        norm = sum(sp_counts.values())
#         for sp, cnts in sp_counts.items():
        spks = []
        probs = []
        for sp, cnt in sp_counts.items():
            spks.append(sp)
            probs.append(cnt/norm)
        
        spks, probs = zip(*sorted(zip(spks, probs), key=lambda x:x[1], reverse=True))
        
        context_to_spkr_probs[cont] = [spks, probs]
        
    return context_to_spkr_probs

def get_top_k_preds(test_seqs, train_probs, k=1):
    preds = []
    pred_probs = []
    
    for seq in test_seqs:
        all_cont = " ".join(seq[:-1])
        
        if all_cont not in train_probs:
            #candidates: all chars in seq
            #equal probability
            spk_counter = Counter(all_cont.split(" ")).most_common()
            cands = [x[0] for x in spk_counter[:k]]
            preds.append(cands)
            pred_probs.append([1./len(cands) for _ in cands])
            # preds.append([-1 for _ in range(k)])
            # pred_probs.append([-1 for _ in range(k)])
        
        else:
            preds.append(train_probs[all_cont][0][:k])
            pred_probs.append(train_probs[all_cont][1][:k])
    
    return preds, pred_probs

#check accuracy
def get_recall_at_k(preds, gold):
    count = 0
    total = 0
    for p, g in zip(preds, gold):
        if g in p:
            count += 1
        total += 1
    return count/total

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

            prev_context = [x.split("_")[1] for x in prev_context]
            next_context = [x.split("_")[1] for x in next_context]
            mens_in_quote = [x.split("_")[1] for x in mens_in_quote]
            spkr = spkr.split("_")[1]

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

def test_split(split_folder, write_folder):
    print("Reading from: ", split_folder)
    os.makedirs(write_folder, exist_ok=True)

    train_seqs, val_seqs, test_seqs = get_seqs(split_folder)
    train_lm_seqs = [make_lm_seq(x) for x in train_seqs]
    val_lm_seqs = [make_lm_seq(x) for x in val_seqs]
    test_lm_seqs = [make_lm_seq(x) for x in test_seqs]

    train_probs = get_train_probs(train_lm_seqs)

    val_preds, val_pred_probs = get_top_k_preds(val_lm_seqs, train_probs, k = 10)
    test_preds, test_pred_probs = get_top_k_preds(test_lm_seqs, train_probs, k = 10)

    with open(os.path.join(write_folder, 'val_preds.csv'), 'w') as f:
        writer = csv.writer(f)
        for ip_seq, preds, probs in zip(val_lm_seqs, val_preds, val_pred_probs):
            ip = " ".join(ip_seq[:-1])
            gold = ip_seq[-1]
            writer.writerow([ip, gold, preds, probs])

    with open(os.path.join(write_folder, 'test_preds.csv'), 'w') as f:
        writer = csv.writer(f)
        for ip_seq, preds, probs in zip(test_lm_seqs, test_preds, test_pred_probs):
            ip = " ".join(ip_seq[:-1])
            gold = ip_seq[-1]
            writer.writerow([ip, gold, preds, probs])

    print("Outputs written: ", write_folder)

def kfold_test(all_seqs):
    rows = []
    for k in range(1, 4):
        kf = KFold(n_splits=5, random_state=RANDOM_SEED)

        for i, (train_inds, test_inds) in enumerate(kf.split(all_seqs)):
            train_seqs = [all_seqs[x] for x in train_inds]
            test_seqs = [all_seqs[x] for x in test_inds]

            train_probs = get_train_probs(train_seqs)
            test_preds, test_pred_probs = get_top_k_preds(test_seqs, train_probs, k=k)
            test_gold = [x[-1] for x in test_seqs]
            num_missed = len([x for x in test_preds if -1 in x])
            recall = get_recall_at_k(test_preds, test_gold)
            rows.append([k, i, recall, num_missed/len(test_preds)])

    res = pd.DataFrame(rows, columns=['k', 'fold', 'r@k', 'prop_missed'])
    return res

if __name__ == '__main__':
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    test_split(input_folder, output_folder)
