import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import random
from argparse import ArgumentParser
import pickle as pkl

parser = ArgumentParser()
parser.add_argument('--input_folder', type=str, metavar='FOLDER', help='path to train/val/test IDs')
parser.add_argument('--output_folder', type=str, metavar='FOLDER', help='path to store train/test/val sequences')

PDNC_ROOT = '/h/vkpriya/quoteAttr/data'
IND2LETTER = list('ABCDEFGHIJKLMNOPQRSTUVWXY') #save Z for padding symbol
LETTER2IND = {x:i for i,x in enumerate(IND2LETTER)}
PERS_PRONS = ['i', 'me', 'my', 'myself', 'mine', 'we', 'us','our']

DEFAULT_HYPERS = {
    'USE_SPEAKERS': True,
    'USE_MENTIONS': True,
    'USE_IN_CUR_QUOTE': True,
    #model
    'LOOKAHEAD_WINDOW': 1,
    'PREV_WINDOW': 5,
}

# HYPER_COMBS = [
        
#     {
#         'USE_SPEAKERS': True,
#         'USE_MENTIONS': True,
#         'USE_IN_CUR_QUOTE': True,
#         #model
#         'LOOKAHEAD_WINDOW': 1,
#         'PREV_WINDOW': 5,
#     },
#     {
#         'USE_SPEAKERS': True,
#         'USE_MENTIONS': True,
#         'USE_IN_CUR_QUOTE': True,
#         #model
#         'LOOKAHEAD_WINDOW': 5,
#         'PREV_WINDOW': 5,
#     }
# ]

HYPER_COMBS = [
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': False,
        'USE_IN_CUR_QUOTE': False,
        #model
        'LOOKAHEAD_WINDOW': 1,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': False,
        'USE_IN_CUR_QUOTE': False,
        #model
        'LOOKAHEAD_WINDOW': 5,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': False,
        'USE_IN_CUR_QUOTE': True,
        #model
        'LOOKAHEAD_WINDOW': 1,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': False,
        'USE_IN_CUR_QUOTE': True,
        #model
        'LOOKAHEAD_WINDOW': 5,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': True,
        'USE_IN_CUR_QUOTE': False,
        #model
        'LOOKAHEAD_WINDOW': 1,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': True,
        'USE_IN_CUR_QUOTE': False,
        #model
        'LOOKAHEAD_WINDOW': 2,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': True,
        'USE_IN_CUR_QUOTE': True,
        #model
        'LOOKAHEAD_WINDOW': 1,
        'PREV_WINDOW': 5,
    },
    {
        'USE_SPEAKERS': True,
        'USE_MENTIONS': True,
        'USE_IN_CUR_QUOTE': True,
        #model
        'LOOKAHEAD_WINDOW': 2,
        'PREV_WINDOW': 5,
    },
]
# 9 in total
# adding last one

def read_split_file(file):
    novel2qids = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            novel = row[0]
            qid = row[1]
            if novel not in novel2qids:
                novel2qids[novel] = []
            
            novel2qids[novel].append(qid)
    
    return novel2qids


def read_train_splits(folder):
    train_file = os.path.join(folder, 'train.csv')
    val_file = os.path.join(folder, 'val.csv')
    test_file = os.path.join(folder, 'test.csv')

    train_ids = read_split_file(train_file)
    val_ids = read_split_file(val_file)
    test_ids = read_split_file(test_file)

    return train_ids, val_ids, test_ids

def generate_novel_seqs(novel,train_qids, val_qids, test_qids, HYPERS=None):
    '''
    train_qids, val_qids, test_qids: set of qIDs in each split for the novel
    '''
    

    if not HYPERS:
        HYPERS = DEFAULT_HYPERS

    USE_SPEAKERS = HYPERS['USE_SPEAKERS']
    USE_MENTIONS = HYPERS['USE_MENTIONS']
    USE_IN_CUR_QUOTE = HYPERS['USE_IN_CUR_QUOTE']
    LOOKAHEAD_WINDOW = HYPERS['LOOKAHEAD_WINDOW']
    PREV_WINDOW = HYPERS['PREV_WINDOW']

    with open(os.path.join(PDNC_ROOT, novel, 'novel.txt'), 'r') as f:
        ntext = f.read().strip()

    #add whether exp or pron
    charInfo = pkl.load(open(os.path.join(PDNC_ROOT, novel, 'charInfo.dict.pkl'), 'rb'))
    
    qdf = pd.read_csv(os.path.join(PDNC_ROOT, novel, 'quote_info.csv'))
    qdf.sort_values(by='startByte', inplace=True)

    mdf = pd.read_csv(os.path.join(PDNC_ROOT, novel, 'mentions_used.csv'))
    mdf = mdf[~pd.isna(mdf['startByte'])]
    mdf.sort_values(by='startByte', inplace=True)

    qstarts = {}
    for x, y in zip(qdf['startByte'], qdf['endByte']):
        qstarts[x] = y

    print("Novel: {}; num_quotes: {}; num_mentions: {}".format(novel, len(qdf), len(mdf)))

    #for each byte, check if inside quote or outside
    byte2qpos = {}
    byte = 0
    while byte < len(ntext):
        if byte in qstarts:
            end = qstarts[byte]
            while byte < end:
                byte2qpos[byte] = 1
                byte += 1
        else:
            byte2qpos[byte] = 0
            byte += 1

    mdf['inQuote'] = [byte2qpos[x] for x in mdf['startByte']]

    #read chapter info
    chapInfo = pd.read_csv(os.path.join(PDNC_ROOT, novel, 'chap_info.csv'))
    chapInfo.sort_values(by='titleStartByte', inplace=True)
    # chapInfo['chapID'] = [i for i in range(1, len(chapInfo)+1)]

    def get_chap_sequences():
        #only use speakers from training set
        chap2keys = {}
        sequences = []
        seq_ids = []

        for _, row in chapInfo.iterrows():
            chap = row['chapID']
            sb, eb = int(row['textStartByte']), int(row['textEndByte'])
            
            cmdf = mdf[(mdf['startByte']>=sb)&(mdf['endByte']<=eb)]
            chap_mens = cmdf['pdncID'].tolist()
            
            # if EXCLUDE_IN_QUOTE:
            cmdf = cmdf[cmdf['inQuote']==0]
            
            mens = cmdf['pdncID'].tolist()
            mspans = cmdf['startByte'].tolist()
            mids = cmdf['mID'].tolist()

            cqdf = qdf[(qdf['startByte']>=sb)&(qdf['endByte']<=eb)]

            # train_cqdf = cqdf[cqdf['qID'].isin(train_qids)]
            
            spks = cqdf['speakerID'].tolist()
            qspans = cqdf['startByte'].tolist()
            qids = cqdf['qID'].tolist()
            
            all_pids = mens + spks
            all_starts = mspans + qspans
            all_ids = mids + qids
            mseqs = ['M_' + str(x) for x in mens]
            qseqs = ['S_' + str(x) for x in spks]
            all_seqs = mseqs + qseqs

            sorted_seq, sorted_starts, sorted_ids = [], [], []
            if len(all_seqs) > 0:
                sorted_seq, sorted_starts, sorted_ids = zip(*sorted(zip(all_seqs, all_starts, all_ids), key=lambda x: x[1]))
            
            sequences.append(sorted_seq)
            seq_ids.append(sorted_ids)

        return sequences, seq_ids

    sequences, seq_ids = get_chap_sequences()
    

    def create_training_seqs(sequences, seq_ids):
        training_seqs = []
        gold_training_seqs = []
        training_seq_ids = []

        for chap_ind, chap_seq in enumerate(sequences):
            chap_seq_ids = seq_ids[chap_ind]

            # print("Chapter deets: {}".format(chap_ind+1))
            # print(chap_seq)
            # print(chap_seq_ids)

            cid = chap_ind + 1

            sp_inds = []
            chap_qids = []
            for i, x in enumerate(chap_seq):
                if x.split("_")[0] == 'S':
                    sp_inds.append(i)
                    chap_qids.append(chap_seq_ids[i])
            
            for qid, sp_ind in zip(chap_qids, sp_inds):
                qrow = qdf[qdf['qID']==qid]
                qsb, qeb = int(qrow['startByte']), int(qrow['endByte'])
                
                cur_label = chap_seq[sp_ind]

                prev_seq = chap_seq[:sp_ind]
                prev_seq_ids = chap_seq_ids[:sp_ind]

                next_seq = chap_seq[sp_ind + 1:]
                next_seq_ids = chap_seq_ids[sp_ind + 1:]

                if USE_SPEAKERS:
                    #keep only quotes that are in training set
                    keep_ids = []
                    for i, x in enumerate(prev_seq_ids):
                        if (x[0] == 'Q') and (x in train_qids):
                            keep_ids.append(i)
                        elif x[0] == 'M':
                            keep_ids.append(i)
                    prev_seq = [prev_seq[i] for i in keep_ids]
                    prev_seq_ids = [prev_seq_ids[i] for i in keep_ids]

                    keep_ids = []
                    for i, x in enumerate(next_seq_ids):
                        if (x[0] == 'Q') and (x in train_qids):
                            keep_ids.append(i)
                        elif x[0] == 'M':
                            keep_ids.append(i)
                    next_seq = [next_seq[i] for i in keep_ids]
                    next_seq_ids = [next_seq_ids[i] for i in keep_ids]

                if not USE_MENTIONS:
                    keep_ids = [i for i, x in enumerate(prev_seq) if x.split("_")[0] == 'S']
                    prev_seq = [prev_seq[i] for i in keep_ids]
                    prev_seq_ids = [prev_seq_ids[i] for i in keep_ids]

                    keep_ids = [i for i, x in enumerate(next_seq) if x.split("_")[0] == 'S']
                    next_seq = [next_seq[i] for i in keep_ids]
                    next_seq_ids = [next_seq_ids[i] for i in keep_ids]

                if not USE_SPEAKERS:
                    keep_ids = [i for i, x in enumerate(prev_seq) if x.split("_")[0] == 'M']
                    prev_seq = [prev_seq[i] for i in keep_ids]
                    prev_seq_ids = [prev_seq_ids[i] for i in keep_ids]

                    keep_ids = [i for i, x in enumerate(next_seq) if x.split("_")[0] == 'M']
                    next_seq = [next_seq[i] for i in keep_ids]
                    next_seq_ids = [next_seq_ids[i] for i in keep_ids]

                prev_start = max(0, len(prev_seq) - PREV_WINDOW)
                next_stop = min(len(next_seq), LOOKAHEAD_WINDOW + 1)

                prev_context = prev_seq[prev_start:]
                prev_context_ids = prev_seq_ids[prev_start:]
                next_context = next_seq[:next_stop]
                next_context_ids = next_seq_ids[:next_stop]
                
                chars_in_context = set([int(x.split("_")[1]) for x in prev_context + next_context])

                mens_in_quote = []
                mids = []
                if USE_IN_CUR_QUOTE:
                    qmdf = mdf[(mdf['startByte']>=qsb)&((mdf['endByte']<=qeb))]
                    mpids = [int(x) for x in qmdf['pdncID'].tolist()]
                    mids = qmdf['mID'].tolist()
                    mtxts = [x.lower() for x in qmdf['text'].tolist()]
                    mpids = [x for x,t in zip(mpids, mtxts) if t not in PERS_PRONS]
                    mids = [x for x, t in zip(mids, mtxts) if t not in PERS_PRONS]
                    keep_ids = [i for i, x in enumerate(mpids) if x in chars_in_context]
                    mpids = [mpids[i] for i in keep_ids]
                    mids = [mids[i] for i in keep_ids]
                    mens_in_quote = ['M_'+str(x) for x in mpids]

                gold_training_seqs.append([prev_context, next_context, mens_in_quote, cur_label])
                training_seq_ids.append([prev_context_ids, next_context_ids, mids, qid])

                # print(qid, prev_context, next_context, mens_in_quote, cur_label)
                #create local seqs
                lind = 0
                char2let = {}
                for c in prev_context + next_context + [cur_label]:
                    cid = c.split("_")[1]
                    if cid not in char2let:
                        char2let[cid] = IND2LETTER[lind]
                        lind += 1
                # print(char2let)

                def create_local_char(x, char2let):
                    pre, xid = x.split("_")
                    return "_".join([pre, char2let[xid]])

                training_seqs.append([
                    [create_local_char(x, char2let) for x in prev_context],
                    [create_local_char(x, char2let) for x in next_context],
                    [create_local_char(x, char2let) for x in mens_in_quote],
                    create_local_char(cur_label, char2let)
                ])
        return training_seqs, gold_training_seqs, training_seq_ids

    training_seqs, gold_training_seqs, training_seq_ids = create_training_seqs(sequences, seq_ids)
    print("Created seqs: {}/{}/{}".format(len(training_seqs), len(gold_training_seqs), len(training_seq_ids)))

    data_splits = {'train': {'local_seqs': [], 'gold_seqs': [], 'seq_ids': []}, 'val': {'local_seqs': [], 'gold_seqs': [], 'seq_ids': []}, 'test': {'local_seqs': [], 'gold_seqs': [], 'seq_ids': []}}
    tot = 0
    for loc_seq, gold_seq, seq_id in zip(training_seqs, gold_training_seqs, training_seq_ids):
        qid = seq_id[-1]
        if qid in train_qids:
            data_splits['train']['local_seqs'].append(loc_seq)
            data_splits['train']['gold_seqs'].append(gold_seq)
            data_splits['train']['seq_ids'].append(seq_id)
            tot += 1

        elif qid in val_qids:
            data_splits['val']['local_seqs'].append(loc_seq)
            data_splits['val']['gold_seqs'].append(gold_seq)
            data_splits['val']['seq_ids'].append(seq_id)
            tot += 1

        elif qid in test_qids:
            data_splits['test']['local_seqs'].append(loc_seq)
            data_splits['test']['gold_seqs'].append(gold_seq)
            data_splits['test']['seq_ids'].append(seq_id)
            tot += 1
        
        else:
            print("Quote {} was not found in any split".format(qid))

    return data_splits

def get_reformatted_token(t):
    p,c = t.split("_")
    return "_".join(['C', c])


def main(input_folder, output_folder, HYPERS=None):
    print("Reading from: ", input_folder)
    print("Writing to: ", output_folder)

    train_ids, val_ids, test_ids = read_train_splits(input_folder)
    os.makedirs(output_folder, exist_ok=True)

    novels = sorted(list(set(list(train_ids.keys()) + list(val_ids.keys()) + list(test_ids.keys()))))

    if HYPERS is None:
        HYPERS = DEFAULT_HYPERS
    
    print("Hyper: ")
    print(HYPERS)

    data_splits = {'train': {'local_seqs': [], 'gold_seqs': [], 'seq_ids': [], 'novel_ids': []}, 
                    'val': {'local_seqs': [], 'gold_seqs': [], 'seq_ids': [], 'novel_ids': []}, 
                    'test': {'local_seqs': [], 'gold_seqs': [], 'seq_ids': [], 'novel_ids': []}}
    for novel in novels:
        train_qids, val_qids, test_qids = set(), set(), set()
        if novel in train_ids:
            train_qids = set(train_ids[novel])
        
        if novel in val_ids:
            val_qids = set(val_ids[novel])
        
        if novel in test_ids:
            test_qids = set(test_ids[novel])

        novel_data_splits = generate_novel_seqs(novel, train_qids, val_qids, test_qids, HYPERS)

        for split in ['train', 'val', 'test']:
            num_points = len(novel_data_splits[split]['seq_ids'])
            if num_points > 0:
                data_splits[split]['novel_ids'].extend([novel for _ in range(num_points)])
                for key, vals in novel_data_splits[split].items():
                    data_splits[split][key].extend(vals)

    for split, inf in data_splits.items():
        print(split, inf.keys())
        assert len(inf['local_seqs']) == len(inf['gold_seqs']) == len(inf['seq_ids']) == len(inf['novel_ids'])

    # #new run jan12: without s/m indicators
    # for split in ['train', 'val', 'test']:
    #     file_name = split + '_local_seqs.csv'
    #     with open(os.path.join(output_folder, file_name), 'w') as f:
    #         writer = csv.writer(f)
    #         for row in data_splits[split]['local_seqs']:
    #             new_row = [[get_reformatted_token(x) for x in row[0]], [get_reformatted_token(x) for x in row[1]], [get_reformatted_token(x) for x in row[2]], get_reformatted_token(row[3])]
    #             writer.writerow(new_row)

    #     file_name = split + '_gold_seqs.csv'
    #     with open(os.path.join(output_folder, file_name), 'w') as f:
    #         writer = csv.writer(f)
    #         for row in data_splits[split]['gold_seqs']:
    #             writer.writerow(row)

    #     file_name = split + '_seq_ids.csv'
    #     with open(os.path.join(output_folder, file_name), 'w') as f:
    #         writer = csv.writer(f)
    #         for row1, row2 in zip(data_splits[split]['seq_ids'], data_splits[split]['novel_ids']):
    #             writer.writerow(row1 + [row2])

    #og config
    for split in ['train', 'val', 'test']:
        file_name = split + '_local_seqs.csv'
        with open(os.path.join(output_folder, file_name), 'w') as f:
            writer = csv.writer(f)
            for row in data_splits[split]['local_seqs']:
                writer.writerow(row)

        file_name = split + '_gold_seqs.csv'
        with open(os.path.join(output_folder, file_name), 'w') as f:
            writer = csv.writer(f)
            for row in data_splits[split]['gold_seqs']:
                writer.writerow(row)

        file_name = split + '_seq_ids.csv'
        with open(os.path.join(output_folder, file_name), 'w') as f:
            writer = csv.writer(f)
            for row1, row2 in zip(data_splits[split]['seq_ids'], data_splits[split]['novel_ids']):
                writer.writerow(row1 + [row2])
        


if __name__ == '__main__':
    # READ_ROOT = '/h/vkpriya/bookNLP/booknlp-en/pdnc_data/leave-x-out/'
    # WRITE_ROOT = '/h/vkpriya/bookNLP/booknlp-en/seq/data/leave-x-out'
    args = parser.parse_args()
    READ_ROOT = args.input_folder
    WRITE_ROOT = args.output_folder

    os.makedirs(WRITE_ROOT, exist_ok=True)

    for ind, HYPER in enumerate(HYPER_COMBS):
        print(ind, HYPER)
        hyper_write = os.path.join(WRITE_ROOT, 'HYPER_'+str(ind))
        os.makedirs(hyper_write, exist_ok=True)
        
        for split_f in os.scandir(READ_ROOT):
            if os.path.isdir(split_f):
                print("Random split: ", split_f.name)
                write_folder = os.path.join(hyper_write, split_f.name)
                print(write_folder)
                main(split_f, write_folder, HYPER)

