import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import pickle as pkl

DEFAULT_HYPERS = {
    'EXCLUDE_IN_QUOTE': True,
    'USE_ONLY_EXPLICIT': False,
    'USE_PREV_SPEAKERS': True,

    'USE_IN_CUR_QUOTE': True,
    #model
    'LOOKAHEAD_WINDOW': 1,
    'PREV_WINDOW': 5,
    'MODEL_TYPE': 'char'
}

IND2LETTER = list('ABCDEFGHIJKLMNOPQRSTUVWXY') #save Z for padding symbol
LETTER2IND = {x:i for i,x in enumerate(IND2LETTER)}
PERS_PRONS = ['i', 'me', 'my', 'myself', 'mine', 'we', 'us','our']

def is_tok_spk(tok):
    return tok.split("_")[0]=='SPK'

def strip_spk(c):
    if is_tok_spk(c):
        return c.split("_")[1]
    return c

def get_seq_token(tok, mode='char'):
    if not is_tok_spk(tok):
        return tok
    
    else:
        if mode == 'token':
            return tok.split("_")[1]
        else:
            return tok

def get_mention_seqs(novel, HYPERS=None):

    if not HYPERS:
        HYPERS = DEFAULT_HYPERS

    EXCLUDE_IN_QUOTE = HYPERS['EXCLUDE_IN_QUOTE']
    USE_ONLY_EXPLICIT = HYPERS['USE_ONLY_EXPLICIT']
    USE_PREV_SPEAKERS = HYPERS['USE_PREV_SPEAKERS']

    USE_IN_CUR_QUOTE = HYPERS['USE_IN_CUR_QUOTE']
    #model
    LOOKAHEAD_WINDOW = HYPERS['LOOKAHEAD_WINDOW']
    PREV_WINDOW = HYPERS['PREV_WINDOW']
    MODEL_TYPE = HYPERS['MODEL_TYPE']


    with open('/h/vkpriya/quoteAttr/data/'+novel+'/novel.txt', 'r') as f:
        ntext = f.read().strip()

    #add whether exp or pron
    charInfo = pkl.load(open(os.path.join('/h/vkpriya/data/pdnc/'+novel+'/charInfo.dict.pkl'), 'rb'))
    
    qdf = pd.read_csv('/h/vkpriya/quoteAttr/data/'+novel+'/quote_info.csv')

    qdf['startByte'] = [eval(x)[0] for x in qdf['qSpan']]
    qdf['endByte'] = [eval(x)[1] for x in qdf['qSpan']]

    qdf.sort_values(by='startByte', inplace=True)

    qstarts = {}
    for x, y in zip(qdf['startByte'], qdf['endByte']):
        qstarts[x] = y

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

    mdf = pd.read_csv('/h/vkpriya/quoteAttr/data/'+novel+'/mentions_used.csv', index_col=0)
    mdf.sort_values(by='startByte', inplace=True)

    exp_keys = set([x.lower() for x in charInfo['name2id']])

    isExplicit = []
    for txt in mdf['text']:
        if txt.lower() in exp_keys:
            isExplicit.append(1)
        else:
            isExplicit.append(0)

    mdf['isExplicit'] = isExplicit
    mdf['inQuote'] = [byte2qpos[x] for x in mdf['startByte']]

    chapInfo = pd.read_csv('/h/vkpriya/quoteAttr/data/'+novel+'/chap_info.csv')

    chapInfo.sort_values(by='titleStartByte', inplace=True)

    chapInfo['chapID'] = [i for i in range(1, len(chapInfo)+1)]


    def get_chap_sequences():
        chap2keys = {}
        sequences = []

        for _, row in chapInfo.iterrows():
            chap = row['chapID']
            sb, eb = int(row['textStartByte']), int(row['textEndByte'])
            
            cmdf = mdf[(mdf['startByte']>=sb)&(mdf['endByte']<=eb)]
            chap_mens = cmdf['pdncID'].tolist()
            
            if EXCLUDE_IN_QUOTE:
                cmdf = cmdf[cmdf['inQuote']==0]
            
            cqdf = qdf[(qdf['startByte']>=sb)&(qdf['endByte']<=eb)]
            
            mens = cmdf['pdncID'].tolist()
            mspans = cmdf['startByte'].tolist()
            
            spks = cqdf['speakerID'].tolist()
            qspans = cqdf['startByte'].tolist()
            
            all_ids = mens + spks
            all_starts = mspans + qspans
            
            uniq_ids = list(set(all_ids + chap_mens)) #for in_quote mentions later
            id2ind = {x: str(i) for i,x in enumerate(uniq_ids)}
            
            mseqs = [id2ind[x] for x in mens]
            qseqs = ['SPK_'+id2ind[x] for x in spks]
            all_seqs = mseqs + qseqs
            sorted_seq, sorted_starts = [], []
            if len(all_seqs) > 0:
                sorted_seq, sorted_starts = zip(*sorted(zip(all_seqs, all_starts), key=lambda x: x[1]))
            
            sequences.append(sorted_seq)
            chap2keys[chap] = id2ind

        return sequences, chap2keys

    sequences, chap2keys = get_chap_sequences()


    def make_char_seqs(sequences):
        training_seqs = []
        seq_ids = []
        
        for chap_ind, chap_seq in enumerate(sequences):
            cid = chap_ind + 1
            # print("Chapter: ", cid)
            crow = chapInfo[chapInfo['chapID']==cid]
            csb, ceb = int(crow['textStartByte']), int(crow['textEndByte'])
            cqdf = qdf[(qdf['startByte']>=csb)&(qdf['endByte']<=ceb)]
    #         cqdf = cqdf.sort_values(by='startByte')
            cqids = cqdf['qID'].tolist()
            
    #         print(chap2keys[cid])
            speaker_inds = []
            for i, x in enumerate(chap_seq):
                if x.split("_")[0] == 'SPK':
                    speaker_inds.append(i)
            
            assert len(cqids) == len(speaker_inds)
            for qind,sp_ind in enumerate(speaker_inds):
                qid = cqids[qind]
                # print(qid)
                qrow = qdf[qdf['qID']==qid]
                qsb, qeb = int(qrow['startByte']), int(qrow['endByte'])
                
                cur_context = []
                cur_label = chap_seq[sp_ind]
                if MODEL_TYPE == 'token':
                    cur_label = cur_label.split("_")[1]
                
                prev_start = sp_ind - PREV_WINDOW
                next_stop = sp_ind + LOOKAHEAD_WINDOW + 1

                if not USE_PREV_SPEAKERS:
                    for ti in range(prev_start, sp_ind):
                        if (ti>=0) and (is_tok_spk(chap_seq[ti])):
                            prev_start -= 1
                    
                    for ti in range(sp_ind + 1, next_stop):
                        if (ti<len(chap_seq)) and (is_tok_spk(chap_seq[ti])):
                            next_stop += 1
                            
                            
                for ind in range(prev_start, sp_ind):
                    if ind < 0:
                        cur_context.append('P')
                    else:
                        if (not USE_PREV_SPEAKERS) and (is_tok_spk(chap_seq[ind])):
                            continue
                        else:
                            cur_context.append(get_seq_token(chap_seq[ind], MODEL_TYPE))
                
                for ind in range(sp_ind+1, next_stop):
                    if ind >= len(chap_seq):
                        cur_context.append('P')
                    else:
                        if (not USE_PREV_SPEAKERS) and (is_tok_spk(chap_seq[ind])):
                            continue
                        else:
                            cur_context.append(get_seq_token(chap_seq[ind], MODEL_TYPE))
                
                mens_in_quote = []
                if USE_IN_CUR_QUOTE:
                    qmdf = mdf[(mdf['startByte']>=qsb)&((mdf['endByte']<=qeb))]
                    mpids = qmdf['pdncID'].tolist()
                    mtxts = [x.lower() for x in qmdf['text'].tolist()]

                    mpids = [x for x,t in zip(mpids, mtxts) if t not in PERS_PRONS]
                    mens_in_quote = [chap2keys[cid][x] for x in mpids]
                
                training_seqs.append((cur_context, mens_in_quote, cur_label))
                seq_ids.append("_".join([str(cid), qid]))
                qind += 1
        
        return seq_ids, training_seqs

    seq_ids, char_seqs = make_char_seqs(sequences)

    return chap2keys, seq_ids, char_seqs

def make_training_seqs(char_seqs):
    vocab = set()
    max_len = 0
    train_seqs = []
    for char_seq in char_seqs:
        context = [strip_spk(x) for x in char_seq[0]]
        in_quote = [strip_spk(x) for x in char_seq[1]]
        in_quote = [x for x in in_quote if x in context]

        speaker = strip_spk(char_seq[2])

        # uniq_chars = sorted(list(set(context + in_quote + [speaker])))
        # char2let = {c:IND2LETTER[i] for i,c in enumerate(uniq_chars)}
        lind = 0
        char2let = {'P': 'Z'}
        for c in context + in_quote + [speaker]:
            if c not in char2let:
                char2let[c] = IND2LETTER[lind]
                lind += 1

        train_seq = [[char2let[x] for x in context], [char2let[x] for x in in_quote], char2let[speaker]]
        vocab.update(char2let.values())
        len_ = len(context) + len(in_quote) + 1
        if len_ > max_len:
            max_len = len_

        train_seqs.append(train_seq)
    
    return train_seqs, max_len, vocab
    