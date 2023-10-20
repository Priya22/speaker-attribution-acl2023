import spacy
import time

import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
import pickle as pkl

DATA_SOURCE = 'data/pdnc_source'
WRITE_FOLDER = 'coref/outputs/spacy'
os.makedirs(WRITE_FOLDER, exist_ok=True)


def process_novel(novel, chap_dict):
    print("Processing: {}".format(novel))
    write_f = os.path.join(WRITE_FOLDER, novel)
    os.makedirs(write_f, exist_ok=True)

    ent_rows = []
    coref_rows = []

    global_coref_counter = 0

    for cid, cinf in chap_dict.items():
        ctext = cinf['text']
        cstart, cend = int(cinf['startByte']), int(cinf['endByte'])
        # print("Processing novel {}, chapter {}".format(novel, cid))
        start_time = time.time()
        doc = nlp(ctext)
        # print("Done in {} seconds".format(time.time() - start_time))

        #entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                ent_rows.append([ent.text, cstart + ent.start_char, cstart + ent.end_char])

        #corefs
        for key, vals in doc.spans.items():
            if key.split("_")[0] == 'coref':
                for val in vals:
                    coref_rows.append([global_coref_counter, val.text, cstart + val.start_char, cstart + val.end_char])
                global_coref_counter += 1
        
    with open(os.path.join(write_f, 'entities.csv'), 'w') as f:
        writer = csv.writer(f)
        for row in ent_rows:
            writer.writerow(row)
    
    with open(os.path.join(write_f, 'corefs.csv'), 'w') as f:
        writer = csv.writer(f)
        for row in coref_rows:
            writer.writerow(row)


    print()


if __name__ == '__main__':
    spacy.require_gpu()
    source_nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_coreference_web_trf")

    nlp.add_pipe('ner', source=source_nlp)

    print(nlp.pipe_names)

    metadf = pd.read_csv(os.path.join(DATA_SOURCE, 'ListOfNovels.txt'))
    novels = sorted(metadf['folder'].tolist())

    texts = []
    for novel in novels:
        with open(os.path.join(DATA_SOURCE, novel, 'novel.txt'), 'r') as f:
            text = f.read().strip()
        texts.append(text)

    novel2chaps = {}
    for novel, text in zip(novels, texts):
        print(novel)
        novel2chaps[novel] = {}
        span_ind = 0
        chap_df = pd.read_csv(os.path.join(DATA_SOURCE, novel, 'chap_info.csv'))
        para_df = pd.read_csv(os.path.join(DATA_SOURCE, novel, 'para_info.csv'))
        for cid, cstart, cend in zip(chap_df['chapID'], chap_df['textStartByte'], chap_df['textEndByte']):
            # chapt = text[cstart: cend]
            chap_paras = para_df[(para_df['startByte']>=cstart)&(para_df['endByte']<=cend)]
            chap_paras.sort_values(by='startByte', inplace=True)
            start = None
            end = None
            cur_len = 0
            cur_texts = []
            i = 0
            while i<len(chap_paras):
                row = chap_paras.iloc[i]
                pid, pstart, pend = row['paraID'], row['startByte'], row['endByte']
                if start is None:
                    start = pstart
                    end = pend

                cur_text = text[pstart: pend].split()

                if len(cur_text) > 150:
                    novel2chaps[novel][span_ind] = {
                        'startByte': start,
                        'endByte': end,
                        'text': " ".join(cur_text)
                    }
                    span_ind += 1
                    start = None
                    end = None
                    cur_texts = []
                    cur_len = 0
                    i += 1

                else:
                    cur_len += len(cur_text)
                    if cur_len > 150:
                        novel2chaps[novel][span_ind] = {
                            'startByte': start,
                            'endByte': end,
                            'text': " ".join(cur_texts)
                        }
                        span_ind += 1
                        start = None
                        end = None
                        cur_texts = []
                        cur_len = 0
                    else:
                        cur_texts.append(" ".join(cur_text))
                        end = pend
                        i += 1
            

    for novel, chap_dict in novel2chaps.items():
        process_novel(novel, chap_dict)
    