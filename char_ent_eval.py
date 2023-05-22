import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
import pickle as pkl

sys.path.append('/h/vkpriya/qa_eval')
import pdnc_utils
import booknlp_utils
import gen_utils

novel2major = json.load(open('/h/vkpriya/bookNLP/booknlp-en/coref/novel2majorchars.json', 'r'))
novel2all = json.load(open('/h/vkpriya/bookNLP/booknlp-en/coref/novel2allchars.json', 'r'))

def eval_novel_booknlp(novel, limit_speakers=True):
    entdf = booknlp_utils.read_entdf(novel)

    if limit_speakers:
        qadf = booknlp_utils.read_qdf(novel)

        ass_entids = qadf['char_id'].unique().tolist()
        entdf = entdf[entdf['COREF'].isin(ass_entids)]

    else:
        entdf = entdf[(entdf['cat']=='PER')&(entdf['prop']=='PROP')]

    coref2names = {}
    name2corefs = {}
    for coref, text in zip(entdf['COREF'], entdf['text']):
        name = text.lower()
        if coref not in coref2names:
            coref2names[coref] = []
        coref2names[coref].append(name)
        if name not in name2corefs:
            name2corefs[name] = []
        name2corefs[name].append(coref)

    #PDNC
    charInfo = pdnc_utils.read_char_info(novel)
    eCharInfo = pdnc_utils.get_enhanced_char_list(charInfo['name2id'], add_lowercase=True)
    e_name2id = eCharInfo['name2id']
    e_id2names = eCharInfo['id2names']

    coref2pids = {}
    for coref, names in coref2names.items():
        if coref not in coref2pids:
            coref2pids[coref] = []
        for name in names:
            if name in e_name2id:
                coref2pids[coref].append(e_name2id[name])

    name2pids = {}
    for name, corefs in name2corefs.items():
        if name not in name2pids:
            name2pids[name] = []
        for coref in corefs:
            name2pids[name].extend(coref2pids[coref])
        

    num_unique = 0
    num_mult = 0
    num_none = 0
    unmatched_pdnc = 0
    single_pdnc = 0
    set_matched = set()
    set_unique_matched = set()

    for coref, pids in coref2pids.items():
        pids_set = set(pids)
        if len(pids_set) == 1:
            num_unique += 1
            set_unique_matched.add(pids[0])
        elif len(pids_set) == 0:
            num_none += 1
        else:
            num_mult += 1
    
        set_matched.update(pids_set)

    print(set_unique_matched)

    major_pids = [x[0] for x in novel2major[novel]]
    print(major_pids)
    for p in set(major_pids):
        if p not in set_matched:
            unmatched_pdnc += 1
        if p in set_unique_matched:
            single_pdnc += 1
    
    coref_row = [len(coref2pids), num_unique, num_mult, num_none, len(set(major_pids)), single_pdnc, unmatched_pdnc]
    props = []
    for pid in major_pids:
        cnames = set([x.lower() for x in e_id2names[pid]])
        cnames_present = [x for x in cnames if x in name2corefs]
        if len(cnames_present) > 1:
            
            cname_pids = [set(name2pids[x]) for x in cnames_present]
            print(cnames_present, cname_pids)
            set_pids = set().union(*cname_pids)
            s2count = {s: 0 for s in set_pids}
            for spids in cname_pids:
                for s in spids:
                    s2count[s] += 1
            
            max_presence = max(s2count.values())
            props.append(max_presence/len(cname_pids))

    coref_row.extend([len(props), np.mean(props)])
    
    return coref2pids, coref_row
        

NER_HEADER = ['# clusters', '# unique_pid_match', '# mult_pid_match', '# no pid match', '# pids', '# pid_unique_match', '# pid_no_match', '# chars_with_aliases', 'prop_alias_match']

