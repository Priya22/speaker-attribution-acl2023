import os, re, sys, json, csv, string, gzip
import numpy as np
import importlib
import pandas as pd
from collections import Counter, defaultdict

DATA_FOLDERS = {
    'random': {
        'OP_ROOT': '/h/vkpriya/bookNLP/booknlp-en/seq/outputs/tok_rnn/random/HYPER_10',
        'IP_ROOT': '/h/vkpriya/bookNLP/booknlp-en/seq/data/random/HYPER_10',
        'WRITE_ROOT': '/h/vkpriya/bookNLP/booknlp-en/seq/results/random/'
    },
    'leave-x-out': {
        'OP_ROOT': '/h/vkpriya/bookNLP/booknlp-en/seq/outputs/tok_rnn/leave-x-out/HYPER_10',
        'IP_ROOT': '/h/vkpriya/bookNLP/booknlp-en/seq/data/leave-x-out/HYPER_10',
        'WRITE_ROOT': '/h/vkpriya/bookNLP/booknlp-en/seq/results/leave-x-out/'
    }
}

SPECIAL_TOKENS = ['<B>', '<P>', '<Q>', '</Q>', '</B>']

def get_outputs(op_folder, ip_folder, split='val'):
    lines = []
    with open(os.path.join(op_folder, split+'_op.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)
    pred_lines = []
    for ip_seq, gold_seq, op_seqs, op_probs in lines:
        ip_seq = eval(ip_seq)
        gold_seq = eval(gold_seq)
        
        viable_cands = set()
        for c in ip_seq:
            if c not in SPECIAL_TOKENS:
                viable_cands.add(c)
                # c_ = c.split("_")[-1]
                # viable_cands.add("S_" + c_)
                # viable_cands.add("M_"+c_)
        # viable_cands.add(gold_seq[0])
        
        op_seqs = eval(op_seqs)
        op_probs = eval(op_probs)
        
        assert gold_seq[-1] == '</B>'
        pred_cands = []
        for ops, opp in zip(op_seqs, op_probs):
            if ops[-1] == '</B>':
                if ops[0] in viable_cands:
                    pred_cands.append((ops[0], opp[0]))
        
        pred_lines.append([ip_seq, gold_seq[0], pred_cands])

    
    ip_gold_seqs = []
    ip_local_seqs = []
    ip_seq_qids = []

    lines = []
    with open(os.path.join(ip_folder, split+'_gold_seqs.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)
    for pc, nc, mc, spk in lines:
        ip_gold_seqs.append([eval(pc), eval(nc), eval(mc), spk])
        
    lines = []
    with open(os.path.join(ip_folder, split+'_local_seqs.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)
    for pc, nc, mc, spk in lines:
        ip_local_seqs.append([eval(pc), eval(nc), eval(mc), spk])
        

    lines = []
    with open(os.path.join(ip_folder, split+'_seq_ids.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)
    for pc, nc, mc, qid, novel in lines:
        ip_seq_qids.append([novel, qid])

    
    assert len(pred_lines) == len(ip_gold_seqs) == len(ip_local_seqs) == len(ip_seq_qids)

    op_rows = []
    for ip_gold_seq, ip_loc_seq, ip_seq_id, pred_line in zip(ip_gold_seqs, ip_local_seqs, ip_seq_qids, pred_lines):
        #first create local id to char id match
        novel = ip_seq_id[0]
        qid = ip_seq_id[1]
        ip_ = pred_line[0]
        
        assert len(ip_gold_seq) == len(ip_loc_seq) #4 elemeents in each
        gold2loc = {}
        loc2gold = {}
        
        spk_gold = ip_gold_seq[-1]
        spk_loc = ip_loc_seq[-1]
        
        for sg,sl in zip(ip_gold_seq, ip_loc_seq):
            
            if isinstance(sg, list):
                assert len(sg) == len(sl)
                for cg, cl in zip(sg, sl):
                    # g_ = cg.split("_")[-1]
                    # l_ = cl.split("_")[-1]
                    if cg not in gold2loc:
                        gold2loc[cg] = cl
                    else:
                        assert cl == gold2loc[cg]
                        
                    if cl not in loc2gold:
                        loc2gold[cl] = cg
                    else:
                        assert cg == loc2gold[cl]
            else:
                assert isinstance(sg, str)
                assert isinstance(sl, str)
                # g_ = sg.split("_")[-1]
                # l_ = sl.split("_")[-1]
                if sg not in gold2loc:
                    gold2loc[sg] = sl
                else:
                    assert gold2loc[sg] == sl
                if sl not in loc2gold:
                    loc2gold[sl] = sg
                else:
                    assert loc2gold[sl] == sg
        # print(ip_gold_seq, ip_loc_seq, loc2gold)
        assert pred_line[1] == spk_loc
        assert loc2gold[spk_loc] == spk_gold
        ipg = [loc2gold[x] if x in loc2gold else x for x in ip_]
        ipg = " ".join(ipg)
        
        gold_preds = []
        for lp in pred_line[-1]:
            # lp_ = lp[0].split("_")[-1]
            # assert lp[0] in loc2gold, lp
            if lp[0] not in loc2gold:
                cand = lp[0]
            else:
                cand = loc2gold[lp[0]]
            gold_preds.append((cand, lp[1]))
            
        op_rows.append([novel, qid, " ".join(ip_), ipg, spk_gold, gold_preds])

    return op_rows
        

def main():
    for data_split, folders in DATA_FOLDERS.items():
        print("Data split: {}".format(data_split))
        op_folder = folders['OP_ROOT']
        ip_folder = folders['IP_ROOT']
        write_folder = folders['WRITE_ROOT']

        os.makedirs(write_folder, exist_ok=True)

        for sp_num in range(5):
            print("KFold split: {}".format(sp_num))
            split_name = 'split_'+str(sp_num)
            opf = os.path.join(op_folder, split_name)
            ipf = os.path.join(ip_folder, split_name)
            wf = os.path.join(write_folder, split_name)
            os.makedirs(wf, exist_ok=True)

            for split in ['val', 'test']:
                print("Training split: {}".format(split))
                op_rows = get_outputs(opf, ipf, split=split)

                with open(os.path.join(wf, split+'_preds.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['novel', 'qID', 'gold_label', 'preds'])
                    for row in op_rows:
                        writer.writerow(row)

if __name__ == '__main__':
    main()

        