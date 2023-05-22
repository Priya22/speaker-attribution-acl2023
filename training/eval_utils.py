import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers

from accelerate import Accelerator, DistributedType
# from accelerate.logging import get_logger
from accelerate.utils import set_seed

from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import Trainer

import os, re, sys, json, csv, string, gzip
import sklearn
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle as pkl
from tqdm import tqdm
import collections

import scipy.stats as stats
import scipy
import importlib
sys.path.append('/h/vkpriya/bookNLP/booknlp-en/')
import utils
import training.vad_dimensions.configs.generate_configs as generate_configs

def read_vad_validation_results(read_root):
    DIMS = ['arousal', 'dominance', 'valence']
    res_dfs = []

    for thresh in [100, 250, 350]:
        for dim in DIMS:
            for pool in generate_configs.TERM_EMB_AGG_METHODS:
                for axm in generate_configs.AXIS_METHODS:
    #                 sm = 'proj'
                    rows = []
                    for layer in generate_configs.LAYERS:
                        pr_corrs = []
                        sp_corrs = []
                        high_cons = []
                        low_cons = []
                        mean_cons = []
                        
                        iden_str = "-".join([str(thresh), dim, "_".join([str(x) for x in layer]), pool, axm, 'proj'])
                        fp = os.path.join(read_root, iden_str, 'res_rows.csv')
                    
                        with open(fp, 'r') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                _, pr_corr, pr_p, sp_corr, sp_p = row
                                pr_corrs.append(float(pr_corr))
                                sp_corrs.append(float(sp_corr))
                                
                        iden_str = "-".join([str(thresh), dim, "_".join([str(x) for x in layer]), pool, axm, 'cos'])
                        fp = os.path.join(read_root, iden_str, 'res_rows.csv')
                    
                        with open(fp, 'r') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                _, hc, lc, mc, _, _, _, _ = row
                                high_cons.append(float(hc))
                                low_cons.append(float(lc))
                                mean_cons.append(float(mc))
                        
                        rows.append([thresh, dim, pool, axm, "_".join([str(x) for x in layer]), \
                                    np.mean(pr_corrs), np.mean(sp_corrs), np.mean(high_cons), np.mean(low_cons), \
                                    np.mean(mean_cons)])
                        
                    resdf = pd.DataFrame(rows, columns=['thresh', 'dim', 'pool', 'axm', 'layer', 'pr_corr', \
                                                    'sp_corr', 'high_cons', 'low_cons', 'mean_cons'])
                    res_dfs.append(resdf)
    return res_dfs


def read_style_validation_results(read_root):
    DIMS = ['literary', 'abstract', 'objective', 'colloquial', 'concrete', 'subjective']

    res_dfs = []

    for dim in DIMS:
        for pool in generate_configs.TERM_EMB_AGG_METHODS:
            for axm in generate_configs.AXIS_METHODS:

                rows = []
                for layer in generate_configs.LAYERS:
                    pr_corrs = []
                    sp_corrs = []
                    high_cons = []
                    low_cons = []
                    mean_cons = []
                    
                    l_str = "_".join([str(x) for x in layer])

                    iden_str = "-".join([dim, l_str, pool, axm, 'proj'])
                    fp = os.path.join(read_root, iden_str, 'res_rows.csv')
                    
                    if not os.path.exists(fp):
                        print("Failed to find file: {}".format(iden_str))
                
                    with open(fp, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            _, pr_corr, pr_p, sp_corr, sp_p = row
                            pr_corrs.append(float(pr_corr))
                            sp_corrs.append(float(sp_corr))
                            
                    iden_str = "-".join([dim, l_str, pool, axm, 'cos'])
                    fp = os.path.join(read_root, iden_str, 'res_rows.csv')
                    if not os.path.exists(fp):
                        print("Failed to find file: {}".format(iden_str))
                
                    with open(fp, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            _, hc, lc, mc, _, _, _, _ = row
                            high_cons.append(float(hc))
                            low_cons.append(float(lc))
                            mean_cons.append(float(mc))
                    
                    rows.append([dim, pool, axm, l_str, \
                                np.mean(pr_corrs), np.mean(sp_corrs), np.mean(high_cons), np.mean(low_cons), \
                                np.mean(mean_cons)])
                    
                resdf = pd.DataFrame(rows, columns=['dim', 'pool', 'axm', 'layer', 'pr_corr', \
                                                'sp_corr', 'high_cons', 'low_cons', 'mean_cons'])
                res_dfs.append(resdf)
    return res_dfs