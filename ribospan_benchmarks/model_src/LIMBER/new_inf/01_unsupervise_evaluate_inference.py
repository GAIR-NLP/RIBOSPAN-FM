import argparse
import json
import sys
from pathlib import Path
import torch
from tqdm import trange, tqdm
from pathlib import Path
import numpy as np
from math import comb
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from categorical_jarcobian_evaluation import language_model_norm_auprc


repo_root = Path("/home/yunrui12/scratch.tiwary-prj/software/RNA_FM").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaConfig, HybridMambaForMaskedLM

vocab_path = "./vocab.txt"
config_path = "./model_config.json"
weight_path = "./pytorch_model.bin"

with open(vocab_path, "r") as f:
    vocab_list = [line.strip() for line in f if line.strip()]
VOCAB = {token: i for i, token in enumerate(vocab_list)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(config_path, "r") as f:
    config_dict = json.load(f)
cfg = HybridMambaConfig(**config_dict)
model = HybridMambaForMaskedLM(cfg)
state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)

for name in ['RFAM']:
    df = pd.read_csv('../bpRNA_1m_90_'+name+'_less512.cvs')
    folder_path = "./"+name+"_less512_evaluate_inference_results"

    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    for idx in range(21142, 35000):
        try:
            idx = int(idx)
            results = language_model_norm_auprc(seq=df.loc[idx]['seq'], vocabulary=VOCAB, dbn=df.loc[idx]['dbn'], model=model, device=device)
            outputs = {'name':df.loc[idx]['name'], 'id':df.loc[idx]['id'], 'length':df.loc[idx]['length'], 'NormAUPR': results['NormAUPR']}
            outputs_df = pd.DataFrame([outputs])
            outputs_df.to_csv("./"+str(folder_path)+"/seq_"+str(idx)+"_results.csv", index=False)
            print('******************************', idx, results['NormAUPR'], '******************************')
        except Exception as e:
            print(f"[ERROR] idx={idx} failed: {e}")
            continue
