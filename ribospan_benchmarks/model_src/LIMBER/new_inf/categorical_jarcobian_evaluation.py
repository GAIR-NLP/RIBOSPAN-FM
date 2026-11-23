import numpy as np
import torch
from tqdm import tqdm, trange
from math import comb
from sklearn.metrics import precision_recall_curve, auc

def tokenize_rna(seq: str, vocabulary : dir) -> tuple[torch.Tensor, list[int]]:
    ids = [vocabulary["[CLS]"]]
    for i in range(len(seq)):
        ids.append(vocabulary[seq[i]])
    ids.append(vocabulary["[SEP]"])
    return torch.tensor([ids], dtype=torch.long)

def compute_cj_contact_matrix(seq, model, vocabulary, device):
    x_length = len(seq)
    token_x = tokenize_rna(seq, vocabulary); token_x = token_x.to(device)
    model = model.to(device).eval()
    base_ids = [vocabulary[b] for b in "AUCG"]
    with torch.no_grad():
        wt_logits=model(token_x)['logits'][0][1:x_length+1][:, base_ids]
        categorical_jacobian = torch.zeros([x_length, 4, x_length, 4], device=device)
        token_x_tile = torch.tile(token_x, [4, 1])
        for i in trange(x_length):
            _token_x_tile = torch.clone(token_x_tile)
            _token_x_tile[:, i+1] = torch.arange(base_ids[0], base_ids[-1]+1, device=device)
            wt_logits_tile = model(_token_x_tile)['logits'][:, 1:x_length+1][:, :, base_ids]
            categorical_jacobian[i, :, :, :] = wt_logits_tile
        categorical_jacobian = categorical_jacobian - wt_logits[None, None, :, :]
    categorical_jacobian = categorical_jacobian.cpu().numpy()
    for i in range(4):
        if categorical_jacobian.shape[i] > 1:
            categorical_jacobian -= categorical_jacobian.mean(axis=i, keepdims=True)
    contact_matrix = np.sqrt(np.square(categorical_jacobian).sum(axis=(1, 3)))
    contact_matrix = (contact_matrix + contact_matrix.T) / 2
    np.fill_diagonal(contact_matrix, 0)
    average_product_correction = contact_matrix.sum(0, keepdims=True) * contact_matrix.sum(1, keepdims=True) / max(contact_matrix.sum(), 1e-8)
    contact_matrix = contact_matrix - average_product_correction
    return contact_matrix


def get_base_pairs(dbn):
    stacks = {"(": [], "[": [], "{": [], "<": []}; matching = {")": "(", "]": "[", "}": "{", ">": "<"}
    pairs = []
    for i, ch in enumerate(dbn):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in matching:
            open_ch = matching[ch]
            if stacks[open_ch]:
                j = stacks[open_ch].pop()
                pairs.append((j, i))
    return pairs


def compute_pair_baseline(dbn):
    L = len(dbn)
    pairs = get_base_pairs(dbn)
    total = comb(L, 2)
    if total == 0:
        return None
    return len(pairs) / total


def contact_matrix_to_pr_inputs(contact_matrix, dbn, min_seq_sep=1):
    contact_matrix = np.asarray(contact_matrix)
    L = len(dbn)
    if contact_matrix.shape != (L, L):
        raise ValueError(f"Shape mismatch: contact_matrix has shape {contact_matrix.shape}, "f"but dbn length is {L}.")
    true_pairs = {tuple(sorted(pair)) for pair in get_base_pairs(dbn)}
    y_true = []; y_score = []
    for i in range(L):
        for j in range(i + 1, L):
            if j - i < min_seq_sep:
                continue
            y_true.append(1 if (i, j) in true_pairs else 0)
            y_score.append(contact_matrix[i, j])
    return np.asarray(y_true), np.asarray(y_score)


def compute_auprc(contact_matrix, dbn, min_seq_sep=1):
    y_true, y_score = contact_matrix_to_pr_inputs(contact_matrix=contact_matrix, dbn=dbn, min_seq_sep=min_seq_sep)
    if len(y_true) == 0:
        raise ValueError("No nucleotide pairs were evaluated. Check min_seq_sep or dbn length.")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    baseline = compute_pair_baseline(dbn)
    if baseline is None or baseline >= 1.0:
        norm_aupr = np.nan
    else:
        norm_aupr = (aupr - baseline) / (1.0 - baseline)
    return {"NormAUPR": norm_aupr,
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds}

      
def language_model_norm_auprc(seq, vocabulary, dbn, model, device, min_seq_sep=4):
    contact_matrix = compute_cj_contact_matrix(seq=seq, model=model, vocabulary=vocabulary, device=device)
    return compute_auprc(contact_matrix=contact_matrix, dbn=dbn, min_seq_sep=min_seq_sep)
