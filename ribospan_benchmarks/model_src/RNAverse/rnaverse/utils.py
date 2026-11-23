import random
import os
import torch
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tokenize_struct_score(struct_score_tensor, bin_num=10, null_value=-1):
    '''
    Turn struct score value into token index, using a uniform binning method

    this function accept struct score value between 0 and 1. 
    the actual bin number is bin_num + 1, include a null_value(usually -1) to represent the missing struct score
    
    for example, if bin_num = 10, null_value = -1, then the binning is like:
    input:
    tensor([[ 0.3589,  0.1585,  0.6046,  0.3905,  0.1800,  0.8631],
        [ 0.1254,  0.0892,  0.9832,  0.2812,  0.2688,  0.0184],
        [ 0.5032,  0.2345,  1.0000, -1.0000,  0.6493,  0.4585],
        [ 0.5591,  0.9754,  0.9500,  0.5000, -1.0000,  0.3509]])
    output:
    tensor([[ 3,  1,  6,  3,  1,  8],
            [ 1,  0,  9,  2,  2,  0],
            [ 5,  2,  9, 10,  6,  4],
            [ 5,  9,  9,  4, 10,  3]])
    '''
    t_div = (torch.arange(bin_num + 1)/bin_num)
    mask_1 = (struct_score_tensor != null_value)
    struct_score_tensor = struct_score_tensor * mask_1
    z = struct_score_tensor[...,None] - t_div[None,None]
    index_1 = (z.sign()>0).sum(axis=2) - 1
    mask_2 = (index_1 != -1)
    index_1 = index_1 * mask_2
    index_1 = index_1*mask_1.float() + bin_num*(1-mask_1.float())
    return index_1.long()
