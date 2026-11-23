import argparse
import os
import numpy as np
import random
import time
import torch
import joblib
from tqdm import tqdm

from rnaverse import RNAverseForMaskedLM, RNAverseModel, RNAverseTokenizer, RNAverseConfig
from rnaverse.tasks.protein_binding import PrismNetLM


def parse_arguments():
    parser = argparse.ArgumentParser(description='Genetic algorithm for UTR sequence optimization')
    
    # Input/Output parameters
    parser.add_argument('--out_dir', type=str, default="evo_out/", help='output directory')

    # Sequence design parameters
    parser.add_argument('--cds', type=str, default="ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGT", help='50nt of CDS')
    parser.add_argument('--design_length', type=int, default=150, help='design length of 5UTR')
    parser.add_argument('--nbr_sequences', type=int, default=500, help='number of sequences to generate')
    parser.add_argument('--no_uaug', action='store_true', default=True, help='exclude uAUG')
    parser.add_argument('--no_stop', action='store_true', default=False, help='exclude stop codons')
    
    # Genetic algorithm parameters
    parser.add_argument('--evolved_iter', type=int, default=5, help='number of evolution iterations')
    parser.add_argument('--nbr_bases_to_mutate', type=int, default=5, help='number of bases to mutate')
    
    # Model parameters
    parser.add_argument('--rnaverse_shape_model_path', type=str, required=True,
                        help='Path to RNAverse-shape model for shape prediction')
    parser.add_argument('--rnaverse_utr_model_file', type=str, required=True,
                        help='Path to RNAverse-UTR model file (best_model.pt)')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use (cuda or cpu)')
    
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass


def vectorizeSequence(seq):
    """Convert RNA sequence to one-hot encoding"""
    ltrdict = {'a':[1,0,0,0], 'c':[0,1,0,0], 'g':[0,0,1,0], 't':[0,0,0,1], 'n':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq.lower()])


def ret_rand_nuc():
    """Return random nucleotide one-hot encoding"""
    x = random.randint(0, 3)
    return [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]][x]


def vector_to_nuc(arr):
    """Convert one-hot encoding back to RNA sequence"""
    seq = ''
    nuc_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    for i in range(len(arr)):
        for j in range(4):
            if arr[i, j] == 1:
                seq += nuc_map[j]
                break
    return seq


def save_fafile(out_dir, filename, seqs):
    probs_path = os.path.join(out_dir, filename + '.fa')
    with open(probs_path, "w") as f:
        for i in range(len(seqs)):
            f.write(">seq" + str(i) + "\n")
            f.write(seqs[i] + "\n")
    print("output fasta file:", probs_path)


def cut_bin_10(shape_num):
    """Discretize shape values into 10 bins"""
    if shape_num == -1:
        return 0
    elif 0 <= shape_num <= 0.1:
        return 1
    elif 0.1 < shape_num <= 0.2:
        return 2
    elif 0.2 < shape_num <= 0.3:
        return 3
    elif 0.3 < shape_num <= 0.4:
        return 4
    elif 0.4 < shape_num <= 0.5:
        return 5
    elif 0.5 < shape_num <= 0.6:
        return 6
    elif 0.6 < shape_num <= 0.7:
        return 7
    elif 0.7 < shape_num <= 0.8:
        return 8
    elif 0.8 < shape_num <= 0.9:
        return 9
    elif shape_num > 0.9:
        return 10
    return 0


def turn_first_and_last_shape_to_unk(shape_list, first_num, last_num):
    """Set first and last N shape values to unknown"""
    shape_list_new = []
    for i, shape in enumerate(shape_list):
        if i < first_num or i >= len(shape_list) - last_num:
            shape_list_new.append(-1.0)
        else:
            shape_list_new.append(shape)
    return shape_list_new



def make_random_sequences(nbr_sequences, length, constant='', no_uaug=False, no_stop=False):
    """Generate random RNA sequences with option to exclude uAUG and stop codons"""
    seqs = []
    nucs = {0:'A', 1:'T', 2:'C', 3:'G'}
    i = 0
    while i < nbr_sequences:
        new_seq = ''
        for n in range(length - len(constant)):
            new_seq = new_seq + nucs[random.randint(0, 3)]
        if no_uaug == False or (no_uaug == True and 'ATG' not in new_seq):
            if no_stop == False or (no_stop == True and ('TAG' not in new_seq and 'TGA' not in new_seq and 'TAA' not in new_seq)):
                new_seq = new_seq + constant
                seqs.append(new_seq)
                i += 1
    return seqs


def simple_mutate(seq, muta_len=150, nbr_bases=1, prob=1):
    """Perform simple mutation on sequence"""
    if nbr_bases > 1 and prob > random.random():
        nbr_bases = nbr_bases
    else:
        nbr_bases = 1
    for i in range(nbr_bases):
        pos = random.randint(0, muta_len-1)
        seq[pos] = ret_rand_nuc()
    return seq


def check_for_uaug(seq):
    """Check if sequence contains uAUG"""
    seq = vector_to_nuc(seq)
    return 'ATG' in seq[:150]


def check_for_stops(seq):
    """Check if sequence contains stop codons"""
    seq = vector_to_nuc(seq)
    if 'TAG' in seq[:150] or 'TGA' in seq[:150] or 'TAA' in seq[:150]:
        return True
    return False


def selection_to_target3(seq, no_uaug=False, no_stop=False, nbr_bases_to_mutate=5, multi_mutate_prob=1, mut_len=150):
    """Conditional mutation selection strategy"""
    seqs = np.empty([2, len(seq), 4])
    seqs[0] = seq.copy()
    
    flag1 = 1
    flag2 = 1
    while flag1 + flag2 > 0:
        seqs[1] = simple_mutate(seq.copy(), mut_len, nbr_bases=nbr_bases_to_mutate, prob=multi_mutate_prob)
        
        if no_uaug == True:
            if not check_for_uaug(seqs[1][0:mut_len]):
                flag1 = 0
        else:
            flag1 = 0
        if no_stop == True:
            if not check_for_stops(seqs[1][0:mut_len]):
                flag2 = 0
        else:
            flag2 = 0
    return seqs[1]


def genetic_initiation(cds_seq, out_dir, filename, seq_len, no_uaug=True, no_stop=False, nbr_sequences=500):
    """Randomly generate initial sequences + CDS"""
    print('Generating {} initial sequences with length {}'.format(nbr_sequences, seq_len))
    
    rand_seqs = make_random_sequences(nbr_sequences, seq_len, constant=cds_seq, no_uaug=no_uaug, no_stop=no_stop)
    save_fafile(out_dir, filename, rand_seqs)
    
    return rand_seqs


def genetic_mutation(init_seqs, seq_len, no_uaug=True, no_stop=False, nbr_bases_to_mutate=5, multi_mutate_prob=1.0):
    """Mutate initial sequences"""
    test_sequences = np.empty([len(init_seqs), seq_len, 4])
    evol_sequences = np.empty([len(init_seqs), seq_len, 4])
    evolved_seqs = []
    
    # One-hot encode sequences
    for i, seq in enumerate(init_seqs):
        test_sequences[i] = vectorizeSequence(seq.lower())
    
    # Mutate sequences (only mutate UTR part, keep CDS)
    cds_len = 50  # CDS length
    mut_len = seq_len - cds_len
    
    for i in range(len(test_sequences)):
        evol_sequences[i] = selection_to_target3(
            seq=test_sequences[i],
            no_uaug=no_uaug,
            no_stop=no_stop,
            nbr_bases_to_mutate=nbr_bases_to_mutate,
            multi_mutate_prob=multi_mutate_prob,
            mut_len=mut_len
        )
        evolved_seqs.append(vector_to_nuc(evol_sequences[i]))
    
    return evolved_seqs



def tailor_batch_shape(seq_list, tokenizer):
    """Prepare batch data for SHAPE prediction"""
    input_dict_list = []
    for seq in seq_list:
        input_dict_list.append({
            'seq': seq,
            'probe_method': 'SHAPE-MaP',
        })
    data_dict = tokenizer(input_dict_list, padding=True, return_tensors='pt')
    return data_dict


def tailor_batch_utr(seq_shape_list, tokenizer):
    """Prepare batch data for UTR prediction"""
    seqs = []
    shape_value_list_lisan = []
    shape_value_list = []
    
    for seq, shape_value in seq_shape_list:
        seqs.append(seq)
        shape_value_lisan = torch.tensor([cut_bin_10(y) for y in shape_value])
        shape_value_list_lisan.append(shape_value_lisan)
        shape_value_list.append(shape_value)
    
    max_len = max([len(seq) for seq in seqs])
    weight_mask = torch.ones((len(seqs), max_len + 1))
    
    input_dict_list = []
    for seq, shape_list in zip(seqs, shape_value_list):
        input_dict_list.append({
            'seq': seq,
            'struct_score_list': shape_list,
            'probe_method': 'SHAPE-MaP',
        })
    
    data_dict = tokenizer(input_dict_list, padding=True, return_tensors='pt')
    post_token_length = torch.zeros(data_dict['attention_mask'].shape)
    
    data_dict['shape_value_lisan'] = torch.cat([x1.unsqueeze(0) for x1 in shape_value_list_lisan], 0)
    data_dict['weight_mask'] = weight_mask
    data_dict['post_token_length'] = post_token_length
    
    return data_dict


def infer_shape(args, seq_list):
    # infer SHAPE values using RNAverse-shape model
    print("Inferring SHAPE values...")
    tokenizer = RNAverseTokenizer.from_pretrained(args.rnaverse_shape_model_path)
    model = RNAverseForMaskedLM.from_pretrained(args.rnaverse_shape_model_path)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    data_loader = torch.utils.data.DataLoader(
        seq_list, batch_size=args.bs, shuffle=False,
        collate_fn=lambda x: tailor_batch_shape(x, tokenizer)
    )
    
    pred_list = []
    for batch in tqdm(data_loader, desc="Inference SHAPE"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            shape_output = torch.sigmoid(outputs.struct_logits[:, 1:, :].squeeze(2))
            for i in range(shape_output.shape[0]):
                pred_list.append(shape_output[i].cpu().detach().numpy().tolist())
    
    return_list = []
    for i, seq in enumerate(seq_list):
        shape_list = pred_list[i]
        shape_list_new = turn_first_and_last_shape_to_unk(shape_list, 3, 0)
        return_list.append((seq, shape_list_new))
    
    return return_list


def infer_utr(args, seq_shape_list):
    # infer RNA translation value as fitness using RNAverse-UTR model, given sequence and SHAPE
    print("Inferring UTR fitness values...")
    
    seq_shape_list_input = seq_shape_list


    transformer_file = os.path.join(os.path.dirname(args.rnaverse_utr_model_file), 'transformer.pkl')
    print(f"Loading transformer from: {transformer_file}")
    with open(transformer_file, 'rb') as f:
        loaded_scaler = joblib.load(f)
    
    # Load tokenizer and extractor
    rnaverse_utr_model_path = os.path.dirname(args.rnaverse_utr_model_file)
    tokenizer = RNAverseTokenizer.from_pretrained(rnaverse_utr_model_path)
    config = RNAverseConfig.from_pretrained(rnaverse_utr_model_path)
    extractor = RNAverseModel(config, add_pooling_layer=False)

    class Args:
        def __init__(self):
            self.model_type = 'rnaverse'
    
    args_for_model = Args()
    model = PrismNetLM(args_for_model, extractor, tokenizer)
    
    # Load fine-tuned weights
    dict_pretrain = torch.load(args.rnaverse_utr_model_file, map_location='cpu')
    model.load_state_dict(dict_pretrain)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    # Inference
    data_loader = torch.utils.data.DataLoader(
        seq_shape_list_input, batch_size=args.bs, shuffle=False,
        collate_fn=lambda x: tailor_batch_utr(x, tokenizer)
    )
    
    pred_list = []
    for batch in tqdm(data_loader, desc="Inference RNA translation"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch).squeeze(-1)
            preds_np = outputs.cpu().detach().numpy().reshape(-1, 1)
            preds_np_scaled = loaded_scaler.inverse_transform(preds_np)
            pred_list.extend(preds_np_scaled.flatten().tolist())
    
    return pred_list


def predict_fitness(args, seq_list):
    # infer SHAPE values first, then infer RNA translation value as fitness using RNAverse-UTR model, given sequence and SHAPE
    # turn T to U for RNAverse input, although for RNAverse, U and T are treated the same
    seq_list_rna = [seq.replace('T', 'U') for seq in seq_list]
    
    # Predict shape
    seq_shape_list = infer_shape(args, seq_list_rna)
    
    # Predict UTR fitness
    fitness_list = infer_utr(args, seq_shape_list)
    
    return fitness_list


def main(args):
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    seed_everything(args.random_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Sequence length (UTR + CDS)
    seq_len = args.design_length + len(args.cds)
    
    # Generate initial sequences
    print("Generating initial sequences...")
    init_seqs = genetic_initiation(
        cds_seq=args.cds,
        out_dir=args.out_dir,
        filename="random_seq01",
        seq_len=seq_len,
        no_uaug=args.no_uaug,
        no_stop=args.no_stop,
        nbr_sequences=args.nbr_sequences
    )
    
    # Predict fitness for initial sequences
    print("Predicting fitness for initial sequences...")
    orig_seqs = init_seqs.copy()
    orig_prob = predict_fitness(args, orig_seqs)
    
    # Save initial results
    out_result_file = os.path.join(args.out_dir, 'final_results.tsv')
    fw = open(out_result_file, 'w')
    fw.write("Generation\tIndex\tSequence\tFitness\n")
    for i in range(len(orig_seqs)):
        outstr = "0\t" + str(i) + "\t" + orig_seqs[i] + "\t" + str(round(orig_prob[i], 3)) + "\n"
        fw.write(outstr)
    
    # Genetic algorithm iteration
    print("Starting evolution iterations...")
    tick1 = time.time()
    
    for k in range(args.evolved_iter):
        print('-'*10 + f'=iteration: {k + 1}/{args.evolved_iter}' + '-'*10)
        
        # Mutate sequences
        evol_seqs = genetic_mutation(
            init_seqs=orig_seqs,
            seq_len=seq_len,
            no_uaug=args.no_uaug,
            no_stop=args.no_stop,
            nbr_bases_to_mutate=args.nbr_bases_to_mutate,
        )
        
        # Predict fitness for mutated sequences
        new_prob = predict_fitness(args, evol_seqs)
        
        # Select better sequences
        for i in range(len(orig_seqs)):
            if new_prob[i] > orig_prob[i]:
                orig_prob[i] = new_prob[i]
                orig_seqs[i] = evol_seqs[i]
        
        # Save current iteration results
        for i in range(len(orig_seqs)):
            outstr = str(k + 1) + "\t" + str(i) + "\t" + orig_seqs[i] + "\t" + str(round(orig_prob[i], 3)) + "\n"
            fw.write(outstr)
    
    fw.close()
    
    tick2 = time.time()
    print(f"[Completed] Total time: {tick2 - tick1:.2f} seconds")
    print(f"Results saved to: {out_result_file}")
    
    # Print best result
    best_idx = np.argmax(orig_prob)
    print(f"Best sequence (fitness={orig_prob[best_idx]:.3f}):")
    print(f"  {orig_seqs[best_idx]}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
