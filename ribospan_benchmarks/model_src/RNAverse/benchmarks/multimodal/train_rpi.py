import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from rnaverse.tasks.protein_binding import get_extractor, GradualWarmupScheduler, SeqicSHAPE, PrismNetLM, tailor_batch_prism_opensource
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from transformers import set_seed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train protein-RNA binding prediction model lm + prismnet.")
    parser.add_argument('--bs', type = int, default = 640,
                        help = 'the batch size of training and validation, int, such as 128, default is 640')
    parser.add_argument('--lr_up', type = float, default = 1e-5,
                        help = 'learning rate for pretrain bert, float, such as 1e-5, default is 1e-5')
    parser.add_argument('--lr_down', type = float, default = 1e-3,
                        help = 'learning rate for down prism, float, such as 1e-3, default is 1e-3')
    parser.add_argument('--lr_mid', type = float, default = 1e-3,
                        help = 'learning rate for middle layer, float, such as 1e-3, default is 1e-3')
    parser.add_argument('--pos_weight', type = float, default = 3,
                        help = 'pos_weight used for BCEwithlogitsloss, float, such as 3, default is 3')
    parser.add_argument('--random_seed', type = int, default=37,
                        help = 'the random seed, int, such as 37, default is 37')
    parser.add_argument('--whether_grad_norm_clip', type = int, default = 0,
                        help = 'whether use grad_norm_clip, 1 mean yes, 0 means no. default is 0.')
    parser.add_argument('--grad_norm_clip', type = float, default = 5,
                        help = 'must be specified if whether_grad_norm_clip == 1, means the value of grad norm clip, float, \
                        defalut is 5.0')
    parser.add_argument('--RBP_name_file', type = str,
                        help = 'the RBP and cell line name, such as "SUPV3L1_K562". one line one rbp, means train one by one')
    parser.add_argument('--data_dir', type = str, default=None,
                        help = 'the directory of h5 files.')
    parser.add_argument('--max_epoch', type = int, default = 200,
                        help = 'the max epoch for training, int, such as 200, default is 200')
    parser.add_argument('--tol_epoch', type = int, default = 20,
                        help = 'the tolerance epoch for early stop, int, such as 20, default is 20')
    parser.add_argument('--model_type', type=str, default='rnaverse')
    parser.add_argument('--fusion_mode', type=str, default='late', help='early or late, default late')
    parser.add_argument('--model_name_or_path', type = str, help = 'the model name or path for lm extractor')
    parser.add_argument('--save_model_dir', type=str, default=None, help = 'save model dir, default is None, means not save model')
    parser.add_argument('--output_file', type = str, default=None, help = 'output file to save rbp and its best auc, auprc, acc')
    parser.add_argument('--json_file', type = str, default=None, help = 'output json file to save per-rbp and mean train/val/test loss, acc, auc, auprc')

    args = parser.parse_args()
    return args

def loss_fn(output, label, pos_weight):
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    return loss(output, label)

def compute_acc(pred_:list, label_:list):
    # pred_ have experienced sigmoid
    pred_ = np.array(pred_)
    label_ = np.array(label_)
    pred_ = (pred_ > 0.5).astype(int)
    acc = (pred_ == label_).astype(int).sum() / len(pred_)
    return acc

def compute_auc(pred_:list, label_:list):
    # pred_ have experienced sigmoid
    pred_ = np.array(pred_)
    label_ = np.array(label_)
    fpr, tpr, thresholds = roc_curve(label_, pred_)
    auc_ = auc(fpr, tpr)
    return auc_

def compute_auprc(pred_:list, label_:list):
    # pred_ have experienced sigmoid
    pred_ = np.array(pred_)
    label_ = np.array(label_)
    precision, recall, thresholds = precision_recall_curve(label_, pred_)
    auprc = auc(recall, precision)
    return auprc


def train_rbp(rbp_name, args):
    tick1 = time.time()
    set_seed(args.random_seed)
    extractor, tokenizer, args = get_extractor(args)
    print('-'*20 + 'train for rbp: ' + rbp_name + '-'*20)

    rbp_data = os.path.join(args.data_dir, '%s.h5'%(rbp_name))
    if not os.path.exists(rbp_data):
        print('the rbp data file not exist:', rbp_data)
        return None
    train_loader = DataLoader(SeqicSHAPE(rbp_data), batch_size=args.bs, shuffle=True, collate_fn = lambda batch:tailor_batch_prism_opensource(batch, tokenizer, args), num_workers=2, drop_last=True)
    val_loader = DataLoader(SeqicSHAPE(rbp_data, is_val=True), batch_size=args.bs, shuffle=False, collate_fn = lambda batch:tailor_batch_prism_opensource(batch, tokenizer, args))
    test_loader = DataLoader(SeqicSHAPE(rbp_data, is_test=True), batch_size=args.bs, shuffle=False, collate_fn = lambda batch:tailor_batch_prism_opensource(batch, tokenizer, args))

    model = PrismNetLM(args, extractor, tokenizer)
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)

    optimizer_pretrain = torch.optim.Adam(model.extractor.parameters(), lr=args.lr_up)
    optimizer_down = torch.optim.Adam(model.downprism.parameters(), lr=args.lr_down, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = GradualWarmupScheduler(
                optimizer_down, multiplier=8, total_epoch=float(200), after_scheduler=None)
    
    # Build optimizer_mid based on whether model has shape_embed (for RNAverse-seq)
    mid_params = [
        {"params": model.dense.parameters(), "lr": args.lr_mid},
        {"params": model.LayerNorm.parameters(), "lr": args.lr_mid},
        {"params": model.transfer.parameters(), "lr": args.lr_mid},
    ]
    if hasattr(model, 'shape_embed'):
        mid_params.append({"params": model.shape_embed.parameters(), "lr": args.lr_mid})
    optimizer_mid = torch.optim.Adam(mid_params, lr=args.lr_mid)
    
    for epoch in range(args.max_epoch):
        print('-------epoch:%s-------' %(epoch))
        model.train()
        train_loss = 0.0

        train_pred = []
        train_label = []

        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = loss_fn(outputs, batch['label'], args.pos_weight)
            optimizer_pretrain.zero_grad()
            optimizer_mid.zero_grad()
            optimizer_down.zero_grad()
            loss.backward()
            if args.whether_grad_norm_clip == 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer_pretrain.step()
            optimizer_mid.step()
            optimizer_down.step()
            train_loss += loss.item()*batch['label'].shape[0]
            pred_result = torch.sigmoid(outputs)
            train_pred.extend(torch.flatten(pred_result).cpu().tolist())
            train_label.extend(torch.flatten(batch['label']).cpu().tolist())
        train_loss_ave =  train_loss/len(train_loader.dataset)
        train_acc = compute_acc(train_pred, train_label)
        train_auc = compute_auc(train_pred, train_label)
        train_auprc = compute_auprc(train_pred, train_label)
        print('Training/Loss:', train_loss_ave)
        print('Training/Accuracy:', train_acc)
        print('Training/AUC:', train_auc)
        print('Training/AUPRC:', train_auprc)

        model.eval()

        def evaluate(loader):
            eval_loss = 0
            eval_pred = []
            eval_label = []
            for batch in tqdm(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(batch)
                loss = loss_fn(outputs, batch['label'], args.pos_weight)
                eval_loss += loss.item()*batch['label'].shape[0]
                pred_result = torch.sigmoid(outputs)
                eval_pred.extend(torch.flatten(pred_result).cpu().tolist())
                eval_label.extend(torch.flatten(batch['label']).cpu().tolist())
            eval_loss_ave = eval_loss/len(loader.dataset)
            eval_acc = compute_acc(eval_pred, eval_label)
            eval_auc = compute_auc(eval_pred, eval_label)
            eval_auprc = compute_auprc(eval_pred, eval_label)
            return eval_loss_ave, eval_acc, eval_auc, eval_auprc

        val_loss_ave, val_acc, val_auc, val_auprc = evaluate(val_loader)
        test_loss_ave, test_acc, test_auc, test_auprc = evaluate(test_loader)
        print('Validation/Loss:', val_loss_ave)
        print('Validation/Accuracy:', val_acc)
        print('Validation/AUC:', val_auc)
        print('Validation/AUPRC:', val_auprc)
        print('Test/Loss:', test_loss_ave)
        print('Test/Accuracy:', test_acc)
        print('Test/AUC:', test_auc)
        print('Test/AUPRC:', test_auprc)
        scheduler.step(epoch+1)

        # early stop on validation auc
        if epoch == 0:
            best_auc = val_auc
            best_epoch = epoch
            best_metric = [train_loss_ave, train_acc, train_auc, train_auprc,
                           val_loss_ave, val_acc, val_auc, val_auprc,
                           test_loss_ave, test_acc, test_auc, test_auprc]
            if args.save_model_dir != None:
                if os.path.exists(args.save_model_dir) == False:
                    os.makedirs(args.save_model_dir)
                torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'best_model.pt'))
        else:
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                best_metric = [train_loss_ave, train_acc, train_auc, train_auprc,
                               val_loss_ave, val_acc, val_auc, val_auprc,
                               test_loss_ave, test_acc, test_auc, test_auprc]
                if args.save_model_dir != None:
                    if os.path.exists(args.save_model_dir) == False:
                        os.makedirs(args.save_model_dir)
                    torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'best_model.pt'))
            else:
                if epoch - best_epoch >= args.tol_epoch:
                    print('Early stop at epoch %s, the best epoch is %s' % (epoch, best_epoch))
                    break
    
    print('finish training after %s epochs'%(epoch+1))
    print('for RBP %s, best auc is:'%(rbp_name), best_auc, 'epoch is:', best_epoch)
    # best_metric: [train_loss, train_acc, train_auc, train_auprc, val_loss, val_acc, val_auc, val_auprc, test_loss, test_acc, test_auc, test_auprc]
    print('the best epoch train loss, acc, auc, auprc are:', best_metric[0], best_metric[1], best_metric[2], best_metric[3])
    print('the best epoch val loss, acc, auc, auprc are:', best_metric[4], best_metric[5], best_metric[6], best_metric[7])
    print('the best epoch test loss, acc, auc, auprc are:', best_metric[8], best_metric[9], best_metric[10], best_metric[11])

    if args.output_file != None:
        out_dir = os.path.dirname(args.output_file)
        if os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        with open(args.output_file, 'a') as f:
            f.write(rbp_name + '\t' + str(best_metric[10]) + '\t' + str(best_metric[11]) + '\t' + str(best_metric[9]) + '\n')

    tick2 = time.time()
    print('cost time for training rbp %s:'%(rbp_name), tick2 - tick1)
    return best_metric
            
def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))

    tick1 = time.time()

    fo1 = open(args.RBP_name_file, 'r')
    fo1all = fo1.readlines()
    fo1.close()

    RBP_name_list = [x.rstrip('\n') for x in fo1all]

    dict_result = {}
    for rbp_name in RBP_name_list:
        best_metric = train_rbp(rbp_name, args)
        if best_metric is not None:
            dict_result[rbp_name] = {
                'train': {'loss': best_metric[0], 'acc': best_metric[1], 'auc': best_metric[2], 'auprc': best_metric[3]},
                'val': {'loss': best_metric[4], 'acc': best_metric[5], 'auc': best_metric[6], 'auprc': best_metric[7]},
                'test': {'loss': best_metric[8], 'acc': best_metric[9], 'auc': best_metric[10], 'auprc': best_metric[11]},
            }
        time.sleep(5)

    if args.json_file != None:
        import json
        metric_names = ['loss', 'acc', 'auc', 'auprc']
        mean_result = {}
        for split in ['train', 'val', 'test']:
            mean_result[split] = {m: float(np.mean([dict_result[r][split][m] for r in dict_result])) for m in metric_names}
        dict_result['mean'] = mean_result
        out_dir = os.path.dirname(args.json_file)
        if out_dir != '' and os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        with open(args.json_file, 'w') as f:
            json.dump(dict_result, f, indent=2, ensure_ascii=False)
        print('json result saved to %s' % args.json_file)

    tick_all = time.time()
    print('all cost time:', tick_all - tick1)
    print('done!')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)