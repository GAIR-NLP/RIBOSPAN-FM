import torch
import numpy as np
import h5py
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_rnaverse import RNAverseConfig
from ..modeling_rnaverse import RNAverseModel
from ..tokenization_rnaverse import RNAverseTokenizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau



def get_extractor(args):
    """
    Load RNAverse model and tokenizer.
    
    Args:
        args: Arguments containing model configuration
            - model_type: 'rnaverse' or 'rnaverse-seq'
            - model_name_or_path: Path to pretrained model
    
    Returns:
        extractor: RNAverse model
        tokenizer: RNAverse tokenizer
        args: Updated arguments
    """
    args.train_from_scratch = False
    args.cache_dir = None
    
    # Set token type for RNAverse (single token)
    args.token_type = 'single'
    
    # Set max model length for RNAverse
    args.model_max_length = 1025
    
    # Determine if it's seq-only model
    is_seq_only = (args.model_type == 'rnaverse-seq')
    
    print(f'Loading {args.model_type} model')
    
    # Load config
    config = RNAverseConfig.from_pretrained(
        args.model_name_or_path,
        is_seq_only_model=is_seq_only,
    )
    
    # Load tokenizer
    tokenizer = RNAverseTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    
    # Load model
    extractor = RNAverseModel.from_pretrained(
        args.model_name_or_path,
    )
    
    print(tokenizer)
    return extractor, tokenizer, args


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class SeqicSHAPE(torch.utils.data.Dataset):
    def __init__(self, data_path, select_path=None, is_test=False, is_val=False, is_infer=False, use_structure=True):
        """data loader
        
        Args:
            data_path ([str]): h5 file path
            is_test (bool, optional): testset or not. Defaults to False.
        """
        if is_infer:
            self.dataset = self.__load_infer_data__(data_path, use_structure=use_structure)
            print("infer data: ", self.__len__()," use_structure: ", use_structure)
        else:
            dataset = h5py.File(data_path, 'r')
            X_train = np.array(dataset['X_train']).astype(np.float32)
            Y_train = np.array(dataset['Y_train']).astype(np.int32)
            X_test  = np.array(dataset['X_test']).astype(np.float32)
            Y_test  = np.array(dataset['Y_test']).astype(np.int32)
            if 'X_val' in dataset:
                X_val = np.array(dataset['X_val']).astype(np.float32)
                Y_val = np.array(dataset['Y_val']).astype(np.int32)
            else:
                X_val = None
                Y_val = None
            if len(Y_train.shape) == 1:
                Y_train = np.expand_dims(Y_train, axis=1)
                Y_test  = np.expand_dims(Y_test, axis=1)
                if Y_val is not None:
                    Y_val = np.expand_dims(Y_val, axis=1)
            X_train = np.expand_dims(X_train, axis=3).transpose([0, 3, 2, 1])
            X_test  = np.expand_dims(X_test,  axis=3).transpose([0, 3, 2, 1])
            if X_val is not None:
                X_val = np.expand_dims(X_val, axis=3).transpose([0, 3, 2, 1])

            train = {'inputs': X_train, 'targets': Y_train}
            test  = {'inputs': X_test,  'targets': Y_test}
            val   = {'inputs': X_val,   'targets': Y_val}

            labels, nums = np.unique(Y_train,return_counts=True)
            # print("train:", labels, nums)
            labels, nums = np.unique(Y_test,return_counts=True)
            # print("test:", labels, nums)

            train = self.__prepare_data__(train)
            test  = self.__prepare_data__(test)
            if X_val is not None:
                val = self.__prepare_data__(val)

            if is_val:
                if val['inputs'] is None:
                    raise ValueError('the h5 file %s has no X_val' % data_path)
                self.dataset = val
            elif is_test:
                if select_path != None:
                    index_list = []
                    fo1 = open(select_path, 'r')
                    fo1all = fo1.readlines()
                    fo1.close()

                    for i1,x1 in enumerate(fo1all):
                        if x1.startswith('>'):
                            tmp_list = x1.strip().split('_')
                            index_ = int(tmp_list[-2])
                            index_list.append(index_)
                    # bs * 1 * 101 * 5
                    # bs * 1
                    test['inputs'] = test['inputs'][index_list, :, :, :]
                    test['targets'] = test['targets'][index_list, :]

                self.dataset = test
            else:
                if select_path != None:
                    index_list = []
                    fo1 = open(select_path, 'r')
                    fo1all = fo1.readlines()
                    fo1.close()

                    for i1,x1 in enumerate(fo1all):
                        if x1.startswith('>'):
                            tmp_list = x1.strip().split('_')
                            index_ = int(tmp_list[-2])
                            index_list.append(index_)
                    # bs * 1 * 101 * 5
                    # bs * 1
                    train['inputs'] = train['inputs'][index_list, :, :, :]
                    train['targets'] = train['targets'][index_list, :]
                self.dataset = train
    
    def __prepare_data__(self, data):
        inputs    = data['inputs'][:,:,:,:4]
        structure = data['inputs'][:,:,:,4:]
        structure = np.expand_dims(structure[:,:,:,0], axis=3)
        inputs    = np.concatenate([inputs, structure], axis=3)
        data['inputs']  = inputs
        return data

    def __to_sequence__(self, x):
        x1 = np.zeros_like(x[0,:,:1])
        for i in range(x1.shape[0]):
            # import pdb; pdb.set_trace()
            x1[i] = np.argmax(x[0,i,:4])
            # import pdb; pdb.set_trace()
        return x1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.dataset['inputs'][index]
        # x = self.__to_sequence__(x)
        y = self.dataset['targets'][index]
        return x, y


    def __len__(self):
        return len(self.dataset['inputs'])
    

class ResidualBlock1D(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.c1 = nn.Conv1d(planes,   planes,   kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes,   planes*2, kernel_size=11, stride=1,
                     padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes*2)
        self.c3 = nn.Conv1d(planes*2, planes*8, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes,   planes*8,   kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes*8),
        )
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock2D(nn.Module):

    def __init__(self, planes, kernel_size=(11,5), padding=(5,2), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes,   planes,   kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes,   planes*2, kernel_size=kernel_size, stride=1,
                     padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes*2)
        self.c3 = nn.Conv2d(planes*2, planes*4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes,   planes*4,   kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes*4),
        )
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
        
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class PrismNet(nn.Module):
    def __init__(self, mode="pu", n_features=768):
        super(PrismNet, self).__init__()
        self.mode = mode
        h_p, h_k = 2, 5 
        if mode=="pu":
            self.n_features = 5
        elif mode=="seq":
            self.n_features = 4
            h_p, h_k = 1, 3 
        elif mode=="str":
            self.n_features = 1
            h_p, h_k = 0, 1
        elif mode == 'custom_lm':
            self.n_features = n_features
            h_p, h_k = 2, 5
        else:
            raise "mode error"
        
        base_channel = 8
        self.conv    = Conv2d(1, base_channel, kernel_size=(11, h_k), bn = True, same_padding=True)
        self.se      = SEBlock(base_channel)
        self.res2d   = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p)) 
        self.res1d   = ResidualBlock1D(base_channel*4) 
        self.avgpool = nn.AvgPool2d((1,self.n_features))
        self.gpool   = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(base_channel*4*8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        """[forward]
        
        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode=="seq":
            input = input[:,:,:,:4]
        elif self.mode=="str":
            input = input[:,:,:,4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x*z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x


def prism2seq(seq_array):
    # shape is (101,4)
    seq = ''
    for x1 in seq_array:
        if x1.sum() == 0:
            seq += 'N'
        else:
            seq += ['A', 'C', 'G', 'T'][x1.argmax().item()]
    return seq


def shape_bin(shape_num):
    # shape_num is a float, -1 means null
    # 0-1 is cut into 10 equal bins: (0,0.1]->1, (0.1,0.2]->2, ..., (0.9,1.0]->10
    if shape_num == -1:
        return 0
    elif shape_num < 0:
        return 0
    else:
        bin_id = min(int(shape_num * 10) + 1, 10)
        return bin_id


def tailor_batch_prism_opensource(batch_data, tokenizer, args):
    """
    Prepare batch data for RNAverse model.
    
    Args:
        batch_data: List of (input, target) tuples
        tokenizer: RNAverse tokenizer
        args: Arguments containing model configuration
    
    Returns:
        data_dict: Dictionary containing batched data
    """
    target_list = []
    four_list = []
    raw_str_batch = []
    shape_value_list = []
    
    for input, target in batch_data:
        seq_ = input[:, :, :4].squeeze(0) # shape is (101,4)
        four_list.append(torch.from_numpy(seq_))
        seq_str = prism2seq(seq_)
        target_list.append(torch.from_numpy(target))
        shape_value_ = input[:,:,4] # shape is (1, 101)

        shape_list = [float(x1) for x1 in shape_value_.squeeze(0)]
        shape_value_lisan = torch.tensor([shape_bin(y) for y in shape_list]) # size is 101
        shape_tech = 'icSHAPE'
        shape_value_list.append(shape_value_lisan)
        # Prepare raw_str_batch for RNAverse tokenizer
        raw_str_batch.append({
            'seq': seq_str,
            'struct_score_list': shape_list,
            'probe_method': shape_tech,
        })
    
    # Tokenize sequences using RNAverse tokenizer
    data_dict = tokenizer(raw_str_batch,
                    padding='longest',
                    max_length=args.model_max_length,
                    truncation=True,
                    return_tensors='pt')
    
    # Calculate weight_mask for CLS token only (RNAverse has no SEP token)
    max_len = data_dict['input_ids'].shape[1]
    weight_mask = torch.ones((len(target_list), max_len)) #including [cls], dim= [bz, max_len]
    
    data_dict['label'] = torch.cat([x1.unsqueeze(0) for x1 in target_list], 0).float()
    data_dict['four_id'] = torch.cat([x1.unsqueeze(0) for x1 in four_list], 0).float()
    data_dict['weight_mask'] = weight_mask
    data_dict['post_token_length'] = torch.zeros(data_dict['attention_mask'].shape)
    data_dict['shape_value'] = torch.cat([x1.unsqueeze(0) for x1 in shape_value_list], 0).long()
    
    return data_dict


class PrismNetLM(nn.Module):
    def __init__(self, args, extractor, tokenizer):
        super(PrismNetLM, self).__init__()
        self.extractor = extractor
        self.config = extractor.config
        self.tokenizer = tokenizer
        self.args = args
        self.fusion_mode = getattr(args, 'fusion_mode', 'late')

        # RNAverse uses hidden_size from config
        embed_dim = self.config.hidden_size

        self.dense = torch.nn.Linear(embed_dim, embed_dim)
        self.transform_act_fn = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(embed_dim, eps=1e-5)
        self.transfer = torch.nn.Linear(embed_dim, embed_dim)

        self.downprism = PrismNet(mode='custom_lm', n_features=embed_dim)
        if self.config.is_seq_only_model:
            self.shape_embed = torch.nn.Embedding(11, embed_dim)


    def forward(self,x):
       
        # Get embeddings from RNAverse model
        input_ids = x['input_ids']
        input_struct_tokens = x.get('input_struct_tokens', None)
        input_probe_method_tokens = x.get('input_probe_method_tokens', None)
        input_func_tokens = x.get('input_func_tokens', None)
        attention_mask = x['attention_mask']
        
        if self.config.is_seq_only_model and self.fusion_mode == 'early':
            # early fusion for rnaverse-seq: add struct embedding into the input embeddings
            seq_embeds = self.extractor.get_input_embeddings()(input_ids)
            struct_embeds = self.shape_embed(x['shape_value'])
            inputs_embeds = seq_embeds.clone()
            slen = struct_embeds.shape[1]
            inputs_embeds[:, 1:slen+1, :] = inputs_embeds[:, 1:slen+1, :] + struct_embeds
            output = self.extractor(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
        else:
            output = self.extractor(
                input_ids=input_ids,
                input_struct_tokens=input_struct_tokens,
                input_probe_method_tokens=input_probe_method_tokens,
                input_func_tokens=input_func_tokens,
                attention_mask=attention_mask
            )
        
        # RNAverse returns last_hidden_state
        hidden_states = output.last_hidden_state

        weight_mask = x['weight_mask'] #[bz,ori_max_len] for CLS only
        post_token_length = x['post_token_length']
        
        # RNAverse uses single token, so no k-mer mapping needed
        if  weight_mask.shape!=post_token_length.shape:
            print("weight_mask shape: ", weight_mask.shape)
            print("post_token_length shape: ", post_token_length.shape)
            raise ValueError("weight_mask and post_token_length should have the same shape")
        mapping_hidden_states = hidden_states
        
        hidden_states = mapping_hidden_states * weight_mask.unsqueeze(2)
        # RNAverse only has CLS token, no SEP token, so remove only CLS (index 0)
        hidden_states = hidden_states[:, 1:, :]
        if self.config.is_seq_only_model and self.fusion_mode == 'late':
            shape_embed = self.shape_embed(x['shape_value'])
            hidden_states = hidden_states + shape_embed
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.transfer(hidden_states)
        hidden_states = torch.unsqueeze(hidden_states, 1)
        hidden_states = self.downprism(hidden_states)
        return hidden_states
