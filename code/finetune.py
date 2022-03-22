from turtle import down
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import CrossAttentionAVTransformer
import torch.nn.functional as F
import torch.optim as optim
import json
import torch

class mlp(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.downstream_model = nn.Sequential(
                                    nn.Linear(input_dim, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, n_class)
                                )
    
    def forward(self, x):
        return self.downstream_model(x)

if __name__ == '__main__':
    # load model from checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = 'base'

    # base configurations
    if(model_type=='base'):
        config_path = "../config/base.json"
        with open(config_path) as json_file:
            config = json.load(json_file)
        print('initializing..')
        batch_size = config['batch_size']
        #seq_len = config['seq_len']
        seq_len = 100
        embed_size = config['embed_size']
        n_heads = config['n_heads']
        n_code = config['n_code']
        #n_workers = config['n_workers']
        n_workers = 0
        device = config['device']
        epochs = config['epochs']
        warmup_portion = config['warmup']
        output_name = "../saved_models/base.pt"
    else:
        # Large configurations
        config_path = "../config/large.json"
        with open(config_path) as json_file:
            config = json.load(json_file)
        print('initializing..')
        batch_size = config['batch_size']
        #seq_len = config['seq_len']
        seq_len = 100
        embed_size = config['embed_size']
        n_heads = config['n_heads']
        n_code = config['n_code']
        #n_workers = config['n_workers']
        n_workers = 0
        device = config['device']
        epochs = config['epochs']
        warmup_portion = config['warmup']
        output_name = "../saved_models/large.pt"


    model = CrossAttentionAVTransformer(n_code, n_heads, embed_size, None, None, None, None, None)
    model.load_state_dict(torch.load(output_name))
    model = model.to(device)
    
    # this line is for feature_extraction only (no tuning on upstream model)
    #for p in model.parameters():  p.requires_grad=False 

    downstream_model = mlp(embed_size*2, n_class=6) # 6 class classification
    downstream_model = downstream_model.to(device)

    # x_a, x_v: BxTxD (the temporal dimension must be similar), sampling at 5fps
    x_a, x_v = torch.randn(8, 150, 512).to(device), torch.randn(8, 150, 17).to(device)
    labels = torch.randint(0, 6, (8,)).to(device)

    h_av, h_av_last = model.extract_feature(x_a, x_v)

    predictions = downstream_model(h_av_last)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    print(loss.item())
    loss.backward()
    
    