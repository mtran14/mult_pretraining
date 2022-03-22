import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from dataloader import ContinuousAVDataset
from model import CrossAttentionAVTransformer, init_weights
import os
import json
from optimizers import BertAdam, WarmupLinearSchedule
import wandb
# =======================================
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
# =======================================

def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


config_path = "../config/config_continuous.json"
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
output_name = config['output_name']
p_mask = 0.15
wandb.init(project="avc", config=config)

print('loading dataset...')
metadata_path_train, metadata_path_val = "../../metadata_seqlen100_F.csv", "../../metadata_seqlen100_F_val.csv"
train_dataset = ContinuousAVDataset(metadata_path_train, seq_len)
val_dataset = ContinuousAVDataset(metadata_path_val, seq_len)

kwargs = {'num_workers': n_workers, 'shuffle': True,
          'drop_last': True, 'pin_memory': True, 'batch_size': batch_size}
train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

print('initializing model...')
model = CrossAttentionAVTransformer(n_code, n_heads, embed_size, None, None, None, None, None)
model.to(device)

n_iteration = epochs * len(train_loader)  # 10 epochs
print('initializing optimizer and loss...')

optimizer = BertAdam(model.parameters(), lr=1e-4, warmup=warmup_portion,
                     t_total=n_iteration, schedule='warmup_linear')

bestDevLoss = float('inf')
criterion = nn.L1Loss()

print("Done init, start training ...")
model.train()
print_each, eval_every, save_every = 50, 1000, 1000
batch_iter = iter(train_loader)

it = 0
#for it in range(n_iteration):
    ## get batch
    #try:
        #batch, batch_iter = get_batch(train_loader, batch_iter)
    #except:
        #continue
print(len(train_loader), len(train_dataset))
for epoch in range(epochs):
    for batch_step, batch in enumerate(train_loader):
        masked_input_a = batch['input_audio']
        masked_input_v = batch['input_video']
        masked_target_a = batch['target_audio']
        masked_target_v = batch['target_video']
        mask_bool = batch['mask'].cuda(non_blocking=True)
    
        masked_input_a = masked_input_a.cuda(non_blocking=True)
        masked_input_v = masked_input_v.cuda(non_blocking=True)
        masked_target_a = masked_target_a.cuda(non_blocking=True)
        masked_target_v = masked_target_v.cuda(non_blocking=True)
        
        output =  model(masked_input_a, masked_input_v)
    
        output_a, output_v = output
        output_a = output_a.view(-1, output_a.size(-1))
        output_v = output_v.view(-1, output_v.size(-1))
        target_a = masked_target_a.view(-1, masked_target_a.size(-1))
        target_v = masked_target_v.view(-1, masked_target_v.size(-1))
        mask_bool = mask_bool.view(-1, )
        
        loss_a = criterion(output_a[mask_bool], target_a[mask_bool])
        loss_v = criterion(output_v[mask_bool], target_v[mask_bool])
        loss = loss_a + loss_v
        
        #print('Sample pred: ', output_v[mask_bool][0])
        # compute gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # apply gradients
        optimizer.step()    
        
        if it % print_each == 0:
            print('it:', it,
                  ' | loss', np.round(loss.item(), 3))
            wandb.log({'Train_loss': loss})
    
        # reset gradients
        optimizer.zero_grad()
        
        if it % eval_every == 0:
            model.eval()
            loss_list = []
            with torch.no_grad():
                for batch_step, batch_data in enumerate(val_loader):
                    masked_input_a = batch_data['input_audio']
                    masked_input_v = batch_data['input_video']
                    masked_target_a = batch_data['target_audio']
                    masked_target_v = batch_data['target_video']
                    mask_bool = batch_data['mask'].cuda(non_blocking=True)
                
                    masked_input_a = masked_input_a.cuda(non_blocking=True)
                    masked_input_v = masked_input_v.cuda(non_blocking=True)
                    masked_target_a = masked_target_a.cuda(non_blocking=True)
                    masked_target_v = masked_target_v.cuda(non_blocking=True)
                    
                    output =  model(masked_input_a, masked_input_v)
                
                    output_a, output_v = output
                    output_a = output_a.view(-1, output_a.size(-1))
                    output_v = output_v.view(-1, output_v.size(-1))
                    target_a = masked_target_a.view(-1, masked_target_a.size(-1))
                    target_v = masked_target_v.view(-1, masked_target_v.size(-1))
                    mask_bool = mask_bool.view(-1, )
                    
                    loss_a = criterion(output_a[mask_bool], target_a[mask_bool])
                    loss_v = criterion(output_v[mask_bool], target_v[mask_bool])
                    loss = loss_a + loss_v
                    
                    loss_list.append(loss.item())
            eval_loss = np.mean(loss_list)
            #if(eval_loss < bestDevLoss):
            torch.save(model.state_dict(), output_name)
                #bestDevLoss = eval_loss            
            print('Eval on devset:',
                  ' | loss', np.round(eval_loss, 3))
            wandb.log({'Val_loss': eval_loss})
            model.train()
        it += 1
            
        
        
    
