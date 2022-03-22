#!/home/mtran/anaconda3/bin/python
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torch
import math
from transformer import TransformerEncoder
from multiprocessing import Pool, Process, Manager

# =============================================================================
# Transformer
# =============================================================================
def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])

    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

#        self.q_linear = nn.Linear(out_dim, out_dim)
#        self.k_linear = nn.Linear(out_dim, out_dim)
#        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y

        qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L

        #break into n_heads
        q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD

        #n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE

        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(inner_transformer_size, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:,:x.size(1), :] #x.size(1) = seq_len
    


class Transformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, input_dim, dropout=.1):
        super().__init__()

        #model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.projection = nn.Linear(input_dim, embed_size)
        self.input_dim = input_dim
        self.pe = PositionalEmbedding(embed_size, seq_len)

        #backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        #language model
        self.norm = nn.LayerNorm(embed_size, eps=1e-12)
        if(input_dim == 1):
            self.linear = nn.Linear(embed_size, n_embeddings, bias=False)
        else:
            self.linear = nn.Linear(embed_size, input_dim)



    def forward(self, x):
        if(self.input_dim == 1):
            x = self.embeddings(x)
        else:
            x = self.projection(x)
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)

        x = self.norm(x)
        x = self.linear(x)
        return x

    def extract_feature(self, x):
        #take output of encoder
        if(self.input_dim == 1):
            x = self.embeddings(x)
        else:
            x = self.projection(x)
        # print(x.size(), self.pe(x).size())
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x

    # def finetune(self, x):
    #     if(self.input_dim == 1):
    #         x = self.embeddings(x)
    #     else:
    #         x = self.projection(x)
    #     # print(x.size(), self.pe(x).size())
    #     x = x + self.pe(x)
    #     for encoder in self.encoders:
    #         x = encoder(x)
    #     x = torch.mean(x, axis=1) #x: BxTxD
    #     logits = self.final_linear(x)
    #     return logits

class AVTransformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, input_dim, dropout=.1):
        super().__init__()

        #model input
        self.embeddings_a = nn.Embedding(n_embeddings, embed_size)
        self.embeddings_v = nn.Embedding(n_embeddings, embed_size)
        self.projection = nn.Linear(input_dim, embed_size)
        self.input_dim = input_dim
        self.pe_a = PositionalEmbedding(embed_size, seq_len)
        self.pe_v = PositionalEmbedding(embed_size, seq_len)

        #backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size*2, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        #language model
        self.norm = nn.LayerNorm(embed_size*2, eps=1e-12)
        if(input_dim == 1):
            self.linear_a = nn.Linear(embed_size*2, n_embeddings)
            self.linear_v = nn.Linear(embed_size*2, n_embeddings)
        else:
            self.linear = nn.Linear(embed_size, input_dim)

    def forward(self, x_a, x_v):
        if(self.input_dim == 1):
            x_a = self.embeddings_a(x_a)
            x_v = self.embeddings_v(x_v)
        # else:
        #     x = self.projection(x)
        x_a = x_a + self.pe_a(x_a)
        x_v = x_v + self.pe_a(x_v)
        
        xav = torch.cat((x_a, x_v), dim=-1)
        for encoder in self.encoders:
            xav = encoder(xav)

        xav = self.norm(xav)
        x_a = self.linear_a(xav)
        x_v = self.linear_v(xav)
        return x_a, x_v
    
    def extract_feature(self, x_a, x_v):
        if(self.input_dim == 1):
            x_a = self.embeddings_a(x_a)
            x_v = self.embeddings_v(x_v)
        # else:
        #     x = self.projection(x)
        x_a = x_a + self.pe_a(x_a)
        x_v = x_v + self.pe_a(x_v)

        xav = torch.cat((x_a, x_v), dim=-1)
        for encoder in self.encoders:
            xav = encoder(xav)
            
        return xav    

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        m.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)
    if isinstance(m, nn.Linear) and m.bias is not None:
        m.bias.data.zero_()    
        
class AVTransformerNew(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, input_dim, dropout=.1):
        super().__init__()

        #model input
        self.embeddings_a = nn.Embedding(n_embeddings, embed_size)
        self.embeddings_v = nn.Embedding(n_embeddings, embed_size)
        self.input_dim = input_dim

        #backbone
        self.transformer = TransformerEncoder(embed_dim=embed_size*2,
                                              num_heads=n_heads,
                                              layers=n_code,
                                              attn_dropout=0.0,
                                              relu_dropout=0.1,
                                              res_dropout=0.1,
                                              embed_dropout=0.25,
                                              attn_mask=True)
        
        if(input_dim == 1):
            self.linear_a = nn.Linear(embed_size*2, n_embeddings)
            self.linear_v = nn.Linear(embed_size*2, n_embeddings)
        else:
            self.linear = nn.Linear(embed_size, input_dim)

    def forward(self, x_a, x_v):
        if(self.input_dim == 1):
            x_a = self.embeddings_a(x_a)
            x_v = self.embeddings_v(x_v)
        
        xav = torch.cat((x_a, x_v), dim=-1)
        xav = self.transformer(xav)
        x_a = self.linear_a(xav)
        x_v = self.linear_v(xav)
        return x_a, x_v  
    
    
class CrossAttentionAVTransformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, input_dim, dropout):
        super(CrossAttentionAVTransformer, self).__init__()
        self.orig_d_a, self.orig_d_v = 512, 17
        self.d_a, self.d_v = embed_size, embed_size
        self.num_heads = n_heads
        self.layers = n_code
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0
        self.embed_dropout = 0.25
        self.attn_mask = True   
        combined_dim = self.d_a + self.d_v
        
        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)    
        
        # 2. Crossmodal Attentions
        self.acoustic_cross_visual = self.get_network(embed_dim=self.d_a, attn_dropout=self.attn_dropout_a) 
        self.visual_cross_acoustic = self.get_network(embed_dim=self.d_v, attn_dropout=self.attn_dropout_v)
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_mem = self.get_network(embed_dim=combined_dim, attn_dropout=self.attn_dropout)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)   
        
        self.proj1_a = nn.Linear(combined_dim, combined_dim)
        self.proj2_a = nn.Linear(combined_dim, combined_dim)   
        
        self.proj1_v = nn.Linear(combined_dim, combined_dim)
        self.proj2_v = nn.Linear(combined_dim, combined_dim)           
        self.out_layer_a = nn.Linear(combined_dim, self.orig_d_a)
        self.out_layer_v = nn.Linear(combined_dim, self.orig_d_v)
            
    def get_network(self, embed_dim, attn_dropout, layers=-1):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers if layers > 0 else self.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)        
        
    def forward(self, x_a, x_v):
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
    
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        v_cross_a = self.visual_cross_acoustic(proj_x_v, proj_x_a, proj_x_a)    
        a_cross_v = self.acoustic_cross_visual(proj_x_a, proj_x_v, proj_x_v)    
        h_av = torch.cat([v_cross_a, a_cross_v], dim=2)
        h_av = self.trans_mem(h_av).permute(1,0,2)
        
        # A residual block
        h_proj_a = self.proj2_a(F.dropout(F.relu(self.proj1_a(h_av)), p=self.out_dropout, training=self.training))
        h_proj_a += h_av 
        
        h_proj_v = self.proj2_v(F.dropout(F.relu(self.proj1_v(h_av)), p=self.out_dropout, training=self.training))
        h_proj_v += h_av          
        
        output_a = self.out_layer_a(h_proj_a)
        output_v = self.out_layer_v(h_proj_v)
        return output_a, output_v
    
    def extract_feature(self, x_a, x_v):
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
    
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        v_cross_a = self.visual_cross_acoustic(proj_x_v, proj_x_a, proj_x_a)    
        a_cross_v = self.acoustic_cross_visual(proj_x_a, proj_x_v, proj_x_v)    
        h_av = torch.cat([v_cross_a, a_cross_v], dim=2)
        h_av = self.trans_mem(h_av)
        h_av_last = h_av[-1]
        return h_av, h_av_last    
    
class CrossAttentionAVTransformerDiscrete(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, input_dim, dropout):
        super(CrossAttentionAVTransformerDiscrete, self).__init__()
        self.embeddings_a = nn.Embedding(n_embeddings, embed_size)
        self.embeddings_v = nn.Embedding(n_embeddings, embed_size)
        
        self.orig_d_a, self.orig_d_v = embed_size, embed_size
        self.d_a, self.d_v = embed_size, embed_size
        self.num_heads = n_heads
        self.layers = n_code
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0
        self.attn_dropout_v = 0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0
        self.embed_dropout = 0.25
        self.attn_mask = True   
        combined_dim = self.d_a + self.d_v
        
        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)    
        
        # 2. Crossmodal Attentions
        self.acoustic_cross_visual = self.get_network(embed_dim=self.d_a, attn_dropout=self.attn_dropout_a) 
        self.visual_cross_acoustic = self.get_network(embed_dim=self.d_v, attn_dropout=self.attn_dropout_v)
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_mem = self.get_network(embed_dim=combined_dim, attn_dropout=self.attn_dropout, layers=3)
                
        self.out_layer_a = nn.Linear(combined_dim, n_embeddings)
        self.out_layer_v = nn.Linear(combined_dim, n_embeddings)
            
    def get_network(self, embed_dim, attn_dropout, layers=-1):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers if layers > 0 else self.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)        
        
    def forward(self, x_a, x_v):
        x_a = self.embeddings_a(x_a)
        x_v = self.embeddings_v(x_v)
        
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
    
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        v_cross_a = self.visual_cross_acoustic(proj_x_v, proj_x_a, proj_x_a)    
        a_cross_v = self.acoustic_cross_visual(proj_x_a, proj_x_v, proj_x_v)    
        h_av = torch.cat([v_cross_a, a_cross_v], dim=2)
        h_av = self.trans_mem(h_av).permute(1,0,2)
        
        output_a = self.out_layer_a(h_av)
        output_v = self.out_layer_v(h_av)
        return output_a, output_v
    
    def extract_feature(self, x_a, x_v):
        x_a = self.embeddings_a(x_a)
        x_v = self.embeddings_v(x_v)
        
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
    
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        v_cross_a = self.visual_cross_acoustic(proj_x_v, proj_x_a, proj_x_a)    
        a_cross_v = self.acoustic_cross_visual(proj_x_a, proj_x_v, proj_x_v)    
        h_av = torch.cat([v_cross_a, a_cross_v], dim=2)
        h_av = self.trans_mem(h_av).permute(1,0,2)

        return h_av    
        
if __name__ == '__main__':
    #testing transformer
    device = 'cpu'
    model = CrossAttentionAVTransformer(3, 6, 30, None, None, None, None, None)
    model.to(device)
    # x = torch.randint(0, 20000, (128, 20,)).to(device)
    #x_a, x_v = torch.randint(0, 200, (8, 100)), torch.randint(0, 200, (8, 100))
    x_a, x_v = torch.randn(8, 100, 512), torch.randn(8, 100, 17) #must be same dimensional seq_len
    x_aout, x_vout = model(x_a, x_v)
    print(x_aout.size(), x_vout.size())
