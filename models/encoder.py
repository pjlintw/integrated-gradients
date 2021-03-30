     
from .transformer_blocks import positional_encoding, MultiHeadAttention, FeedForwardBlock

import torch
import torch.nn as nn

class TransformerEncLayer(nn.Module):
    def __init__(self, d_model, num_head, intermediate_dim, rate=0.1):
        super(TransformerEncLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, d_model, num_head)

        self.ffn = FeedForwardBlock(d_model, intermediate_dim)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(p=rate)
        self.dropout2 = nn.Dropout(p=rate)
        
    def forward(self, x, training, mask):
        
        # (batch_size, src_seq_len, d_model)
        attn_out, attn_weight  = self.mha(x,x,x,mask)
        if training is True:
            atten_out = self.dropout1(attn_out)
        out1 = self.layernorm1(x + attn_out)
        
        ffn_out = self.ffn(out1)
        if training is True:
            ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_head,
                 intermediate_dim, 
                 input_vocab_size,
                 max_src_len,
                 rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(max_src_len, self.d_model)

        self.enc_layers = [TransformerEncLayer(d_model, 
                                               num_head, 
                                               intermediate_dim,
                                               rate) for _ in range(num_layers)]
        self.enc_layers = torch.nn.ModuleList(self.enc_layers)
        self.dropout = nn.Dropout(p=rate)


    def forward(self, src, training, mask, gpu=False):
        """Transformer encoder.

        Args:
          scr: input with shape (batch_size, seq_len, d_model)
        """
        x = src #.cuda()
        #print("shape", x.shape)
        seq_len = x.shape[1]    
        #print("source length", seq_len)
        x = torch.mul(x, (self.d_model**(1/2)))
        #print(x.shape)
        if gpu:
            x += self.pos_encoding[:, :seq_len, :].cuda()
        else:
            x += self.pos_encoding[:, :seq_len, :]


        if training is True:
            x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x



if __name__ == "__main__":
    enc = TransformerEncoder(num_layers=2, d_model=512, num_head=8,
                             intermediate_dim=2048, input_vocab_size=8500,
                             max_src_len=10000)
    inp = torch.rand((64, 62,512))  
    print(inp.shape)

    o = enc(inp, training=False, mask=None)
    print(o.shape)
