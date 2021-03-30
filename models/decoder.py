
from .transformer_blocks import positional_encoding, MultiHeadAttention, FeedForwardBlock


import torch
import torch.nn as nn

class TransformerDecLayer(nn.Module):
    def __init__(self, d_model, num_head, intermediate_dim, rate=0.1):
        super(TransformerDecLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, d_model, num_head)
        self.mha2 = MultiHeadAttention(d_model, d_model, num_head)

        self.ffn = FeedForwardBlock(d_model, intermediate_dim)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(p=rate)
        self.dropout2 = nn.Dropout(p=rate)
        self.dropout3 = nn.Dropout(p=rate)


    def forward(self, x, enc_output, training,
                  look_ahead_mask, padding_mask):
        # enc_output: (batch_size, src_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x,x,x, look_ahead_mask)
        if training is True:
            attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(k=enc_output, v=enc_output, q=out1, mask=padding_mask)
        if training is True:
            attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        

        ffn_out = self.ffn(out2)
        if training is True:
            ffn_out = self.dropout3(ffn_out)
        out3 = self.layernorm3(ffn_out + out2)

        return out3, attn_weights_block1, attn_weights_block2


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_head,
                 intermediate_dim, 
                 target_vocab_size,
                 max_tgt_len,
                 shared_emb_layer=None,
                 rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if shared_emb_layer is None or shared_emb_layer is False:
            self.embedding = nn.Embedding(target_vocab_size, d_model)
            
        else:
            self.embedding = shared_emb_layer
          
        self.pos_encoding = positional_encoding(max_tgt_len, d_model)

        self.dec_layers = [TransformerDecLayer(d_model,
                                    num_head,
                                    intermediate_dim,
                                    rate) for _ in range(num_layers)]
        self.dec_layers = torch.nn.ModuleList(self.dec_layers)
        self.dropout = nn.Dropout(p=rate)


    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask, gpu=False):
        seq_len = x.shape[1]
        attention_weights = dict()

        if gpu:
            x=x.cuda()
        x = self.embedding(x)
        x = torch.mul(x, (self.d_model**(1/2)))

        if gpu:
            x += self.pos_encoding[:, :seq_len, :].cuda()
        else:
            x += self.pos_encoding[:, :seq_len, :]
    
        if training is True:
            x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                       look_ahead_mask, padding_mask)

            attention_weights["decoder_layer{}_block1".format(i+1)] = block1
            attention_weights["decoder_layer{}_block2".format(i+1)] = block2
        
        return x, attention_weights



if __name__ == "__main__":
    d = TransformerDecoder(num_layers=100,
                           d_model=512,
                           num_head=8,
                           intermediate_dim=100, 
                           target_vocab_size=100,
                           max_tgt_len=100,
                           shared_emb_layer=None,
                           rate=0.1)
    print(type(d))
