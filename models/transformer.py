    
    
from encoder import TransformerEncoder
from decoder import TransformerDecoder

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_head,
                 intermediate_dim,
                 input_vocab_size,
                 target_vocab_size,
                 src_max_len,
                 tgt_max_len,
                 padding_idx,
                 shared_emb_layer=None, # Whether use embeeding layer from encoder
                 rate=0.1):
        super(Transformer, self).__init__()

        # (vocab_size, emb_dim)
        self.embedding_layer = nn.Embedding(input_vocab_size, d_model,
                                            padding_idx)
    
        self.encoder = TransformerEncoder(num_layers, d_model, num_head,
                                          intermediate_dim,
                                          input_vocab_size,
                                          src_max_len, 
                                          rate)

        if shared_emb_layer is True:
            self.shared_emb_layer = self.embedding_layer
        else:
            self.shared_emb_layer = shared_emb_layer
        print(self.shared_emb_layer)
        self.decoder = TransformerDecoder(num_layers, d_model, num_head,
                                         intermediate_dim,
                                         target_vocab_size,
                                         tgt_max_len,
                                         self.shared_emb_layer,  # share embedding
                                         rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)
    
        
    def forward(self, src, tgt, training, 
                enc_padding_mask,
                look_ahead_mask, dec_padding_mask):
        """Forward propagate for transformer.
        
        Args:
          src: (batch_size, src_max_len)
            
        """
        # Mapping
        src = self.embedding_layer(src)

        # (batch_size, inp_seq_len, d_model)
        enc_out = self.encoder(src, training, enc_padding_mask)

        # (batch_size, tgt_seq_len, d_model)
        dec_output, dec_attn = self.decoder(tgt, enc_out, training, look_ahead_mask,
                                   dec_padding_mask)

        # (batch_size, tgt_seq_len, target_vcoab_size)
        final_output = self.final_layer(dec_output)

        return final_output, dec_attn

if __name__ == "__main__":
    transf = Transformer(2,512,8,2048,8500, 8000,10000,6000, -100, None)


    i = torch.randint(0, 200, (64,38))  
    tgt_i = torch.randint(0,200, (64, 36))

    output, attn = transf(i,
                          tgt_i,
                          training=False,
                          enc_padding_mask=None,
                          look_ahead_mask=None,
                          dec_padding_mask=None)

    print(output.shape)
