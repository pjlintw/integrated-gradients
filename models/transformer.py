
from .utils import create_transformer_masks
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, 
                 num_layers,
                 d_model, num_head,
                 intermediate_dim,
                 input_vocab_size,
                 target_vocab_size,
                 src_max_len,
                 tgt_max_len,
                 padding_idx,
                 shared_emb_layer=None, # Whether use embeeding layer from encoder
                 rate=0.1):
        super(Transformer, self).__init__()

        self.pad_idx = padding_idx

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
    
        
    def forward(self, src, tgt, training, enc_padding_mask,
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

    def sample(self, inp, max_len, temperature, sos_idx, eos_idx):
        """Forward propagate for transformer.
        
        Args:
          inp 
          max_len
          temperature
          sos_idx
          eos_idx
          
            
        """
        if torch.is_tensor(inp):
            pass
        else:
            inp = torch.tensor(inp)

        eps = 1e-10        
        # Gumbel-Softmax tricks
        batch_size = inp.shape[0]
        #sampled_ids = torch.zeros(batch_size, max_len).type(torch.LongTensor)

        # (batch_size, 1)
        output = torch.tensor([sos_idx]*batch_size).unsqueeze(1)      
        
        assert output.shape[-1] == 1 

        for i in range(max_len):
            # enc_pad_mask, combined_mask, dec_pad_mask
            enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(inp, output, self.pad_idx )


            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _ = self.forward(inp,    # (bathc_size, 1)
                                         output, # (batch_size, 1-TO-MAXLEN)
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            
            # Select the last word from the seq_len dimension
            # (batch_size, 1, vocab_size) to (batch_size, voacb_size) 
            predictions = predictions[: ,-1:, :].squeeze() 
            # print("preds", predictions.shape)

            # (batch_size, 1)
            #assert inp.shape[-1] = 1
            gumbel_distribution = gubel_softmax_sample(predictions, temperature)
            # (batch_size, vocab_size)
            # print("gumbel", gumbel_distribution.shape)

            # (batch_size,) to (bathc_size, 1)
            predicted_idx = torch.argmax(gumbel_distribution, dim=-1).unsqueeze(1)
            
            # print("pred idx", predicted_idx.shape)

            output = torch.cat((output, predicted_idx), 1)
            
            # Update along with col
            #sampled_ids[:,i] = predicted_idx.squeeze()
        #print(sampled_ids==output[:,1:])
        return output

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    noise = torch.rand(shape)
    return -torch.log(-torch.log(noise+eps)+eps)

def gubel_softmax_sample(logits, temperature):
    """Sample from Gumbel softmax distribution.
    Reference:
        1. Gumbel distribution: https://en.wikipedia.org/wiki/Gumbel_distribution
        2. Inverse Tranform Sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    """
    y = logits + sample_gumbel(logits.shape)
    return nn.functional.softmax(y/temperature, dim=-1)

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
