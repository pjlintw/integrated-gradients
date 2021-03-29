"""module for transformer."""

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F



class WarmupScheduler:
    """warm-up scheduler."""
    def __init__(self, model_size, factor, optimizer, warmup=4000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_angles(pos, i, dim):
    # angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim))
    angle_rates = 1 / torch.pow(10000, ((2 * (i//2)) / dim) )
    return pos * angle_rates



def positional_encoding(position, dim):
    # angle_rads = get_angles(np.arange(position)[:, np.newaxis],
    #                       np.arange(dim)[np.newaxis, :],
    #                       dim)
    angle_rads = get_angles(torch.arange(position).unsqueeze_(1),
                            torch.arange(dim).unsqueeze_(0),
                            dim)    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads.unsqueeze_(0)
    return pos_encoding



def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate scaled dot-product attention 
        Formula: softmax(dot(q, k) * 1/sqrt(dim) ) * v
        k, v must have same length in `seq_len` to be able compute (attn_w * v )
    Args:
        q, k, v: shape (batch_size, num_head, seq_len, head_dim) 
        mask: enc_pad_mask, future_mask or dec_pad_mask
            default is None
    Retruns:
        out: shape (batch_size, num_head, seq_len_q, head_dim_v)
        attn_w: shape (..., seq_len_q, seq_len_k)
    """
    # as (batch_size, num_head, head_dim, seq_len_k)
    transposed_k = k.permute(0,1,3,2)

    
    # attention_score: shape (batch_size, num_head, seq_len_q, seq_len_k)
    attn_logit = torch.matmul(q, transposed_k)

    # scaled by head_dim
    head_dim = q.size()[-1]
    # head_dim = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attn_logit = attn_logit / (head_dim **(1/2.0))

    
    
    if mask is not None:
        scaled_attn_logit += (mask * -1e9)

    # softmax normalize on last axis (seq_len_k)
    attn_w = F.softmax(scaled_attn_logit, dim=-1)

    # (..., seq_len_q, seq_len_k) * (..., seq_len_v, head_dim_v)
    out = torch.matmul(attn_w, v)

    return out, attn_w


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, inp_dim, d_model, n_head, use_output_layer=None):
        super(MultiHeadAttention, self).__init__()
        self.inp_dim = inp_dim
        self.d_model = d_model
        self.n_head = n_head
        self.use_output_layer = use_output_layer        
        assert self.d_model % self.n_head == 0
    
        self.head_dim = self.d_model // self.n_head

        self.w_q = nn.Linear(self.inp_dim, self.d_model)
        self.w_k = nn.Linear(self.inp_dim, self.d_model)
        self.w_v = nn.Linear(self.inp_dim, self.d_model)

        # optional, not necessary for attention
        if self.use_output_layer is not None:
            self.output_layer = nn.Linear(self.d_model, self.d_model)


    def split_head(self, x, batch_size):
        """Split `n_head` apart from with origial shape

        Args:
            x: shape (batch_size, max_len, dim)
            batch_size: scalar
        Returns:
            output wiwth shape (batch_size, num_head, seq_len, head_dim)
        """
        # x = tf.reshape(x, [batch_size, -1, self.n_head, self.head_dim])
        x = x.view(batch_size, -1, self.n_head, self.head_dim)
        return x.permute(0, 2, 1, 3)


    def forward(self, q, k, v, mask):
        """Perform self or  cross attentiong
        Args:
            q, k, v: shape (batch_size, seq_len, emb_dim)
        Return:
            out: shape (batch_size, seq_len, diml)
            attn_w: shape (batch_size, num_head, seq_len_q, seq_len_k)
        """
        batch_size = q.shape[0]

        # map embeding dim to dim
        q = self.w_q(q) #.cuda()
        k = self.w_k(k) #.cuda()
        v = self.w_v(v) #.cuda()

        # shape (batch_size, num_head, seq_len, head_dim)
        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)

        # scaled_attn: (batch_size, num_head, seq_len_q, head_dim_v)
        # attn_weight: (batch_size, num_head, seq_len_q, seq_len_k)
        scaled_attn, attn_w = scaled_dot_product_attention(q, k, v, mask)

        # transpose as (batch_size, seq_len, num_head, head_dim)
        scaled_attn = scaled_attn.permute(0, 2, 1, 3)
        # reshape as (batch_size, seq_len, dim)
        out = scaled_attn.reshape(batch_size, -1, self.d_model)
        # print("out after reshape b,seq_len,dim", out.shape)
        
        if self.use_output_layer:
            out = self.output_layer(out)

        #out.cuda()
        #attn_w.cuda()
        return out, attn_w


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, ff_dim):
        super(FeedForwardBlock, self).__init__()

        self.dim = dim
        self.ff_dim = ff_dim

        self.dense1 = nn.Linear(self.dim, self.ff_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.ff_dim, self.dim)

    def forward(self, x):
        """Calculate feed-forward block
        Formual: dense(relu(dense(x)))
        Args:
            x: shape (batch_size, seq_len, dim)
        Retrun:
            out: shape (batch_size, seq_len, dim)
        """

        dense_out = self.relu(self.dense1(x))
        out = self.dense2(dense_out)

        return out



if __name__ == "__main__":
    sample_ffn = FeedForwardBlock(512, 2048)
    x = torch.rand((64,50,512))
    y = sample_ffn(x)   
    print(y.shape)

    opts = [WarmupScheduler(512, 1, 4000, None), 
            WarmupScheduler(512, 1, 8000, None),
            WarmupScheduler(256, 1, 4000, None)]

    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()



