"""Build RNN layer, RNN encoder"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMEncoder(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_dim,
                 lstm_dim,
                 dense_dim,
                 n_class,
                 n_layer,
                 dropout=0.0,
                 padding_idx=0,
                 bidirectional=True):
        """LSTM encoder intialization.

        Args:
          vocab_size: ...
          embedding_dim: ...
          lstm_dim: ...
          dense_dim: ...
          n_class: ...
          n_layer: ...
          dropout (float): ...,
          padding_idx: zero vector for the padding index.
          
          >> embedding = nn.Embedding(5, 2, padding_idx=0)
          >> input = torch.tensor([0,2])
          >> embedding(input)
          tensor([[[0.0000, 0.0000, 0.0000],
                   [0.1232,-2.3201, 0.3223]]])

          bidirectional: bool, ...
        """
        # Constructor
        super(LSTMEncoder, self).__init__()

        ############################################
        # Attributes
        ############################################


        ############################################
        # Layers
        ############################################
        # Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_dim, 
                            n_layer,
                            bidirectional=bidirectional,
                            batch_first=True)

        # Concat the forward and backward hidden from biLSTM
        # shape (lstm_dim+lstm_dim, lstm_dim)
        self.fc1 = nn.Linear(lstm_dim*2, lstm_dim)

        # (lstm_dim, n_class)
        self.output_layer = nn.Linear(lstm_dim, n_class)


    def forward(self, x, lengths=None, hidden=None):
        """Call function

        Note that `PackedSequence` allows faster computation in RNN. But the 
        sentences in batch must be sorted.

        Args:
          input: batch example with shape (batch_size, seq_len)
          seq_len_arr: 1D-array, sequence length for each example in batch

        Reference:
          - https://city.shaform.com/en/2019/01/15/sort-sequences-in-pytorch/
        """

        if lengths is not None:
          ###############################################
          # sorting
          ###############################################
          print(x)
          lenghts_sorted, sorted_i = seq_len_arr.sort(descending=True)
          _, reverse_i = sorted_i.sort()

          print(sorted_i)
          x = x[sorted_i]

          print(x)

          # Map word id to embedding
          emb = self.embedding_layer(x)
          # print(emb)
          # print(emb.shape)

          # Packed sequence
          # Normal batch computation in RNN does 
          # more computations than required.
          # `packed_emb` is tuple of two tenosor.
          # (non-padding-word-embeddings, )
          # reference: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
          packed_emb = pack_padded_sequence(emb, lenghts_sorted, batch_first=True, enforce_sorted=True)

          # It expected a 3D Tensor with shape (batch_size, seq_len, emb_dim)
          # packed_output (batch, seq_len, n_direction * hidden_size) 
          # hidden (num_layer * n_direction, batch, hidden_size)
          packed_output, (hidden, cell) = self.lstm(packed_emb)
    
          # Recover packed sequence to (batch, seq_len, lstm_dim*n_direction)
          out_padded, output_len = pad_packed_sequence(packed_output, batch_first=True)        
          
          # dense layer
          # (lstm_dim*n_direction, lstm_dim)
          dense_out1 = self.fc1(out_padded)
          
          # output layer
          # pred = self.fc2(dense_out1)

          # return pred



if __name__ == '__main__':
    model = LSTMEncoder(vocab_size=10,
                        embedding_dim=2,
                        lstm_dim=2,
                        dense_dim=5,
                        n_class=3,
                        n_layer=1,
                        bidirectional=True)

    sents = torch.tensor([[1,2,5,0],[1,6,5,2] , [2,0,0,0]])

    sent_len_arr = torch.tensor([3, 4, 1])
    # o = model(sents, lengths=sent_len_arr)

    # s = torch.tensor([2,321,3021, 2, -1])
    
    idx = torch.tensor([0,1])
    new_x = sents.index_select(axis=0, index=idx)
    print(new_x)    

