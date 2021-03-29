"""Build RNN layer, RNN encoder"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CustomLSTM(nn.Module):

    def __init__(self, vocab_size, dimension=128):
        super(CustomLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size,
                embedding_dim,
                lstm_dim,
                n_class,
                n_layer,
                dropout=0.1,
                padding_idx=0,
                bidirectional=False):
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

        bidirectional: bool.
        """
        super(LSTMEncoder, self).__init__()

        ############################################
        # Attributes
        ############################################
        if bidirectional:
            num_direction = 2
        else:
            num_direction = 1

        ############################################
        # Layers
        ############################################
        # Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        #print("vocab", vocab_size)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_dim, 
                            n_layer,
                            bidirectional=bidirectional,
                            batch_first=True)

        # Concat the forward and backward hidden from biLSTM
        # shape (lstm_dim+lstm_dim, lstm_dim)
        #self.fc1 = nn.Linear(lstm_dim*num_direction, lstm_dim)

        ###  ###
        # (lstm_dim, n_class)
        self.classify_layer = nn.Linear(lstm_dim*num_direction, n_class)

        ### matching between question and condition ###
        # (lstm_dim, n_class)
        self.score_layer = nn.Linear(lstm_dim*num_direction, n_class)
        

    def forward(self, x, lengths=None, hidden=None, mode="classifcation"):
        """Call function

        Note that `PackedSequence` allows faster computation in RNN. But the 
        sentences in batch must be sorted.

        Args:
          x: batch example with shape (batch_size, seq_len)
          lengths: 1D-array, sequence length for each example in batch
          hiddens:
          mode: str, classification" or "matching" to perform D(x) or D(c, G(c))
    
        Reference:
          - https://city.shaform.com/en/2019/01/15/sort-sequences-in-pytorch/
        """
        if lengths is not None:
            ###############################################
            # sorting
            ###############################################
            if torch.is_tensor(lengths):
                seq_len_arr = lengths
            else:
                seq_len_arr = torch.tensor(lengths)

            #print(x)
            lenghts_sorted, sorted_i = seq_len_arr.sort(descending=True)
            _, reverse_i = sorted_i.sort()

            #print(sorted_i)
            x = x[sorted_i]

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
            packed_emb = pack_padded_sequence(input=emb, 
                                              lengths=lenghts_sorted, 
                                              batch_first=True, 
                                              enforce_sorted=True)

            # It expected a 3D Tensor with shape (batch_size, seq_len, emb_dim)
            # packed_output (batch, seq_len, n_direction * hidden_size) 
            # hidden (num_layer * n_direction, batch, hidden_size)
            packed_output, (hidden, cell) = self.lstm(packed_emb)

            #print(hidden.shape)
            #h = hidden.permute(1,2,0).squeeze()

            # Recover packed sequence to (batch, seq_len, lstm_dim*n_direction)
            out_padded, output_len = pad_packed_sequence(packed_output, batch_first=True)        
            #print("out_padded", out_padded.shape)

            # 3D to 2D
            # (batch_size, 1, lstm_dim*n_direction) -> (batch_size, lstm_dim*n_direction)
            last_h = out_padded[:, -1, :].squeeze()
            # print("last_h", last_h.shape)
            
            # Perform D(x) or D(c, G(c))
            # out: shape (batch_size, 1)
            if mode == "classification":
                logits = self.classify_layer(last_h)
            elif mode == "matching":
                logits = self.score_layer(last_h)
            else:
                raise ValueError("mode has to be \"classifcation\" or \"matching\".")

            pred = torch.sigmoid(logits)
            #print("out after dense layer", logits.shape)
            #print("pred after squeeze", pred.shape)
        return logits, pred


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

