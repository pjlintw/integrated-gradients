
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


def check_k_exmaple_from_tensor(example_lst, y_pred_tensor, y_true_tensor, k_example=20):
    ### Convert to python ###
    # Python list    
    pred_list = y_pred_tensor[:k_example]
    true_list = y_true_tensor[:k_example]

    idx = 0                            
    for sent, y_pred, y_true in zip(example_lst, pred_list, true_list):
        idx+=1
        sentence = " ".join(sent)
        print(f"{y_true}\t{y_pred:.2f}\t{sentence}")

def save_k_exmaple_from_tensor(write_file, example_lst, y_pred_tensor, y_true_tensor, k_example=20):
    ### Convert to python ###
    # Python list    
    pred_list = y_pred_tensor.cpu().detach().numpy()[:k_example]
    true_list = y_true_tensor.cpu().detach().numpy()[:k_example]

    idx = 0
    with open(write_file, "w") as wf:                          
        for sent, y_pred, y_true in zip(example_lst, pred_list, true_list):
            idx+=1
            sentence = " ".join(sent)
            wf.write(f"{y_true}\t{y_pred:.2f}\t{sentence}\n")
    print("Save file to {}".format(write_file))

def convert_tensor_to_tokens(tensor_inp, tok2id, id2tok, first_k_example=None):
    """Convert tensor into list of examples.

    Args:
      tensor_inp: 2D tensor
    """
    examples = list()
    data = tensor_inp.cpu().detach().numpy()
    
    for idx, sent_tensor in enumerate(data):
        tokens = [ id2tok[int(w_idx)] for w_idx in sent_tensor ]
        tokens = [ w  for w in tokens if w != "[PAD]"]
        examples.append(tokens)
        if idx == first_k_example:
            break
    return examples

def prepare_discriminator_data(pos_samples, neg_samples, pos_lengths, neg_lengths, gpu=False):
    """Combine positive and negative examples into a mini-batch.
    
    Args:
      pos_samples: (batch_size, seq_len)
      neg_samples: (batch_size, seq_len)
    
    Returns: inp, target
      inp: (pos_size + neg_size) x seq_len
      target: pos_size + neg_size (boolean 1/0)
    
    Reference:
      https://github.com/suragnair/seqGAN/blob/master/helpers.py
    """
    # concatenate along with the batch axis
    inp = torch.cat((pos_samples, neg_samples), 0)
    
    # (batch*2, seq_len)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # 
    lengths = torch.cat((pos_lengths, neg_lengths), 0)

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]
    lengths = lengths[perm]

    if gpu:
        target=target.cuda()
        inp=inp.cuda()

    return inp, target, lengths



def init_weights(m): 
    if type(m) == nn.Dropout:   
        return None

    if type(m) == TransformerEncoder:
        for layer in m.parameters():
            nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)
        return None

    if type(m) == TransformerDecoder:
        for layer in m.parameters():
            nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)
        return None

    # if m.dim() > 1:        
    nn.init.xavier_uniform(m.weight)
    try:
        m.bias.data.fill_(0.01)
    except:
        print(m, "no bias")

def create_padding_mask(seq, pad_idx):
    """Creating tensor for masking pad tokens of scaled dot-product logits
    Args:
        seq: Tensor with shape (batch_size, seq_len)
             [ [3, 20, 17, 0 ,0 ] [...] [...] ]
        pad_idx: idx to be padded.
    Return:
        seq: Tensor with shape (batch_size, 1, 1, , seq_len)
             [ [ [ [0, 0, 0, 1, 1] [...] [...] ] ] ]
    """
    seq = torch.eq(seq, pad_idx).double()
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, None, None, :]


def create_look_ahead_mask(seq_len):
    """Creating Tensor used for future token masking
    Args:
        src_len: scales
    Returns:
        mask: Tensor with shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask


def create_transformer_masks(src, tgt, pad_idx, gpu=False):
    """Creating three masks for TransformerEncoder and -Decoder 
    Args:
        src: shape (batch_size, src_len)
        tgt: shape (batch_size, tgt_len)
    Return:
        enc_pad_mask: masking pad tokens in the encoder
        combined_mask: used to pad and mask future tokens a.k.a `future_mask`
        dec_pad_mask: masking the encoder outputs in 2nd attention block
    """
    # Encoder padding mask
    enc_pad_mask = create_padding_mask(src, pad_idx)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_pad_mask = create_padding_mask(src, pad_idx)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    if gpu:
        look_ahead_mask = create_look_ahead_mask(tgt.shape[1]).cuda()
        dec_target_padding_mask = create_padding_mask(tgt, pad_idx).cuda()
    else:
         look_ahead_mask = create_look_ahead_mask(tgt.shape[1])
         dec_target_padding_mask = create_padding_mask(tgt, pad_idx)

    combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_pad_mask, combined_mask, dec_pad_mask



if __name__ == "__main__":
    x = torch.tensor([[7, 6, 1, 0, 0], [1, 2, 3, 0, 0], [2, 0, 0, 0, 0]])
    y = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 0, 0], [2, 0, 0, 0, 0]])   
    print(x)
    print(y)
    m = create_padding_mask(x, 0)


    a,b,c = create_transformer_masks(x,y,0)
    print(a)
    print(b)
    print(c)
