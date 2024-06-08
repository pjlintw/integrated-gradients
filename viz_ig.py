import os
import argparse
from functools import partial

import captum
import spacy
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization, IntegratedGradients

from models.rnn import CustomLSTM

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Collect all examples
vis_data_records_ig = list()


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    
    # arguments
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="vizs")
    parser.add_argument("--max_seq_length", type=int, default=20)
    return parser.parse_args()


def read_prediction_file(pred_file):
    """Read output file predicted by model during training."""
    
    # List of tuple containing text (str) and label (int) 
    data_lst = list()
    with open(pred_file, "r") as f:
        for line in f.readlines():
            label, _, sent = line.strip().split("\t")
            data_lst.append((sent, int(float(label))))
    return data_lst


def build_vocab(vocab_file):
    """Build vocabulary."""

    tok2idx = dict()
    idx2tok = dict()
    
    with open(vocab_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            idx = int(idx)
            tok = line.strip()
            
            tok2idx[tok] = idx
            idx2tok[idx] = tok

    return tok2idx, idx2tok


def add_pad_to_sequence(sequence, max_seq_len):
    """Add [PAD] token.

    Args:
      sequence: sentence in string
      max_seq_len: maximum length of the padded sequence
    
    """
    sequence = sequence.split()
    sent_len = len(sequence)

    if sent_len > max_seq_len:
        padded_sentence_lst = sequence[:max_seq_len]
    else:
        padded_sentence_lst = sequence
            
    # Add [PAD]
    num_pad = max_seq_len - sent_len
    padded_sentence_lst += ["[PAD]"] * num_pad
    
    # [CLS] + sentence + [SEP]
    padded_len = len(padded_sentence_lst)

    assert len(padded_sentence_lst) == (max_seq_len)
    return padded_sentence_lst


def interpret_sentence(model, sentence, tok2idx, idx2tok, max_seq_len, label):
    """Apply integrated gradient on sentence."""

    # length tensor
    tokens_len = len(sentence.split()) 
    if len(sentence.split()) > max_seq_len:
        tokens_len = max_seq_len
    length_tensor = torch.tensor([tokens_len])

    # input tensor
    padded_tokens = add_pad_to_sequence(sentence, max_seq_len)
    indexed = [tok2idx[tok] for tok in padded_tokens]    
    input_indices = torch.tensor(indexed)
    input_indices = input_indices.unsqueeze(0)
    
    model.zero_grad()

    # fix `text_len` argument
    model_fn = partial(model, text_len=length_tensor)

    # predict
    pred = model(input_indices, text_len=length_tensor).item()
    pred_ind = round(pred)

    idx2label = {1:"pos", 0:"neg"}

    # generate reference indices for each sample
    token_reference = TokenReferenceBase(reference_token_idx=tok2idx["[PAD]"])
    reference_indices = token_reference.generate_reference(max_seq_len, device=device).unsqueeze(0)


    # compute attributions and approximation delta using layer integrated gradients
    ig = LayerIntegratedGradients(model_fn, model.embedding)
    attributions_ig, delta = ig.attribute(input_indices, reference_indices, \
                                           n_steps=500, return_convergence_delta=True)

    #print('pred: ', Label.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))

    add_attributions_to_visualizer(attributions=attributions_ig, 
                                   text=padded_tokens, 
                                   pred=pred, 
                                   pred_ind=pred_ind,
                                   label=label, 
                                   delta=delta, 
                                   vis_data_records=vis_data_records_ig,
                                   idx2label=idx2label)
    


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records, idx2label):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            idx2label[pred_ind],
                            idx2label[label],
                            idx2label[1],
                            attributions.sum(),
                            text,
                            delta))



def main():
    # Argument parser
    args = get_args()

    # Create folder for saving HTMLs
    mdoel_name = args.model_dir.split("/")[1]
    output_path = os.path.join(args.output_dir, mdoel_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_ckpts = os.listdir(args.model_dir)

    # build vocab
    tok2idx, idx2tok = build_vocab(os.path.join(args.result_dir, "vocab.train"))

    for idx, ckpt in enumerate(model_ckpts):

        model = torch.load(os.path.join(args.model_dir, ckpt))

        # Some model weights are saved as state dict which 
        # has to be load with `model.load_state_dict()`
        try:
            model.eval()
        except:
            # load as state dict
            state_dict = torch.load(os.path.join(args.model_dir, ckpt))
            
            # vocab size: 11471
            model = CustomLSTM(vocab_size=len(tok2idx))
            model.load_state_dict(state_dict)
    
            model.eval()

        model = model.to(device)

        # prefix: `dis` refers to  discriminator and `match` is matching network 
        prefix, epoch, step, suffix = ckpt.split(".")

        # Prediction file
        if prefix == "dis":
            pred_file = "fake.sequence." + epoch + "." + step + ".pred"
        elif prefix == "match":
            pred_file = "condition." + epoch + "." + step + ".pred"
        else:
            print("Can not find correspnding preidct file.")

        html_file = pred_file + '.html'
        print(f"Processing file {idx+1}: ", pred_file) 

        # `pred_file` is the 
        pred_file = os.path.join(args.result_dir, pred_file)
        data_lst = read_prediction_file(pred_file)
        

        # Apply attribution method 
        for example in data_lst:
            sent,label = example 
            interpret_sentence(model, sent, tok2idx, idx2tok, args.max_seq_length, label)
        

        # Visualize and save as HTML 
        html_obj = visualization.visualize_text(vis_data_records_ig)
        with open(os.path.join(output_path, html_file), "w") as f:
            f.write(html_obj.data)


    print(f"Saving {len(model_ckpts)} visualization results to {output_path}")
         


if __name__ == "__main__":
    main()
