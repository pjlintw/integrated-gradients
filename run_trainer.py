"""Train conditional text GANs with trainer."""
import os
import argparse
import captum

import spacy

import torch
import torchtext
import torchtext.legacy.data
import torch.nn as nn
import torch.nn.functional as F

from torchtext import vocab
from torchtext.vocab import Vocab

from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from datasets import ClassLabel, load_dataset, load_metric
from functools import partial

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    # Model
    parser.add_argument('--model_name_or_path', type=str, default="textgan")
    parser.add_argument('--output_dir', type=str, default='tmp/')
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--vocab', type=str, required=True)
    
    # Training
    parser.add_argument('--dataset_script', type=str)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)

    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--max_val_samples', type=int)
    parser.add_argument('--max_test_samples', type=int)

    parser.add_argument('--logging_first_step', default=True, required=False)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=500)
    
    return parser.parse_args()

def build_vocab(vocab_file, min_count=0):
    """Build vocabulary from file.

    Args:
      vocab_file: path  

    Returns
    """
    vocab = set()
    with open(vocab_file, "r") as f:
        for line in f:
            if line == "\n":
                continue

            word, freq = line.strip().split("\t")
            if int(freq) >= min_count:
                vocab.add(word)
    return vocab




def create_imdb_datasets():
    """Create IMDb datasets."""
    nlp = spacy.load('en_core_web_sm')


    TEXT = torchtext.legacy.data.Field(lower=True, tokenize='spacy')
    Label = torchtext.legacy.data.LabelField(dtype = torch.float)


    train_it = torchtext.datasets.IMDB(root='.data/', split="train")
    train_iter = DataLoader(train_it, batch_size=8, shuffle=False)



    # List of two elements
    for exm in train_iter:
        label,texts = exm
        print(len(texts))
        pt(texts[0])
        break


    # train, test = torchtext.datasets.IMDB.splits(text_field=TEXT,
    #                                              label_field=Label,
    #                                              train='train',
    #                                              test='test',
    #                                              path='data/aclImdb')
    #test, _ = test.split(split_ratio = 0.04)

    #loaded_vectors = vocab.GloVe(name='6B', dim=100)

    # If you prefer to use pre-downloaded glove vectors, you can load them with the following two command line
    # loaded_vectors = torchtext.vocab.Vectors('data/glove.6B.100d.txt')
    # TEXT.build_vocab(train, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))
        
    # TEXT.vocab.set_vectors(stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim)
    # Label.build_vocab(train)

    datasets = None
    return datasets



def train():
    """Adversarial training for CGANs.

    Args:
      generator
    """
    for n_step in range(config.num_steps):
        # Sample positive examples

        # Sample noise examples

        # obtain generated data 
        generated_pred = generator()

        # Update discriminator
        reader.step()

        # Update generator
        generator.step()

    
class Trainer:
    def __init__():
        pass



def main():
    # Argument parser
    args = get_args()
    SEED = 49
    
    print(args)
    # Set
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ########## Load dataset from script. ##########
    # 'wiki-table-questions.py'
    datasets = load_dataset(args.dataset_script)
    

    for i in datasets["train"]:
        print(i)
        break

    ### Create vocabulary, token-to-index, index-to-token
    vocab = build_vocab(args.vocab)
    print(len(vocab))
    vocab.update(["[CLS]", "[UNK]", "[SEP]", "[PAD]"])
    print(len(vocab))

    tok2id = {w: idx for idx, w in enumerate(vocab)}
    id2tok = {v: k for k, v in tok2id.items()}

    UNK_ID = tok2id["[UNK]"]
    print(id2tok[UNK_ID])


    ### Access column names and features ###
    if args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features

    # In the event the labels are not a `Sequence[ClassLabel]`,
    # we will need to go through the dataset to get the unique labels.
    if isinstance(features["label"], ClassLabel):
        label_list = features["label"].names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
        print("here")
    else:
        pass
    # `label_list` : label to id
    # `label_to_id`: id to label
    num_labels = len(label_list)
    print(label_list)

  
    ########## Load the custom model, tokenizer and config ##########
    def tokenize_fn(examples, max_seq_len):
        """Tokenize the input sequence and align the label.
        `input_ids` and `label_ids` will be added in the feature example (dict).
        They are required for the forward and loss computation.
        Addtionally. `-100` in `label_ids` is assigned to segmented tokens
        and to speical tokens in BERT. Loss function will ignore them.
        Args:
          Examples: dict of features:
                    {"tokens": [AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06'],
                     "pos_tags": [22, 6, 22, 22, 23, 11]}
        Return:
          tokenized_inputs: dict of futures including two 
                            addtional feature: `input_ids` and `label_ids`.

        Usages:
        >>> tokenized_dataset = datasets.map(tokenize_fn,
        >>>                                  batched=True)
            # Check whether aligned.
        >>> for example in tokenized_dataset:
                tokens = example['tokens']
                input_ids = example['input_ids']
                tokenized_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                label_ids = example['label_ids'] # aligned to max length 
                print(tokens)
                print(tokenized_tokens)
                print(input_ids)
        [ 'SOCCER' ] # token
        [ [CLS], 'S', '##OC, '##CE', '##R', [SEP] ] #converted_tokens 
        [ -100 ,  4 , -100,  -100, -11] # label_ids
        """
        feature_dict = dict()

        token_col_name = 'tokens'
        label_col_name = 'label'

        token_ids = list()
        sent_len = len(examples[token_col_name])
        
        # truncate sentence 
        max_sent_len = max_seq_len if sent_len >= max_seq_len else (sent_len)
        max_sent_len = max_sent_len-2

        #print(examples[token_col_name])
    
        num_pad = max(0, max_seq_len - (max_sent_len + 2))
        # print("max_seq_len", max_seq_len)
        # print("max_sent_len", max_sent_len)
        # print("num_pad", num_pad)

        suffix_lst = ["[SEP]"] + ["[PAD]"]*num_pad
        
        # Extend words list with special tokens
        #print(examples[token_col_name])
        padded_sentence_lst = ["[CLE]"]+ examples[token_col_name][:max_sent_len] + suffix_lst
        #print(padded_sentence_lst)
        assert len(padded_sentence_lst) == max_seq_len

        # Add padded tokens 
        feature_dict["padded_tokens"]= padded_sentence_lst
        
        # Add token ids
        token_ids = [ tok2id[tok] if tok in tok2id else tok2id["[UNK]"] for tok in padded_sentence_lst ]
        feature_dict["token_ids"] = torch.tensor(token_ids)

        # # Convert label to idx
        # # Add `labels` sequence for loss computation
        # feature_dict["labels"] = examples[label_col_name]

        return feature_dict

    tokenize_fn = partial(tokenize_fn, max_seq_len=args.max_seq_length)

    ### Truncate  number of examples ###
    if args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_fn
        )

    if args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_fn
        )

    if args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_fn
        )

    ### Feature
    # `token_ids`, `labels` for training and loss computation
    def generate_batch(data_batch):
      batch_token_ids, batch_labels = [], []
      #print("len batch", len(data_batch))
      for batch_group in data_batch:
        batch_token_ids.append(batch_group["token_ids"])
        batch_labels.append(batch_group["label"])
        #de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        #en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
      return batch_token_ids, batch_labels

    # Iter
    train_iter = DataLoader(train_dataset, batch_size=32,
                            shuffle=True, collate_fn=generate_batch)

    for features, labels in train_iter:
        print(len(features))
        print(len(labels))

        print(type(features))
        print(type(labels))
        break

    ### Training ###

    
    ### Evalutation ### 
    return None
    

if __name__ == "__main__":
    main()
