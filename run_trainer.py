"""Train conditional text GANs with trainer."""
import os
import argparse
import logging
import json
import pathlib
import sys
from functools import partial
import numpy as np

import captum
import spacy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchtext
import torchtext.legacy.data
from torchtext import vocab
from torchtext.vocab import Vocab

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from datasets import ClassLabel, load_dataset, load_metric

from models.rnn import LSTMEncoder,CustomLSTM

from models.transformer import Transformer
from models.utils import create_transformer_masks, init_weights, prepare_discriminator_data,convert_tensor_to_tokens,save_k_exmaple_from_tensor,check_k_exmaple_from_tensor
from models.transformer_blocks import WarmupScheduler

cuda_is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_is_available else "cpu")


class LangugageGAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator


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
    parser.add_argument('--max_seq_length', type=int, default=40)
    parser.add_argument('--vocab', type=str, required=True)
    
    # Modeling discriminator
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_embedding_dims', type=int, default=256)
    parser.add_argument('--rnn_dims', type=int, default=256)
    parser.add_argument('--rnn_classes', type=int, default=1)
    parser.add_argument('--rnn_dropout_rate', type=float, default=0.1)
    parser.add_argument('--rnn_bidirectional', type=bool, default=False)

    # Modeling generator
    parser.add_argument('--tf_layers', type=int, default=2)
    parser.add_argument('--tf_embedding_dims', type=int, default=256)
    parser.add_argument('--tf_dims', type=int, default=512)
    parser.add_argument('--tf_heads', type=int, default=8)
    parser.add_argument('--tf_dropout_rate', type=float, default=0.1)
    parser.add_argument('--tf_shared_emb_layer', type=bool, default=False)
    parser.add_argument('--tf_learning_rate', type=float, default=1e-2)

    # Training
    parser.add_argument('--dataset_script', type=str)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mle_epochs', type=int, default=3)
    parser.add_argument('--train_discriminator_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=50)
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

def train_generator_MLE(generator, 
                        dataset,
                        opt, 
                        logging_steps=50, 
                        epochs=1,
                        tokenizer_dict=None,
                        args=None):
    """Pre-train the generator with MLE."""
    # Prepare for `decode_batch`
    vocab_size = tokenizer_dict["vocab_size"] 
    id2tok = tokenizer_dict["id2tok"] 
    tok2id = tokenizer_dict["tok2id"] 
    unk_idx = tokenizer_dict["tok2id"]["[UNK]"]
    pad_idx = tokenizer_dict["tok2id"]["[PAD]"]

    # nn.NLLLoss: use log-softmax as input 
    # nn.CrossEntropyLoss: use logit as input
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1))
        total_loss = 0
        total_accuracy = 0
        for step, features_dict in enumerate(dataset):
            opt.zero_grad()

            batch_token_ids = features_dict["batch_token_ids"]
            batch_labels = features_dict["batch_labels"]
            batch_lengths = features_dict["batch_lengths"]
    
            # To 2D tensor
            batch_labels = batch_labels.unsqueeze(1) #.cuda()
            #print("shape of encoder's input", batch_labels.shape)
            #print(batch_labels)
            batch_token_ids = torch.tensor(batch_token_ids) #.cuda()
            # Sample a batch of sequences from generator 
            gen_inp = batch_token_ids[:, :-1] #.cuda()
            gen_target = batch_token_ids[:, 1:] #.cuda()

            enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(batch_labels, gen_inp, pad_idx, gpu=args.gpu)
            output, attn = generator(batch_labels,
                                     gen_inp,
                                     training=False,
                                     enc_padding_mask=enc_padding_mask,
                                     look_ahead_mask=combined_mask,
                                     dec_padding_mask=dec_padding_mask,
                                     cuda=args.gpu)
            
            # (batch_size*(seq_len-1), vocab_size)
            viewed_output = output.view(-1, vocab_size)
            # (batch_size*(seq_len-1))
            gen_target = gen_target.reshape(-1)
            
            # print("after reshape", viewed_output.shape)
            # print("after reshape", gen_target.shape)
            loss = loss_fn(viewed_output, gen_target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
            opt.step()

            if (step+1) % logging_steps == 0:
                msg = f"Step: {step+1}, Loss: {loss.item():.2f}"
                logging.info(msg)
                print(msg)

            # NLLLoss(inp, target)
            # inp: (batch)
            # target: (batch_size, seq_len-1)
            # print(gen_inp[0,:])
            # print(gen_target[0,:]) 
            # print("inp shape", gen_inp.shape)
            # print("tgt shape", gen_target.shape)
            # print("token1", decode_batch(gen_inp, id2tok, unk_idx)[0])
            # print("token1", decode_batch(gen_target, id2tok, unk_idx)[0])
        pred_sentences = decode_batch(gen_inp, id2tok, unk_idx)
        print("Done", "", pred_sentences[0])
        print(pred_sentences[1])
    return None

def train_discriminator(generator, 
                        discriminator, 
                        match_network,
                        dataset,
                        opt_dis,
                        opt_match, 
                        logging_steps=50, 
                        epochs=1,
                        tokenizer_dict=None,
                        args=None):
    """Pre-train the discriminator.
    
        (1) Distinguish true example from 
        (2) Measuring wetheater the sentence and condition are right paring.
    """
    # Prepare for `decode_batch`
    vocab_size = tokenizer_dict["vocab_size"] 
    id2tok = tokenizer_dict["id2tok"] 
    tok2id = tokenizer_dict["tok2id"] 
    unk_idx = tokenizer_dict["tok2id"]["[UNK]"]
    pad_idx = tokenizer_dict["tok2id"]["[PAD]"]

    loss_opt_fn = nn.BCELoss(size_average=False)
    loss_match_fn = nn.BCELoss(size_average=False)
    total_loss = list()
    classification_loss = list()

    matching_loss = list()
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1))
        sys.stdout.flush()
        total_classification_loss = 0
        total_matching_loss = 0

        total_classification_acc = 0
        total_matching_acc = 0
        for step, features_dict in enumerate(dataset):
            ###############
            # Batch preparation
            ###############
            batch_size = features_dict["batch_token_ids"].shape[0] 
            half_batch = int(batch_size/2)
            
            # G(c)
            batch_labels = features_dict["batch_labels"][:half_batch]
            batch_lengths = features_dict["batch_lengths"][:half_batch]
  
            #print("type batch length", batch_lengths)
            # D(x)
            batch_token_ids = features_dict["batch_token_ids"][:half_batch,:]
            # D(c, x)
            batch_pos_condition_ids = features_dict["batch_pos_condition_ids"][:half_batch,:]
            batch_neg_condition_ids = features_dict["batch_neg_condition_ids"][:half_batch,:]

            # To 2D tensor
            batch_labels = batch_labels.unsqueeze(1)
            # Sample a batch of sequences from generator 
            fake_seq_target = torch.zeros(half_batch, 1)
        
            ######################
            # Generator 
            ######################
            # (batch_size, max_seq_len)
            fake_seq = generator.sample(inp=batch_labels,
                                        max_len=args.max_seq_length,
                                        temperature=0.3,
                                        training=False,
                                        sos_idx=tok2id["[CLS]"],
                                        eos_idx=tok2id["[SEP]"],
                                        cuda=args.gpu)

            dis_seq_inp, dis_seq_tar, seq_lengths = prepare_discriminator_data(pos_samples=batch_token_ids, 
                                                                               neg_samples=fake_seq,
                                                                               pos_lengths=batch_lengths,
                                                                               neg_lengths=batch_lengths,
                                                                               gpu=args.gpu)
            ##############################
            # Discriminator: perform D(x)
            ##############################
            # Set gradient zero
            opt_dis.zero_grad()

            # `logit` is unnormalized 
            # `pred` is normalized by sigmoid
            # (batch_size*2, 1)
            #print("dis seq inp shape", dis_seq_inp.shape)
            #seq_logits, seq_pred = discriminator(dis_seq_inp, lengths=seq_lengths, mode="classification")
            seq_pred = discriminator(dis_seq_inp, seq_lengths)
            seq_pred = seq_pred.squeeze() # To 1D-tensor
            
            # Check
            # k_example = 10
            # seq_tokens_list = convert_tensor_to_tokens(dis_seq_inp, tok2id, id2tok, first_k_example=k_example)
            # save_k_exmaple_from_tensor('a.out', seq_tokens_list, seq_pred, dis_seq_tar, k_example=10)


            classifcation_loss = loss_opt_fn(seq_pred, dis_seq_tar)
            #print(classifcation_loss)
            classifcation_loss.backward()       
            opt_dis.step()
            
            ##############################
            # Discriminator: perform D(c, G(c))
            ##############################
            # Set gradient zero
            opt_match.zero_grad()
            # Combine pos, neg examples
            dis_pair_inp, dis_pair_tar, dis_pair_lengths = prepare_discriminator_data(pos_samples=batch_pos_condition_ids, 
                                                                                      neg_samples=batch_neg_condition_ids,
                                                                                      pos_lengths=batch_lengths,
                                                                                      neg_lengths=batch_lengths,
                                                                                      gpu=args.gpu)
            
            #pair_logits, pair_pred = match_network(dis_pair_inp, lengths=dis_pair_lengths, mode="matching")
            pair_pred = match_network(dis_pair_inp, dis_pair_lengths)
            
            pair_pred = pair_pred.squeeze() # To 1D-tensor
            # print("pair target", dis_pair_tar[:10])
            # print("pair pred", pair_pred[:10])
            # Check
            # k_example = 10
            # pair_tokens_list = convert_tensor_to_tokens(dis_pair_inp, tok2id, id2tok, first_k_example=k_example)
            # save_k_exmaple_from_tensor('b.out', pair_tokens_list, pair_pred, dis_pair_tar, k_example=10)
            # check_k_exmaple_from_tensor(pair_tokens_list, pair_pred, dis_pair_tar, k_example)
            
            # print("pair pred " , pair_pred.shape)  # (batch_size,)
            # print("pair target", dis_pair_tar.shape)  # (batch_size,)
            
            matching_loss = loss_match_fn(pair_pred, dis_pair_tar)
            matching_loss.backward()   
            opt_match.step()
                
            classification_acc = torch.sum((seq_pred>0.5)==(dis_seq_tar>0.5))
            matching_acc = torch.sum((pair_pred>0.5)==(dis_pair_tar>0.5))
            
            # print("classification after", classifcation_loss.data.item())
            # print("matching after", matching_loss.data.item())
            
            # Total loss
            total_classification_loss += classifcation_loss.data.item()
            total_matching_loss += matching_loss.data.item()
            # Total acc
            total_classification_acc += classification_acc.data.item() 
            total_matching_acc += matching_acc.data.item() 

            if (step+1) % logging_steps == 0 or step == 0:
                avg_cls_loss = classifcation_loss.data.item() / batch_size
                avg_cls_acc = classification_acc.data.item() / batch_size

                avg_match_loss = matching_loss.data.item() / batch_size
                avg_match_acc = matching_acc.data.item() / batch_size

                print(f'Step: {step+1}, Classifcation Loss: {classifcation_loss:.2f}, Avg loss: {avg_cls_loss:.2f}, Avg accuracy {avg_cls_acc:.2f}') 
                print(f'Step: {step+1}, Matching Loss: {matching_loss:.2f}, Avg loss: {avg_match_loss:.2f}, Avg accuracy {avg_match_acc:.2f}') 
            if step+1 == args.max_steps:
                break
                sys.stdout.flush()
        ### Convert to python ###
        # Python list    
        k_example = 20

        fake_tokens_list = convert_tensor_to_tokens(dis_seq_inp, tok2id, id2tok, first_k_example=k_example)
        write_file = get_output_dir(args.output_dir, f'fake.sequence.epoch-{epoch+1}.pred')
        save_k_exmaple_from_tensor(write_file, fake_tokens_list, seq_pred,  dis_pair_tar, k_example)

        pair_tokens_list = convert_tensor_to_tokens(dis_pair_inp, tok2id, id2tok, first_k_example=k_example)
        write_file = get_output_dir(args.output_dir, f'condition.epoch-{epoch+1}.pred')
        save_k_exmaple_from_tensor(write_file, pair_tokens_list, pair_pred, dis_pair_tar, k_example)
        
        ### Convert to python ###
    return ((total_classification_loss, total_matching_loss),
            (total_classification_acc, total_matching_acc))
            



def get_output_dir(output_dir, file):
    """Joint path for output directory."""
    return pathlib.Path(output_dir,file)


def build_dirs(output_dir,logger):
    """Build hierarchical directories."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Create folder for output directory: {output_dir}")

def decode_batch(inp, id2tok, unk_idx, batch=True):
    """Convert word indices into words.

    Args:
      inp: (batch_size, seq_max_len)
      id2tok: dictionary.
      unk_idx: int.
      batch: bool
    """
    batch_example = list()
    if batch:
        for pred_ids in inp:
            batch_example.append([ id2tok[int(w_idx)] if int(w_idx) in id2tok else id2tok[unk_idx] for w_idx in pred_ids ])

    return batch_example

def main():
    # Argument parser
    args = get_args()
    SEED = 49

    args.gpu = cuda_is_available
    # Create output dir
    output_dir = args.output_dir

    # Logger
    logger = logging.getLogger(__name__)
    build_dirs(output_dir, logger)

    log_file = get_output_dir(output_dir, 'example.log')
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(msecs)d %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    logger.info(args)
    # Saving arguments
    write_path = get_output_dir(output_dir, 'hyparams.txt')
    with open(write_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        logger.info(f"Saving hyperparameters to: {write_path}")
    ########## Load dataset from script. ##########
    # 'wiki-table-questions.py'
    datasets = load_dataset(args.dataset_script)
    logger.info("Loading Datasets")

    for i in datasets["train"]:
        print(i)
        break

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
    # ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose']
    condition_list = [ '['+c+']' for c in label_list]

    ### Create vocabulary, token-to-index, index-to-token
    vocab = build_vocab(args.vocab)
    vocab.update(["[CLS]", "[UNK]", "[SEP]", "[PAD]"])
    vocab.update(condition_list) # Add conditions words to vocab

    tok2id = {w: idx for idx, w in enumerate(vocab)}
    id2tok = {v: k for k, v in tok2id.items()}

    vocab_size = len(vocab)
    UNK_IDX = tok2id["[UNK]"]
    PAD_IDX = tok2id["[PAD]"]
    logging.info(f"PAD_IDX: {PAD_IDX}")
    #print("PAD_IDX", PAD_IDX)

    tokenizer_collector = dict()
    tokenizer_collector["vocab"] = vocab
    tokenizer_collector["vocab_size"] = vocab_size
    tokenizer_collector["tok2id"] = tok2id
    tokenizer_collector["id2tok"] = id2tok
    

    ########## Load the custom model, tokenizer and config ##########
    def tokenize_fn(examples, max_seq_len):
        """Add special tokens to input sequence and padd the max lengths.
        
        Args:
          Examples: dict of features:
                    {"tokens": [ 'what', 'was', 'the', 'average', 'in', '2001'],
                     "label": 3} # label index 
        
        Variables:
          tokens:  
            [ '[CLS]','what', 'was', 'the', 'average', 'in', '2001', '[SEP]', '[PAD]']
          condition_tokens:
            [ '[what]',  '[CLS]','what', 'was', 'the', 'average', 'in', '2001', '[SEP]', '[PAD]']

        """
        def _pad_sequence(sequence, max_seq_len, n_special_token=0):
            sent_len = len(sequence)
            max_seq_len = max_seq_len - n_special_token
            max_sent_len = max_seq_len if sent_len >= max_seq_len else (sent_len)
               
            # Extend words list with special tokens
            padded_sentence_lst = ["[CLS]"]+ sequence[:max_sent_len] + ["[SEP]"] 

            # [CLS] + sentence + [SEP]
            padded_len = len(padded_sentence_lst)
            
            # Add [PAD]
            num_pad = max_seq_len+n_special_token - padded_len
            padded_sentence_lst += ["[PAD]"] * num_pad

            #print(padded_sentence_lst)
            assert len(padded_sentence_lst) == (max_seq_len+n_special_token)
            return padded_sentence_lst

        feature_dict = dict()

        token_col_name = 'tokens'
        label_col_name = 'label'

        token_ids = list()
        
        tokens  = examples[token_col_name]
        sent_len = len(tokens)

        # Positive example
        label_idx = examples[label_col_name]
        label_token = condition_list[label_idx]  # [ "[who]", "[when]", "[where]", "[which]" ]
        label_ids = list(range(len(condition_list)))
        condition_tokens = [label_token] + examples[token_col_name]

        # Negative example
        neg_label_idx = label_idx
        while neg_label_idx == label_idx:
            neg_label_idx = np.random.choice(label_ids)
        neg_label = condition_list[neg_label_idx]
        
        ### Add special token and pad ###
        # 2 for [CLS] and [SEP]
        padded_sentence_lst = _pad_sequence(tokens, max_seq_len, 2)
        padded_condition_sentence_lst = [label_token] + padded_sentence_lst[:-1]
        padded_neg_condition_sentence_lst = [neg_label] + padded_sentence_lst[:-1]

        # Add the length up to [SEP]
        if "[PAD]" not in padded_sentence_lst:
            feature_dict["padded_length"] = max_seq_len
        else:
            feature_dict["padded_length"] = padded_sentence_lst.index("[PAD]")

        # Add padded tokens 
        feature_dict["padded_tokens"]= padded_sentence_lst
        feature_dict["padded_pos_condition_tokens"]= padded_condition_sentence_lst
        feature_dict["padded_neg_condition_tokens"]= padded_neg_condition_sentence_lst

        # Add features
        token_ids = [ tok2id[tok] if tok in tok2id else tok2id["[UNK]"] for tok in padded_sentence_lst ]
        feature_dict["token_ids"] = torch.tensor(token_ids)

        condition_token_ids = [ tok2id[tok] if tok in tok2id else tok2id["[UNK]"] for tok in padded_condition_sentence_lst ]
        feature_dict["pos_condition_token_ids"] = torch.tensor(condition_token_ids)

        neg_condition_token_ids = [ tok2id[tok] if tok in tok2id else tok2id["[UNK]"] for tok in padded_neg_condition_sentence_lst ]
        feature_dict["neg_condition_token_ids"] = torch.tensor(neg_condition_token_ids)

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
    def generate_batch(data_batch, gpu):
        """Package feature as mini-batch."""
        features_dict = dict()
        batch_token_ids, batch_labels = list(), list()
        batch_lengths = list() # List of sentence length
        batch_pos_condition_ids = list()
        batch_neg_condition_ids = list()

    
        #print("len batch", len(data_batch))
        for batch_group in data_batch:
            batch_token_ids.append(batch_group["token_ids"])
            batch_labels.append(batch_group["label"])
            batch_lengths.append(batch_group["padded_length"])
            batch_pos_condition_ids.append(batch_group["pos_condition_token_ids"])
            batch_neg_condition_ids.append(batch_group["neg_condition_token_ids"])

        features_dict["batch_token_ids"]= torch.tensor(batch_token_ids)
        features_dict["batch_labels"]= torch.tensor(batch_labels)
        features_dict["batch_lengths"]= torch.tensor(batch_lengths)
        features_dict["batch_pos_condition_ids"] = torch.tensor(batch_pos_condition_ids)
        features_dict["batch_neg_condition_ids"] = torch.tensor(batch_neg_condition_ids)

        if gpu:
            features_dict["batch_token_ids"] = features_dict["batch_token_ids"].cuda()
            features_dict["batch_labels"]= features_dict["batch_labels"].cuda()
            features_dict["batch_pos_condition_ids"] = features_dict["batch_pos_condition_ids"].cuda() 
            features_dict["batch_neg_condition_ids"] = features_dict["batch_neg_condition_ids"].cuda()

        return features_dict

    
    # Construct generator
    generator = Transformer(num_layers=args.tf_layers,
                            d_model=args.tf_dims,
                            num_head=args.tf_heads,
                            intermediate_dim=args.tf_dims*4,
                            input_vocab_size=num_labels,
                            target_vocab_size=vocab_size,
                            src_max_len=5,
                            tgt_max_len=args.max_seq_length,
                            padding_idx=PAD_IDX,
                            shared_emb_layer=args.tf_shared_emb_layer, # Whether use embeeding layer from encoder
                            rate=args.tf_dropout_rate)
    if cuda_is_available:
        generator.cuda()
    #generator.apply(init_weights)
    logging.info(generator.encoder)


    out = torch.tensor([1,0,1,1])
    tar = torch.tensor([0,0,1,1])
    batch_size = tar.shape[0]
    print(torch.sum((out>0.5)==(tar>0.5)).data/batch_size)
            
    # o = torch.tensor([tok2id["[CLS]"]]*10).unsqueeze(0)
    # print(o.shape)

    # Construct discriminator
    # discriminator = LSTMEncoder(vocab_size=vocab_size,
    #                             embedding_dim=args.rnn_embedding_dims,
    #                             lstm_dim=args.rnn_dims,
    #                             n_class=args.rnn_classes,
    #                             n_layer=args.rnn_layers,
    #                             dropout=args.rnn_dropout_rate,
    #                             padding_idx=PAD_IDX,
    #                             bidirectional=True)

    # match_network = LSTMEncoder(vocab_size=vocab_size,
    #                             embedding_dim=args.rnn_embedding_dims,
    #                             lstm_dim=args.rnn_dims,
    #                             n_class=args.rnn_classes,
    #                             n_layer=args.rnn_layers,
    #                             dropout=args.rnn_dropout_rate,
    #                             padding_idx=PAD_IDX,
    #                             bidirectional=True)

    discriminator = CustomLSTM(vocab_size=vocab_size)
    match_network = CustomLSTM(vocab_size=vocab_size)

    
    if cuda_is_available:
        discriminator.cuda()
        match_network.cuda()
    logging.info(discriminator)

    generate_batch_fn = partial(generate_batch, gpu=args.gpu)
    ### Fetch dataset iterator
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size,
                           shuffle=True, collate_fn=generate_batch_fn)
    eval_iter = DataLoader(eval_dataset, batch_size=args.batch_size,
                           shuffle=True, collate_fn=generate_batch_fn)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=True, collate_fn=generate_batch_fn)

    ### Pre-train generator ###
    print("Pre-train the generator")
    gen_optimizer = optim.Adam(generator.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9)
    gen_optimizer = WarmupScheduler(model_size=args.tf_dims,
                                    factor=2,
                                    warmup=4000,
                                    optimizer=gen_optimizer)
    
    train_generator_MLE(generator=generator,
                         dataset=train_iter,
                         opt=gen_optimizer,
                         logging_steps=50, 
                         epochs=args.mle_epochs,
                         tokenizer_dict=tokenizer_collector,
                         args=args)
    
    ### Pre-train generator ###
    print("Pre-train discriminator")
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=1e-2)
    match_optimizer = optim.Adam(match_network.parameters(), lr=1e-2)
    train_discriminator(generator=generator, 
                        discriminator=discriminator,
                        match_network=match_network,
                        dataset=train_iter,
                        opt_dis=dis_optimizer,
                        opt_match=match_optimizer, 
                        logging_steps=args.logging_steps,
                        epochs=args.train_discriminator_epochs,
                        tokenizer_dict=tokenizer_collector,
                        args=args)
    
    ### Training ###
    # Initialize our adversarial Trainer
    # trainer = Trainer(
    #     generator=generator,
    #     discriminator=discriminator,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    

    ### Evalutation ### 
    return None
    

if __name__ == "__main__":
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    inp = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(inp), target)

    print(m(inp))
    print(m(inp).type())
    print(target)
    print(target.type())
    print(output)
    output.backward()
    main()
