"""Train conditional text GANs with trainer."""
import os

import argparse
import logging
import json
import pathlib
import sys
from functools import partial

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

from models.rnn import LSTMEncoder
from models.transformer import Transformer
from models.utils import create_transformer_masks, init_weights
from models.transformer_blocks import WarmupScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument('--rnn_dims', type=int, default=512)
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
    # parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    # parser.add_argument('--per_device_eval_batch_size', type=int, default=64)

    parser.add_argument('--mle_epochs', type=int, default=3)
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


def train_generator_MLE(generator, dataset,
                        opt, logging_steps=50, epochs=1,
                        tokenizer_dict=None):
    """Pre-train the generator with MLE."""
    # Prepare for `decode_batch`
    vocab_size = tokenizer_dict["vocab_size"] 
    id2tok = tokenizer_dict["id2tok"] 
    unk_idx = tokenizer_dict["tok2id"]["[UNK]"]
    pad_idx = tokenizer_dict["tok2id"]["[PAD]"]

    # nn.NLLLoss: use log-softmax as input 
    # nn.CrossEntropyLoss: use logit as input
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        total_loss = 0
        total_accuracy = 0
        for step, features_dict in enumerate(dataset):
            opt.optimizer.zero_grad()

            batch_token_ids = features_dict["batch_token_ids"]
            batch_labels = features_dict["batch_labels"]
            batch_lengths = features_dict["batch_lengths"]
    
            # To tensor
            batch_labels = torch.tensor(batch_labels).unsqueeze(1).cuda()
            #print("shape of encoder's input", batch_labels.shape)
            #print(batch_labels)
            batch_token_ids = torch.tensor(batch_token_ids).cuda()
            # Sample a batch of sequences from generator 
            gen_inp = batch_token_ids[:, :-1].cuda()
            gen_target = batch_token_ids[:, 1:].cuda()

            enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(batch_labels.cuda(), gen_inp.cuda(), pad_idx)
            output, attn = generator(batch_labels.cuda(),
                                     gen_inp.cuda(),
                                     training=False,
                                     enc_padding_mask=enc_padding_mask.cuda(),
                                     look_ahead_mask=combined_mask.cuda(),
                                     dec_padding_mask=dec_padding_mask.cuda())
            
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

            if (step+1) % 10 == 0:
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

def train_discriminator(generator, discriminator, dataset,
                        opt, steps=1, epochs=1,
                        tokenizer_dict=None):
    """Pre-train the discriminator.
    
        (1) Distinguish true example from 
        (2) Measuring wetheater the sentence and condition are right paring.
    """
    # Prepare for `decode_batch`
    id2tok = tokenizer_dict["id2tok"] 
    unk_idx = tokenizer_dict["tok2id"]["[UNK]"]
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0
        total_accuracy = 0
        for step, features_dict in enumerate(dataset):
            #print("step", step)
            batch_token_ids = features_dict["batch_token_ids"]
            batch_labels = features_dict["batch_labels"]
            batch_lengths = features_dict["batch_lengths"]
    
            batch_token_ids = torch.tensor(batch_token_ids)            
            batch_size = batch_token_ids.shape[0]

            # Sample a batch of sequences from generator 
            gen_inp = batch_token_ids[:, :-1]
            gen_target = batch_token_ids[:, 1:]
            print(gen_inp[0,:])
            print(gen_target[0,:])
            break
            print("inp shape", gen_inp.shape)
            print("tgt shape", gen_target.shape)
            
            #fake_out = generator(gen_inp, gen_tgt, training=False)
            target = torch.ones(batch_size, 1)
            #print(target)
            #print(target.shape)

            # Set gradient zero
            opt.zero_grad()
            
            # `pred` is normalized by sigmoid
            # both has shape (batch_size, 1)
            logits, pred = discriminator(batch_token_ids, lengths=batch_lengths)
            
            # Loss and update
            loss_fn = nn.BCELoss()
            loss = loss_fn(pred, target)

            loss.backward()
            
            acc = torch.sum((pred>0.5)==(target>0.5))
            
            if (step+1) % 5 == 0 or step == 0:
                avg_loss = loss.data.item() / batch_size
                avg_acc = acc.data.item() / batch_size

                print(f'Loss: {loss:.2f}, Avg loss: {avg_loss:.2f}, Avg accuracy {avg_acc:.2f}')
            
            # Total loss
            total_loss += loss.data.item()
            total_accuracy += acc.data.item() # 

            opt.step()
            # sample_idx = generator.sample(inp=batch_labels,
            #                               max_len=args.max_seq_length,
            #                               temperature=0.3,
            #                               sos_idx=tok2id["[CLS]"],
            #                               eos_idx=tok2id["[SEP]"])
            #print(sample_idx)
            #r = decode_batch(sample_idx, id2tok, UNK_IDX, batch=True)
            #print(r)
            
            # break
            # step = 0
            # if step % 50 == 0:
            #     r = decode_batch(sample_idx, id2tok, UNK_IDX, batch=True)
            #     print(r)

       

                
class Trainer:
    def __init__(self, model, args, train_dataset=None, eval_dataset=None,
                tokenizer=None, data_collator=None, compute_metrics=None):
        return None

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

            # Sample noise examples


            # Sample conditions
            batch = None

            # Update generator
            generator.step()

    def get_train_dataloader(self):
        return DataLoader()    


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
    #tok2id["[PAD]"] = -100
    id2tok = {v: k for k, v in tok2id.items()}

    vocab_size = len(vocab)
    UNK_IDX = tok2id["[UNK]"]
    PAD_IDX = tok2id["[PAD]"]
    logging.info(f"PAD_IDX: {PAD_IDX}")
    print("PAD_IDX", PAD_IDX)

    tokenizer_collector = dict()
    tokenizer_collector["vocab"] = vocab
    tokenizer_collector["vocab_size"] = vocab_size
    tokenizer_collector["tok2id"] = tok2id
    tokenizer_collector["id2tok"] = id2tok
    

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
                            addtional feature: `padded_tokens`,
                                                `token_ids`,
                                                `discriminator_inp`

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

        # print(examples[token_col_name])
        # print(sent_len)
        # print( ["[SEP]"] +["[PAD]"]*0)
        # truncate sentence 
        max_seq_len = max_seq_len - 2

        max_sent_len = max_seq_len if sent_len >= max_seq_len else (sent_len)
    
        #print(examples[token_col_name])
        # print("max_seq_len", max_seq_len)
        # print("max_sent_len", max_sent_len)
        
        # Extend words list with special tokens
        #print(examples[token_col_name])
        padded_sentence_lst = ["[CLE]"]+ examples[token_col_name][:max_sent_len] + ["[SEP]"] 

        # [CLS] + sentence + [SEP]
        padded_len = len(padded_sentence_lst)
        
        # Add [PAD]
        num_pad = max_seq_len+2 - padded_len
        padded_sentence_lst += ["[PAD]"] * num_pad

        #print(padded_sentence_lst)
        assert len(padded_sentence_lst) == (max_seq_len+2)

        # Add padded tokens 
        feature_dict["padded_tokens"]= padded_sentence_lst
        
        # Add token ids
        token_ids = [ tok2id[tok] if tok in tok2id else tok2id["[UNK]"] for tok in padded_sentence_lst ]
        feature_dict["token_ids"] = torch.tensor(token_ids)

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
        """"""
        features_dict = dict()
        batch_token_ids, batch_labels = [], []
        batch_lengths = list() # List of sentence length
        #print("len batch", len(data_batch))
        for batch_group in data_batch:
            batch_token_ids.append(batch_group["token_ids"])
            batch_labels.append(batch_group["label"])
            batch_lengths.append(batch_group["length"])
    
        features_dict["batch_token_ids"]=batch_token_ids
        features_dict["batch_labels"]=batch_labels
        features_dict["batch_lengths"]=batch_lengths
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
    discriminator = LSTMEncoder(vocab_size=vocab_size,
                                embedding_dim=args.rnn_embedding_dims,
                                lstm_dim=args.rnn_dims,
                                n_class=args.rnn_classes,
                                n_layer=args.rnn_layers,
                                dropout=args.rnn_dropout_rate,
                                padding_idx=PAD_IDX,
                                bidirectional=args.rnn_bidirectional)
    discriminator.cuda()
    logging.info(discriminator)
    
    
    ### Fetch dataset iterator
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=generate_batch)
    eval_iter = DataLoader(eval_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=generate_batch)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=generate_batch)


    ### Pre-train generator ###
    print("Pre-train the generator")
    gen_optimizer = optim.Adam(generator.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9)
    gen_optimizer = WarmupScheduler(model_size=args.tf_dims,
                                    factor=2,
                                    warmup=4000,
                                    optimizer=gen_optimizer)
    #gen_optimizer =torch.optim.SGD(generator.parameters(), lr=5.0)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    train_generator_MLE(generator=generator,
                        dataset=train_iter,
                        opt=gen_optimizer,
                        logging_steps=50, 
                        epochs=args.mle_epochs,
                        tokenizer_dict=tokenizer_collector)
    
    ### Pre-train generator ###
    print("Pre-train discriminator")
    # dis_optimizer = optim.Adagrad(discriminator.parameters())
    # train_discriminator(generator=generator, 
    #                     discriminator=discriminator, 
    #                     dataset=train_iter,
    #                     opt=dis_optimizer,
    #                     steps=1,
    #                     epochs=1,
    #                     tokenizer_dict=tokenizer_collector)
    
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
    main()
