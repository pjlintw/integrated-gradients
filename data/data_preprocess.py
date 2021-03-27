"""Extract POS tags and split dataset."""
import os
import argparse
import random

from collections import Counter
from pathlib import Path

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--eval_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=1000)

    return parser.parse_args()


def sentence_parse(sentence, label_list):
    """Parse sentence into list of text, and label.

    Args:
      sentence: str, sentence line.
      label_list: list of class in string.

    Returns:
      example: tuple of word list, sentence and index
              
    Example:
        ("what"
         4,
         "what is it ?")
    """
    sentence_list = sentence.split()

    # question mark not separate
    if sentence_list[-1][-1] == "?":
        # Remove and add 
        sentence_list[-1] = sentence_list[-1][:-2]
        sentence_list +=  ["?"]

    sent_len = len(sentence_list)
    label = None

    # check first 5 words have question works
    for w_idx in range(min(5, sent_len)):
        for q_word in label_list:
            if q_word == sentence_list[w_idx]:
                label = q_word 
                break
        
    # Return None if no label
    if label == None:
        return None

    examples = (label, sent_len, " ".join(sentence_list))
    return examples


def build_examples(data_file):
    """Create data example of features.
    
    Returns:
      examples: List of tuple contains (1) label,
                (2) length, (3) sentence.
    """
    label_list = ["who", "what", "when", "where", "which", "how", "whose"]

    # List of tuple contain (w_pos, word, pos)
    examples = list()
    
    with open(data_file, 'r') as f:
        w_position_list = list()
        word_list = list()
        pos_list = list()

        for line in f:
            # Ignore the document information
            if line == '\n':
                continue

            if line[:2] == "id":
                continue

            sentence = line.split("\t")[1]
            # Tuple of elements: label, length, sentence
            example = sentence_parse(sentence, label_list)

            if example is not None:
                examples.append(example)
            
    return examples


def save_example_to_file(examples, output_file):
    """Save examples.
    Args:
      examples: List of examples
      output_file: wirte file.
    """
    # as format "what   4   what is it ?"
    with open(output_file, 'w') as wf:
        for (label, sent_len, sentence) in examples:    
            # Separate by tab
            wf.write(f"{label}\t{sent_len}\t{sentence}\n")
    
def save_vocab(examples, output_file):
    """Save examples.
    Args:
      examples: List of examples
      output_file: wirte file.
    """
    vocab_cnt = Counter()

    for pairs in examples:
        _, _, sentence = pairs
        vocab_cnt.update(sentence.split())

    with open(output_file, 'w') as wf:
        for (word, freq) in vocab_cnt.most_common():    
            # Separate by tab
            wf.write(f"{word}\t{freq}\n")
    print(vocab_cnt)
    


def main():
    # Argument parser
    args = get_args()
    SEED = 49

    # Collect examples from `sample.conll`
    examples = build_examples(data_file=args.dataset_name)
    num_examples = len(examples)
    print("Loading {} examples".format(num_examples))

    # # Shuffle 
    random.Random(SEED).shuffle(examples)
    print("Seed {} is used to shuffle examples".format(SEED))
    
    # Write `sample.tsv`
    write_file = Path(args.output_dir, "sample.tsv")
    save_example_to_file(examples=examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(num_examples, write_file))

    ### Train, validation, test splits. ###
    n_eval = int(args.eval_samples)
    n_test = int(args.test_samples)

    # Spliting datasets
    train_examples = examples[:-n_eval-n_test]
    eval_start = -(n_eval+n_test)
    eval_examples = examples[eval_start:-n_test]
    test_examples = examples[-n_test: ]
    
    # Write `sample.train`
    write_file = Path(args.output_dir, "sample.train")
    save_example_to_file(examples=train_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(train_examples), write_file))


    # Write `sample.dev`
    write_file = Path(args.output_dir, "sample.dev")
    save_example_to_file(examples=eval_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(eval_examples), write_file))
    

    # Write `sample.test`
    write_file = Path(args.output_dir, "sample.test")
    save_example_to_file(examples=test_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(test_examples), write_file))


    # Write vocab
    write_file = Path(args.output_dir, "word.vocab")
    save_vocab(examples=train_examples,
               output_file=write_file)
    print("Saving {} vocabulary to {}".format(len(train_examples), write_file))

if __name__ == '__main__':
    main()
