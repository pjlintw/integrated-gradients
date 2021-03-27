"""Get information about the label."""
import re
import os
import argparse
from collections import defaultdict
import math
import numpy as np

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--output_dir', type=str, default='./')

    return parser.parse_args()

def main():
    # Argument parser
    args = get_args()

    # Init variables for counting
    num_sentences = 0
    num_labels = 0
    cur_sentence_length = 0
    sent_length_lst = list()
    label2freq = defaultdict(int)

    # Preprocess tags in `sample.tsv`
    with open(args.dataset_name, 'r') as f:    
        for idx, line in enumerate(f):
            if line == "\n":
                continue
            # Get sentence length when encounting `*`
            num_sentences += 1
            
            line_lst = line.strip().split("\t")
            # Get sentence length when encounting `*`
            # Counting
            assert len(line_lst) == 3 
            label, sent_len, sentence = line_lst

            sent_length_lst.append(int(sent_len))
            label2freq[label] += 1
            num_labels += 1 # count the number of all tags
            
    # Sort 
    sent_length_arr = np.sort(sent_length_lst)

    percentile_list = [50,75,80,90,95,96,97,98,99,100]
    for perc in percentile_list:
        print(f"{perc}% percentile {np.percentile(sent_length_arr, perc)}")
        
    # Maximum, min and mean sequence length
    max_len = max(sent_length_lst)
    min_len = min(sent_length_lst)
    mean_len = sum(sent_length_lst) / len(sent_length_lst)

    # Create output file
    write_file = os.path.join(args.output_dir, 'sample.info')

    with open(write_file, 'w') as wf:
        # Percentile
        for perc in percentile_list:
            wf.write("{}% percentile {}\n".format(perc,
                                                  np.percentile(sent_length_arr, perc)))
        wf.write("\n")
        # Write max, min, mean sentence length and number of sent
        wf.write(f'Max sequence length: {max_len}\n')
        wf.write(f'Min sequence length: {min_len}\n')
        wf.write(f'Mean sequence length: {mean_len}\n')
        wf.write(f'Number of sequence: {num_sentences}\n\n')
        wf.write('Tags:\n')

        # Write POS tag and its percentage of the words
        tag_percentage = [ (k, '%.2f'%(v/num_labels*100)) for k,v in label2freq.items() ]
        tag_percentage = sorted(tag_percentage, key=lambda x: x[0])
        # Sort by tag name
        for (tag, perc)  in sorted(tag_percentage, key=lambda x: x[0]):
            wf.write(f'{tag : <5}\t{perc}%\n')    

if __name__ == '__main__':
    main()

