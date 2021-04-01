# Integrated Gradients for Conditional Text Generation

[**Data**](#dataset-and-preprocessing) | [**Conditional Text GAN**](#conditional-text-gans) | [**Integrated-Gradients**](#integrated-gradients-on-text-gans)]


The repository works on training and analysing the conditional text GANs where it is a Transformer-based generator for the question generation. We leverage the explainable method, integrated gradients, to analyze how discriminator distinguishes the real sentence from the artificial sentence generated by the generator and an additional discriminator, matching network, for scoring the matching between the generated sentence and the condition word.

For conditional GANs, it is important to use an additional discriminator to judge whether the condition word and the sentence is positive or negative pairing. Because it forces the generative model to generate a sentence in consistency with the condition word.

We use `wikiTableQuestions` datasets to generate the question given certain question word. Our generator produce a question sentence to fool the discriminator. The dataset contains 14153 pairs of question sentences and answers. We extract 12179 question sentences from the `wikiTableQuestions`. To reproduce the results, follow the steps below.

* New March 28th, 2021: Train conditional text GANs    


## Files Structure

There are many files in the result folder.
It is mandatory for applying integrated gradients method on the result trained by the conditional text GANs. 

The figure below uses a `results/cgan-exp-50` as example and explains the purpose of these files. Note that `#N` refers to a number with the file. 

```
|--cgan-exp-50
|  └-- ckpt
|      |-- dis.epoch-#N.step-#N.pt     # checkpoint for the discriminator saving at the Nth step of Nth epoch.
|      └-- match.epoch-#N.step-#N.pt   # checkpoint for the matching network saving at the Nth step of Nth epoch.
|    
|-- hyparams.txt					         # Hyperparameters for modeling and training
|-- example.log                        # Log file 
|-- phrase_translator.py               # Implementation of phrase extraction and phrase model
|-- condition.epoch-#N.step-#N.pred      # Prediction from the matching network in the Nth step of Nth epochs.
└── fake.sequence.epoch-#N.step-#N.pred  # Prediction from the discriminator at the Nth step of Nth epochs.
```






## Installation

### Python version

* Python >= 3.8

### Environment

Create an environment from file and activate the environment.

```
conda env create -f environment.yaml
conda activate integrated-gradients-cgan
```

If conda fails to create an environment from `environment.yaml`. This may be caused by the platform-specific build constraints in the file. Try to create one by installing the important packages manually. The `environment.yaml` was built in macOS.

**Note**: Running `conda env export > environment.yaml` will include all the 
dependencies conda automatically installed for you. Some dependencies may not work in different platforms.
We suggest you to use the `--from-history` flag to export the packages to the environment setting file.
Make sure `conda` only exports the packages that you've explicitly asked for.

```
conda env export > environment.yaml --from-history
```

## Dataset and Preprocessing

### Dataset concatenation

We use the `wikiTableQuestions` dataset with the released version `1.0.2` as the examples for training the transformer-based text GANs. The files are in the `data/wikiTable` folder. All examples are stored in `questions.tsv`. You can get the dataset `data/training.tsv` by the official repository [ppasupat/WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions). Rename the file as  `questions.tsv` and save it to the data folder `data/wikiTable`.

To preprocess the dataset and fetch the statistical information regarding the features and labels. Move to the working directory `data`.

```
cd data
```

In the following steps, the script extract the collected file `question.tsv`, then split them into `sample.train`, `sample.dev` and `sample.test`
for building the datasets. You have to change the relative path to `--datasets_name` if you're using a different file directory.


### Preprocessing and Dataset Splitting

The file `questions.tsv` contains irrelevant information for training the neural nets.
We extract the questions and the corresponding question word as the labels. The generator in Text GANs is trained to generate the question given the labels.

Running `data_preprocess.py` to extract `label`, `sentence length` and `question`, then write them to `sample.tsv` in which the label and the features  are separated by tab. 

The arguments `--dataset_name` and `--output_dir` are the file to be passed to the program and the repository for the output file respectively. 

It generates `sample.tsv` for all extracted examples and `sample.train`, `sample.dev` and `sample.test` for the network training.  The examples will be shuffled in the scripts and split into `train`, `validation` and `test` files.  The arguments `--eval_samples` and `--test_samples`
decide the number of samples will be selected from examples. In the datasets, we select 11779 for the training set, 200 for validation and test sets respectively. To preprocess and split the datasets, you need to run the code below. 

```python
python data_preprocess.py \
  --dataset_name wikiTable/questions.tsv \
  --output_dir wikiTable \
  --eval_samples 200 \
  --test_samples 200
```

These output files for building datasets will be under the path `--output_dir`. You will get the result.

```
Loading 12179 examples
Seed 49 is used to shuffle examples
Saving 12179 examples to wikiTable/sample.tsv
Saving 11779 examples to wikiTable/sample.train
Saving 200 examples to wikiTable/sample.dev
Saving 200 examples to wikiTable/sample.test
```

Make sure that you pass the correct **datasets** to the `--dataset_name`argument and it has enough examples for splitting out development and test sets. The output files may have no example, if the number of eval and test examples are more than the examples in the `questions.tsv`

### Data Information

To get the information regarding the questions and the question words. Execute the script `data_information.py` to compute 
the percentiles, maximum, minimum and mean of the question length, number of examples, label and its percentage.

The arguments `--dataset_name` and `output_dir` are the file to be passed to the program and the repository for the output file respectively. 

```python
python data_information.py \
  --dataset_name wikiTable/sample.tsv \
  --output_dir wikiTable/
```

The output file `sample.info` will be exported in the  `--output_dir` directory.



### Using dataset loading script for WikiTableQuestions

We use our dataset loading script `wiki-table-questions.py`for creating dataset. The script builds the train, validation and test sets from the 
dataset splits obtained by the `data_preprocess.py` program. 
Make sure the dataset split files `sample.train`, `sample.dev` , and `sample.test` are included in the datasets folder `data/wikiTable` your dataset folder.

If you get an error message like:

```
pyarrow.lib.ArrowTypeError: Could not convert 1 with type int: was not a sequence or recognized null for conversion to list type
```

You may have run other datasets in the same folder before. The Huggingface already created `.arrow` files once you run a loading script. These files are for reloading the datasets quickly.

Try to move the dataset you would like to use to the other folder and modify the path in the loading script. 

Or delete the relevant folder and files in the `.cache` for datasets. `cd ~/USERS_NAME/.cache/huggingface/datasets/` and `rm -r *`. This means that all the loading records will be removed and
 Hugging Face will create the `.arrows` files again, including the previous loading records. 


## Conditional Text GANs for Question Generation

The text GANs is a conditional generative adversarial network for discrete variable generation. Given a dataset consisting of the question sentence and the corresponding interrogative word, The sequence of question x = [x1, x2, ..., xT ]$ can be performed as the autoregressive modeling process.

Most text GANs use RNN models. As an alternative, we use a transformer-based model for our generator.

Note that dataset script and vocab file are required.

### Train Conditional Text GANs.

To train the architecture, you have to pass the arguments for dataset loading and vocab file. Use `vocab` and `dataset_script` arguments.
 
```python
 python run_trainer.py \
 --model_name_or_path textgan \
 --output_dir results/cgan-mle \
 --vocab data/wikiTable/word.vocab \
 --dataset_script wiki-table-questions.py \
 --max_seq_length 20 \
 --batch_size 48 \
 --do_train True \
 --do_eval True \
 --do_predict True \
 --max_val_samples 200 \
 --max_test_samples 200 \
 --mle_epochs 1 \
 --logging_first_step True \
 --logging_steps 5 \
 --eval_steps 10 \
 --max_steps 50
```

Regarding the training arguments, `logging_first_step`, `logging_steps` determine the steps for logging out the loss and saving the checkpoints form the discriminator and the matching network. The prediction files will be automatically evaluated. 

Furthermore, the generator was trained with maximum log-likelihood with `mle_epochs` epochs. The discriminator and the matching network are trained with the `max_steps` steps. We set the steps for the training discriminator and the matching network since it is sensitive to the generated sentence.

Note that all the output files, including the logger, hyperparameter, vocabulary file,checkpoint and predictions from the discriminator and matching network, will be saved in the `output_dir`.

This is the result folder for applying the integrated gradient script. 


### Integrated Gradients on Text GANs

We apply integrated gradient technique on the two discriminators: a sentence classifier and a matching network. Both models are the discriminator in the GANs architecture where it is to judge whether the input text is real or fake sequence or to predict whether a pair of a sentence and condition belong together or not.

The explainable method uses the gradient-based attribution method, integrated gradient, is used to our discriminator and the matching network. We explore the gradients-based attribution methods applying to the GANs and understand the effectiveness of the discriminator in the text GANs. 

To compute the attribution score, the `viz_ig.py` scripts take the result folder as input. It is mandatory to pass the result folder for the arguments `model_dir`, `result_dir`, `output_dir` and `max_seq_length`.

```python
python viz_ig.py \
 --model_dir results/cgan-mle/ckpt \
 --result_dir results/cgan-mle \
 --output_dir vizs \
 --max_seq_length 20
```

The script will create a folder with the result folder's name under the `output_dir`. For example, the above program saves all the visualization of attribution scores `.html` in the `vizs/cgan-mle` folder.

The visualization results will be saved in the `vizs` folder. 

```
|--cgan-mle
|  |-- condition.epoch-#N.step-#N.pred.html      # visualization of the matching network at the Nth step of Nth epoch.
|  └-- fake.sequence.epoch-#N.step-#N.pred.html  # visualization of the discriminator at the Nth step of Nth epoch.
```



### Contact Information

For help or issues using the code, please submit a GitHub issue.


