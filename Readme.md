# Improving Quotation Attribution in Literary Texts

This repository contains code for the experiments described in the paper [`Improving Automatic Quotation Attribution in Literary Novels`](https://aclanthology.org/2023.acl-short.64/) (ACL 2023 short paper).

Citation:
```
@inproceedings{vishnubhotla2023improving,
  title={Improving Automatic Quotation Attribution in Literary Novels},
  author={Vishnubhotla, Krishnapriya and Rudzicz, Frank and Hirst, Graeme and Hammond, Adam},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={737--746},
  year={2023}
}
```


## Notes
The code for the various components of the attribution model (character identification, coreference resolution, speaker identification), as well as auxiliary data formatting and evaluation scritps, are not very well integrated as of now. I hope to convert this into a neater set of pipelines in the near future, but it is hold as I am working on other projects for now.

## Data
Source data for the PDNC dataset in `data/pdnc_source`. This data is the starting point for all further experiments.

We also split the set of annotated quotations in PDNC into train, dev, and test subsets for training and evaluation the speaker identification model (`data/train_splits`). We stratify the splits in two ways: split the data from each novel independently (`random`), and split the set of novels (`leave-x-out`) where the quotations from a novel can only be in one of the three splits. We use 5-fold cross-validation for the training experiments.

## Models
- The BookNLP model (pretrained) can be run on PDNC using the `pdnc_run_pipeline.py` scripts. Outputs will be stored in `booknlpen/pdnc_output`. These outputs in a few other places subsequently, particularly training on PDNC.

- The spacy coref model can be run using the `coref/run_spacy.py` script. Outputs will be stored in `coref/outputs/spacy`.

- GutenTag is currently not integrated in these scripts. It needs to be run as a standalone application (http://www.cs.toronto.edu/~jbrooke/gutentag/), and only works with English corpora from GutenTag. I have included its outputs for the set of novels in PDNC in the `gutentag/data` folder. I would love to port the Character Identification algorithm from GutenTag to a more accessible version at some point.

- Data and scripts for training the BookNLP speaker identification model on PDNC are the `training` folder. Data is formatted in the required format in the `training/data/pdnc` folder, for both the `random` and `leave-x-out` splits.
The `train_speaker.py` scripts can be run to train and save the model in `training/trained_models`. Predictions on the dev and test sets are also saved in the same folder.

## Evaluation
The `eval_functions.py` script is a repository of functions that evaluate the above models on character identification (gutentag, spacy, booknlp), coreference resolution (spacy, booknlp), and speaker identification (booknlp). 
[I have added comments indicating what each function is evaluating but this is the prime candidate to start the clean-up process.]


### Contact
The primary author can be contacted via email: vkpriya@cs.toronto.edu