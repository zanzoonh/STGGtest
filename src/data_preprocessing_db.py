## Adding protein-conditioning of 

import os
import argparse
from pathlib import Path
import numpy as np
from props.properties import penalized_logp, MolLogP_smiles, qed_smiles, ExactMolWt_smiles, compute_flat_properties_nogap
from model.mxmnet import MXMNet
from data.dataset import get_prot_datasets
from data.target_data import Data as TargetData
DATA_DIR = "../resource/data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='bindingdb')
    parser.add_argument("--MAX_LEN", type=int, default=250)
    parser.add_argument("--force_vocab_redo", action="store_true") # force rebuilding the vocabulary in case of bug or something
    params = parser.parse_args()
    raw_dir = f"{DATA_DIR}/{params.dataset_name}"
    datasets = get_prot_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=False, 
                n_properties=1, force_vocab_redo=params.force_vocab_redo)
    for split, dataset in zip(['train', 'valid', 'test'], datasets):
        max_len = 0
        smile = max(dataset.smiles_list, key = len)
        #print(smile)
        smile_transformed = TargetData.from_smiles(smile, dataset.vocab, randomize_order=False, MAX_LEN=params.MAX_LEN)
        #print(smile_transformed.sequence)
        smile_transformed.featurize()
        smile_reversed = smile_transformed.to_smiles()
        #print(smile_reversed)
        new_len = len(smile_transformed.sequence)
        max_len = max(max_len, new_len)
        #print(f"split={split} Approximate-Max-Length={max_len}")
   