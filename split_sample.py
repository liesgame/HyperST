import os

import numpy as np 

import json
import argparse


from HyperST.utils.utils import fix_seed, split_list_numpy


import matplotlib.pyplot as plt


def main(args):
    folder_list_path = os.path.join(args.process_path, args.folder_list_filename)
    slidename_lst = list(np.genfromtxt(folder_list_path, dtype=str))

    fix_seed(args.seed)

    # split train and val

    for i in range(args.kflod):
        slidename_length = len(slidename_lst)
        test_length = max(round(slidename_length * 0.1), 1)
        val_length =  max(round(slidename_length * 0.1), 1)
        train_length = slidename_length - test_length - val_length
        split_dir = args.split_dataset_kflod_dir
        os.makedirs(split_dir, exist_ok=True)
        dataset_split = split_list_numpy(slidename_lst, [train_length, val_length, test_length])
        dataset_split_json = {
            'train_sample' : [str(i) for i in dataset_split[0]],
            'val_sample' : [str(i) for i in dataset_split[1]],
            'test_sample' : [str(i) for i in dataset_split[2]]
        }

        with open(os.path.join(split_dir, args.split_samples_filename + f'_flod_{i}.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_split_json, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./hest1k_datasets')
    parser.add_argument('--dataset', type=str, default='colorectum')
    parser.add_argument('--folder_list_filename', type=str, default='all_slide_lst.txt')
    parser.add_argument('--split_dir', type=str, default='./split')
    parser.add_argument('--kflod', type=int, default=5)
    parser.add_argument('--split_samples_filename', type=str, default='sample_split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set up config
    args.data_path = os.path.join(args.data_root, args.dataset)
    args.process_path = os.path.join(args.data_path, 'processed_data')
    args.split_dataset_dir = os.path.join(args.split_dir, args.dataset)
    args.split_dataset_kflod_dir = os.path.join(args.split_dataset_dir, f'flod_{args.kflod}')

    main(args)