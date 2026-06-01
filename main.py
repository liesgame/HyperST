import os

import torch.utils
import torch.utils.data
from tqdm.notebook import tqdm

import numpy as np 
import pandas as pd
from huggingface_hub import login
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchmetrics
from torchmetrics.regression import (
    PearsonCorrCoef,
    ConcordanceCorrCoef,
    MeanSquaredError,
    MeanAbsoluteError,
    ExplainedVariance
)

from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

import anndata
from datetime import datetime
from tqdm import tqdm
from loguru import logger
import random
import json
import argparse
import torchvision                                          
import wget

from HyperST.datasets.hySTDataset import HyperSTDataset
from HyperST.model import HyperST
from HyperST.utils.utils import fix_seed, split_list_numpy, EarlyStopping
from pretrained.UNI.uni import get_encoder
import matplotlib.pyplot as plt

class TqdmToLoguru:
    def write(self, msg):
        if msg.strip():
            logger.info(msg.strip())
    
    def flush(self):
        pass

def json_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, 
                          np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor)):  
        return obj.cpu().numpy().tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def load_dataset(
        args:argparse.Namespace, 
        train_samples: list,
        val_samples: list,
        test_samples: list,
        selected_genes: list,
        logger=None,
        preprocess=None,
        base_width=None
    ):

    if args.model_name in 'hyperst':
        train_dataset = HyperSTDataset(
            slidename_lst=train_samples, selected_genes=selected_genes, phase='train',
            data_path=args.data_path, process_path=args.process_path, 
            logger=logger, preprocess=preprocess, base_width=base_width
        )
        val_dataset = HyperSTDataset(
            slidename_lst=val_samples, selected_genes=selected_genes, phase='test', 
            data_path=args.data_path, process_path=args.process_path, 
            logger=logger, preprocess=preprocess, base_width=base_width
        )
        test_dataset = HyperSTDataset(
            slidename_lst=test_samples, selected_genes=selected_genes, phase='test',
            data_path=args.data_path, process_path=args.process_path, 
            logger=logger, preprocess=preprocess, base_width=base_width
        )
    else:
        raise ValueError(f'model {args.model_name} is not supported')
    
    return train_dataset, val_dataset, test_dataset


def load_model(
        args:argparse.Namespace,
        selected_genes:list,
        image_encoder=None
):
    if args.img_pretrained_model == 'uni':
        image_dim=1024
    else:
        raise ValueError(f'img_pretrained_model {args.img_pretrained_model} is not supported')

    if args.model_name == 'hyperst':
        model = HyperST(
            image_dim=image_dim,
            gene_dim=len(selected_genes),
            emb_dim=args.emb_dim,
            num_outputs=len(selected_genes),
            mlp_ratio=2.0,
            image_dropout=0.1,
            gene_dropout=args.gene_dropout,
            decoder_dropout=args.image_dropout,
            predict_norm=args.predict_norm,
            entail_weight=args.entail_weight,
            alignment_beta=args.alignment_beta,
            image_encoder=image_encoder,
            logger=args.logger,
            lora_rank=8,
            lora_alpha=16,
            image_encoder_name=args.img_pretrained_model,
            last_layer=args.last_layer,
            is_lora_ffn=args.is_lora_ffn
        )
    else:
        raise ValueError(f'model {args.model_name} is not supported')

    return model
class Trainer:
    def __init__(
            self,
            model:torch.nn.Module,
            model_name:str,
            optimizer:torch.optim.Optimizer,
            scaler,
            total_epoch:int,
            train_loader:torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,
            train_dataset:torch.utils.data.Dataset,
            val_dataset:torch.utils.data.Dataset,
            test_dataset:torch.utils.data.Dataset,
            train_metrics,
            val_metrics,
            test_metrics,
            device,
            logger,
            tboard,
            early_stopping,
            args:argparse.Namespace
        ) -> None:

        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.scaler = scaler
        self.total_epoch = total_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.args = args
        self.device = device
        self.logger = logger
        self.tboard = tboard
        self.early_stopping = early_stopping

        self.ema=None

        self.is_train_metric = True

        self.predict_samples = []


    def _preprocess_inputs(self, batch, dataset):

        for i in batch:
            batch[i] = batch[i].to(self.device)
        return batch
    
    def train_step(self, batch_data, pbar=None):
        self.model.train()

        self.optimizer.zero_grad()
        # ---> Forward
        with torch.autocast(enabled=self.args.amp, device_type='cuda', dtype=torch.float16):

            results_dict = self.model(**batch_data)
            # ---> Loss 
            preds = results_dict['logits']
            loss = results_dict['loss']
            label = batch_data['label']
            
        # ---> backward

        self.scaler.scale(loss).backward()
        if self.model_name == 'hyerst':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        batch_size = label.shape[0]
        if self.is_train_metric:
            self.train_metrics.update(preds, label)
        if pbar:
            pbar.set_postfix({"loss": f"{loss:.3f}"})
        return loss.item() * batch_size, batch_size
    
    def test_step(self, batch_data, test_metrics, final_test=False, not_generate:bool=True):
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(enabled=self.args.amp, device_type='cuda', dtype=torch.float16):
                # ---> Forward
                results_dict = self.model(**batch_data)
                # ---> Loss 
                loss = results_dict['loss']
                preds = results_dict['logits']
                if final_test:
                    self.predict_samples.append(preds)
                label = batch_data['label']
                # ---> metrics
                batch_size = label.shape[0] 
                if self.is_train_metric or final_test or not not_generate:
                    test_metrics.update(preds, label)

        return loss.item() * batch_size, batch_size

    def train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()

        total_loss = 0
        total_count = 0

        if self.logger:
            self.logger.info(f"Train Epoch {epoch}")
        inner_pbar = tqdm(self.train_loader, file=TqdmToLoguru(), desc="Training Step", position=2)
        for batch in inner_pbar:
            batch = self._preprocess_inputs(batch, dataset=self.train_dataset)
            
            batch_loss, batch_size = self.train_step(batch_data=batch, pbar=inner_pbar)
            total_loss += batch_loss
            total_count += batch_size

        # logging
        train_avg_loss = total_loss / total_count

        if self.is_train_metric:
            train_epoch_result = self.train_metrics.compute()
            self.train_metrics.reset()
            train_metrics_result = {k: y.detach().cpu() for k, y in train_epoch_result.items()}
            train_PCC_sort = sorted(torch.nan_to_num(train_metrics_result['PearsonCorrCoef']))[::-1]
            train_MSE = train_metrics_result['MeanSquaredError'].mean().item()
            train_MAE = train_metrics_result['MeanAbsoluteError'].mean().item()
            train_PCC_10 = np.mean(sorted(train_PCC_sort)[::-1][:10])
            train_PCC_50 = np.mean(sorted(train_PCC_sort)[::-1][:50])
            train_PCC_200 = np.mean(sorted(train_PCC_sort)[::-1][:200])

        if self.is_train_metric:
            metrics_dict = {
                'train_loss' : train_avg_loss,
                'PCC_10' : train_PCC_10,
                'PCC_50' : train_PCC_50,
                'PCC_200' : train_PCC_200,
                'MSE' : train_MSE,
                'MAE' : train_MAE,
                'epoch' : epoch
            }
        else:
            metrics_dict = {
                'train_loss' : train_avg_loss,
                'epoch' : epoch
            }

        return train_avg_loss, metrics_dict
    

    def val_epoch(self, epoch):

        self.model.eval()
        self.val_metrics.reset()

        val_total_loss = 0
        val_total_count = 0

        if self.logger:
            self.logger.info(f"Val Epoch {epoch}")
        for batch in self.val_loader:
            # load data to gpu
            batch = self._preprocess_inputs(batch, dataset=self.val_dataset)
            
            batch_loss, batch_size = self.test_step(batch_data=batch, test_metrics=self.val_metrics)
            val_total_loss += batch_loss
            val_total_count += batch_size

        # logging
        val_avg_loss = val_total_loss / val_total_count
        if self.is_train_metric:
            val_epoch_result = self.val_metrics.compute()
            self.val_metrics.reset()
            val_metrics_result = {k: y.detach().cpu() for k, y in val_epoch_result.items()}
            val_PCC_sort = sorted(torch.nan_to_num(val_metrics_result['val_PearsonCorrCoef']))[::-1]
            val_CCC_sort = sorted(torch.nan_to_num(val_metrics_result['val_ConcordanceCorrCoef']))[::-1]
            val_MSE = val_metrics_result['val_MeanSquaredError'].mean().item()
            val_MAE = val_metrics_result['val_MeanAbsoluteError'].mean().item()
            val_PCC_10 = np.mean(sorted(val_PCC_sort)[::-1][:10])
            val_PCC_50 = np.mean(sorted(val_PCC_sort)[::-1][:50])
            val_PCC_200 = np.mean(sorted(val_PCC_sort)[::-1][:200])
        if self.logger:
            self.logger.info(f"validating epoch: {epoch:8d} val_loss: {val_avg_loss:8.4f}")


        if self.is_train_metric:
            metrics_dict = {
                'val_loss' : val_avg_loss,
                'PCC_10' : val_PCC_10,
                'PCC_50' : val_PCC_50,
                'PCC_200' : val_PCC_200,
                'MSE' : val_MSE,
                'MAE' : val_MAE,
                'epoch' : epoch
            }
        else:
            metrics_dict = {
                'val_loss' : val_avg_loss,
                'epoch' : epoch
            }

        return val_avg_loss, metrics_dict


    def test_epoch(self, epoch, final_test=False):
        self.model.eval()
        self.test_metrics.reset()

        not_generate = True
        
        test_total_loss = 0
        test_total_count = 0

        if self.logger:
            self.logger.info(f"Test Epoch {epoch}")
        inner_pbar = tqdm(self.test_loader, file=TqdmToLoguru(), desc="Testing Step", position=2)
        for batch in inner_pbar:
            # load data to gpu
            batch = self._preprocess_inputs(batch, dataset=self.test_dataset)
            
            batch_loss, batch_size = self.test_step(batch_data=batch, test_metrics=self.test_metrics, final_test=final_test, not_generate=not_generate)
            test_total_loss += batch_loss
            test_total_count += batch_size

        # logging
        test_avg_loss = test_total_loss / test_total_count
        if self.is_train_metric or final_test or not not_generate:
            test_epoch_result = self.test_metrics.compute()
            self.test_metrics.reset()
            test_metrics_result = {k: y.detach().cpu() for k, y in test_epoch_result.items()}
            test_PCC_sort = sorted(torch.nan_to_num(test_metrics_result['test_PearsonCorrCoef']))[::-1]
            test_MSE = test_metrics_result['test_MeanSquaredError'].mean().item()
            test_MAE = test_metrics_result['test_MeanAbsoluteError'].mean().item()
            test_PCC_10 = np.mean(sorted(test_PCC_sort)[::-1][:10])
            test_PCC_50 = np.mean(sorted(test_PCC_sort)[::-1][:50])
            test_PCC_200 = np.mean(sorted(test_PCC_sort)[::-1][:200])

        if self.is_train_metric or final_test or not not_generate:
            metrics_dict = {
                'test_loss' : test_avg_loss,
                'PCC_10' : test_PCC_10,
                'PCC_50' : test_PCC_50,
                'PCC_200' : test_PCC_200,
                'MSE' : test_MSE,
                'MAE' : test_MAE,
                'epoch' : epoch
            }
        else:
            metrics_dict = {
                'test_loss' : test_avg_loss,
                'epoch' : epoch
            }

        return test_avg_loss, metrics_dict


    def train(self):      
        

        for epoch in tqdm(range(self.total_epoch), file=TqdmToLoguru(), desc="Training Epoch", position=1):
            train_avg_loss, train_metrics_dict = self.train_epoch(epoch=epoch)

            val_avg_loss, val_metrics_dict = self.val_epoch(epoch=epoch)

            test_avg_loss, test_metrics_dict = self.test_epoch(epoch=epoch)

            self.early_stopping(val_avg_loss, self.model, epoch, train_metrics_dict, val_metrics_dict, test_metrics_dict, self.ema)
            if self.early_stopping.early_stop:
                break
                    

    def test(self):
        self.model, self.ema = self.early_stopping.load_from_disk(model=self.model, ema=self.ema, device=self.device)
        test_avg_loss, test_metrics_dict = self.test_epoch(epoch=-1, final_test=True)
        with open(os.path.join(args.experiment_result_path, "final_test_result.json"), "w", encoding='utf-8') as f:
            json.dump(test_metrics_dict, f, indent=4, default=json_serializer)
     
        predict_samples = torch.cat(self.predict_samples, dim=0).cpu()
        
        torch.save(predict_samples, os.path.join(self.args.samples_dir, 'predict_samples.pt'))

def main(args):
    folder_list_path = os.path.join(args.process_path, args.folder_list_filename)
    slidename_lst = list(np.genfromtxt(folder_list_path, dtype=str))
    logger = args.logger
    for out in args.slide_out.split(','):
        if out not in slidename_lst:
            continue
        slidename_lst.remove(out)
    # load selected gene list
    gene_list_path = os.path.join(args.process_path, args.gene_list_filename)
    selected_genes = list(np.genfromtxt(gene_list_path, dtype=str))
    input_gene_size = len(selected_genes)
    fix_seed(args.seed)

    # split train and val
    if args.split_dir:
        with open(args.split_dir, 'r', encoding='utf-8') as f:
            dataset_split_json = json.load(f)
    else:
        slidename_length = len(slidename_lst)
        test_length = max(round(slidename_length * 0.1), 1)
        val_length =  max(round(slidename_length * 0.1), 1)
        train_length = slidename_length - test_length - val_length
        split_dir = os.path.join('./split', args.dataset)
        os.makedirs(split_dir, exist_ok=True)
        dataset_split = split_list_numpy(slidename_lst, [train_length, val_length, test_length])
        dataset_split_json = {
            'train_sample' : [str(i) for i in dataset_split[0]],
            'val_sample' : [str(i) for i in dataset_split[1]],
            'test_sample' : [str(i) for i in dataset_split[2]]
        }

        with open(os.path.join(split_dir, args.split_samples_filename), 'w', encoding='utf-8') as f:
            json.dump(dataset_split_json, f, indent=4)

    train_samples = dataset_split_json['train_sample']
    val_samples = dataset_split_json['val_sample']
    test_samples = dataset_split_json['test_sample']

    args.train_samples = train_samples
    args.val_samples = val_samples
    args.test_samples = test_samples
    args.selected_genes = selected_genes
    args.selected_genes_length = input_gene_size


    # load datasets 

    if args.img_pretrained_model == 'uni':
        image_dim=1024
        base_width = 224

        if args.uni_weight_path == None:
            model_image, transform_image = get_encoder(enc_name='uni', device='cpu', assets_dir='./uni_weight', token=args.huggingface_token)
        else:
            model_image, transform_image = get_encoder(enc_name='uni', device='cpu', assets_dir=args.uni_weight_path)
    else:
        raise ValueError()
    
    args.base_width = base_width
    args.image_dim = image_dim

    # save args
    args_dict = {
        k: v for k, v in vars(args).items() 
        if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None)))
    }
    with open(os.path.join(args.experiment_result_path, "config.json"), "w", encoding='utf-8') as f:
        json.dump(args_dict, f, indent=4)

    train_dataset, val_dataset, test_dataset = load_dataset(
        args=args,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        selected_genes=selected_genes,
        logger=logger,
        preprocess=transform_image,
        base_width=base_width
    )
    
    # load dataloader

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, sampler=None, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # load model
    model = load_model(
        args=args,
        selected_genes=selected_genes,
        image_encoder=model_image
    )

    model = model.to(args.device)

    # load optimizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    train_metrics = torchmetrics.MetricCollection([
        PearsonCorrCoef(num_outputs=len(selected_genes)),
        ConcordanceCorrCoef(num_outputs=len(selected_genes)),
        MeanSquaredError(num_outputs = len(selected_genes)),
        MeanAbsoluteError(num_outputs = len(selected_genes)),
        ExplainedVariance() 
    ]).to(args.device)
    test_metrics = train_metrics.clone(prefix = 'test_')
    val_metrics = train_metrics.clone(prefix = 'val_')

    if args.only_test:
        checkpoint_dir = args.checkpoint_path
    else:
        checkpoint_dir = args.checkpoint_dir

    early_stopping = EarlyStopping(
        patience=args.patience,
        save_path=os.path.join(checkpoint_dir, 'best_model.pth'),
        verbose=True,
        logger=logger
    )

    trainer = Trainer(
        model=model,
        model_name=args.model_name,
        optimizer=optimizer,
        scaler=scaler,
        total_epoch=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        device=args.device,
        logger=args.logger,
        tboard=args.tboard,
        early_stopping=early_stopping,
        args=args
    )
    if not args.only_test:
        trainer.train()
    trainer.test()
    args.tboard.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='train')
    parser.add_argument('--data_root', type=str, default='./hest1k_datasets')
    parser.add_argument('--dataset', type=str, default='kidney')
    parser.add_argument('--model_name', type=str, default='hyperst')
    parser.add_argument('--folder_list_filename', type=str, default='all_slide_lst.txt')
    parser.add_argument('--gene_list_filename', type=str, default='selected_gene_list.txt')
    parser.add_argument('--results_dir', type=str, default='./experiments')
    parser.add_argument('--split_dir', type=str, default=None)
    parser.add_argument('--split_samples_filename', type=str, default='sample_split.json')
    parser.add_argument('--experiment_result_path', type=str, default=None)
    parser.add_argument('--img_pretrained_model', type=str, default='uni')

    parser.add_argument('--image_dropout', type=float, default=0.1)

    parser.add_argument('--gene_dropout', type=float, default=0.1)

    parser.add_argument('--emb_dim', type=int, default=1024)
    parser.add_argument('--predict_norm', action='store_true')
    parser.add_argument('--entail_weight', type=float, default=0.4)
    parser.add_argument('--alignment_beta', type=float, default=0.2)

    parser.add_argument('--last_layer', type=int, default=11)
    parser.add_argument('--is_lora_ffn', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_predict_rep', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument("--uni_weight_path", type=str, default=None)
    parser.add_argument('--huggingface_token', type=str, default=None)

    args = parser.parse_args()

    # set up config
    args.data_path = os.path.join(args.data_root, args.dataset)
    args.process_path = os.path.join(args.data_path, 'processed_data')
    args.tif_path = os.path.join(args.data_path, 'wsis')
    args.st_path = os.path.join(args.data_path, 'st')
    args.slide_out = ""
    args.num_aug_ratio = 7
    # generate experiment logs

    args.results_dataset_dir = os.path.join(os.path.join(os.path.join(args.results_dir, args.model_name), args.experiment_name), args.dataset)
    if not args.experiment_result_path:
        os.makedirs(args.results_dataset_dir, exist_ok=True)
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d-%H:%M:%S.%f')
        if args.split_dir:
            split_name = args.split_dir.split('.')[1].split('/')[-1]
            args.experiment_result_path = os.path.join(os.path.join(args.results_dataset_dir, split_name), current_time)
            print(args.experiment_result_path)
        else:
            args.experiment_result_path = os.path.join(args.results_dataset_dir, current_time)
        os.makedirs(args.experiment_result_path)
        
        args.samples_dir = os.path.join(args.experiment_result_path, 'samples')
        os.makedirs(args.samples_dir)
    else:
        args.samples_dir = os.path.join(args.experiment_result_path, 'samples')

    args.checkpoint_dir = os.path.join(args.experiment_result_path, 'checkpoints')


    args.logger =logger
    args.tboard = SummaryWriter(log_dir=args.experiment_result_path)

    args.device = 'cuda:' + str(args.gpu)

    main(args)