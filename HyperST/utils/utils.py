import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import wget

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def split_list_numpy(lst, sizes):

    arr = np.array(lst)
    np.random.shuffle(arr)  
    split_indices = np.cumsum(sizes)[:-1] 
    return np.split(arr, split_indices)



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True, 
                 save_path=None, verbose=False, logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.save_path = save_path
        self.save_flod = os.path.dirname(self.save_path)
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_ema_model_state = None
        self.last_model_state = None
        self.last_ema_model_state = None
        self.best_epoch = 0
        self.logger = logger
        
        # Create directory if save_path is provided
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
    def __call__(self, val_loss, model=None, epoch=None, train_metrics_dict=None, val_metrics_dict=None, test_metrics_dict=None, ema=None):
        self._save_last(model, epoch, ema)
        if self.best_loss is None:
            self._update_best(val_loss, model, epoch, ema)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._update_best(val_loss, model, epoch, ema)
            self.counter = 0

    def _save_last(self, model, epoch, ema):

        if model is not None:
            self.last_model_state = model.state_dict().copy()
            if ema is not None:
                self.last_ema_model_state = ema.state_dict().copy()
            if self.save_path is not None and model is not None:
                torch.save(model.state_dict(), os.path.join(self.save_flod, 'final_model.pth'))
                if ema is not None:
                    torch.save(ema.state_dict(), os.path.join(self.save_flod, 'final_ema_model.pth'))  

    def _update_best(self, val_loss, model, epoch, ema):
        improved = self.best_loss is None or val_loss < self.best_loss - self.min_delta
        
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch if epoch is not None else 0
            
            if model is not None:
                
                self.best_model_state = model.state_dict()

                if ema is not None:
                    self.best_ema_model_state = ema.state_dict()
                
                if self.save_path is not None and model is not None:
                    torch.save(model.state_dict(), self.save_path)
                    if ema is not None:
                        torch.save(ema.state_dict(), os.path.join(self.save_flod, 'best_ema_model.pth'))
    
    def load_from_disk(self, model, device, ema=None, best:float=False):

        if self.save_path is not None and os.path.exists(self.save_path):
            if best:
                model.load_state_dict(torch.load(os.path.join(self.save_flod, 'best_model.pth'), map_location=device))
            else:
                model.load_state_dict(torch.load(os.path.join(self.save_flod, 'final_model.pth'), map_location=device))
            
            best_ema_model_path = os.path.join(self.save_flod, 'best_ema_model.pth')
            if os.path.isfile(best_ema_model_path):
                if best:
                    ema.load_state_dict(torch.load(os.path.join(self.save_flod, 'best_ema_model.pth'), map_location=device))
                else:
                    ema.load_state_dict(torch.load(os.path.join(self.save_flod, 'final_ema_model.pth'), map_location=device))
            
            mod = 'best' if best else 'final'
        return model, ema