import os

import numpy as np 
import pandas as pd
import torch
import anndata
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from PIL import Image
class HyperSTDataset(Dataset):
    def __init__(
        self,
        slidename_lst:list,
        selected_genes:list,
        data_path:str,
        process_path:str,
        num_aug_ratio:int=7,
        phase:str='train',
        img_pretrained_model:str='uni',
        logger=None,
        preprocess=None,
        base_width:int=224
    ):
        self.img_pretrained_model = img_pretrained_model
        self.num_aug_ratio = num_aug_ratio
        self.random_num = num_aug_ratio + 1
        self.phase = phase
        self.process_path = process_path
        self.slidename_lst = slidename_lst 


        self.preprocess = preprocess

        self.base_width = base_width

        # load spot patch
        first_slide=True
        all_spot_patch_list = []
        all_niche_patch_list = []

        all_spot_count_mtx_ori = None


        slice_niche_idx_list = []

        slice_niche_idx_keep_list = []

        if logger:
            logger.info("Loading original data...")
        for sni in range(len(slidename_lst)):
            sample_name = slidename_lst[sni]
            test_adata = anndata.read_h5ad(process_path + "/" + sample_name + "_filter.h5ad")
            test_count_mtx = pd.DataFrame(test_adata[:, selected_genes].X.toarray(), 
                                            columns=selected_genes, 
                                            index=[i for i in range(test_adata.shape[0])])
            

            if first_slide:

                niche_idx_list = np.load(process_path + "/niche/idx/" + sample_name + "_idx.npy")
                slide_niche_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :]
                slide_niche_count_mtx_ori_df = pd.DataFrame(slide_niche_count_mtx_ori.values, columns=selected_genes, index=list(range(slide_niche_count_mtx_ori.shape[0])))

                # remove the spot with all NAN/zero in count mtx
                all_count_mtx_all_nan_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.isnull().all(axis=1)]
                all_count_mtx_all_zero_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.sum(axis=1) == 0]
                niche_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
                niche_idx_to_keep = list(set(slide_niche_count_mtx_ori_df.index) - set(niche_idx_to_remove))
                if logger:
                    logger.info(f"{sample_name} loaded All NAN spot index: {niche_idx_list[all_count_mtx_all_nan_spot_index]}")
                    logger.info(f"{sample_name} loaded All zero spot index: {niche_idx_list[all_count_mtx_all_zero_spot_index]}")
                niche_idx_list = niche_idx_list[niche_idx_to_keep]

                slice_niche_idx_list.append(niche_idx_list)
                slice_niche_idx_keep_list.append(niche_idx_to_keep)
                all_spot_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :].values



                # gene expression
                niche_mean_gene_mtx = pd.DataFrame(np.load(process_path + '/niche/neighbors_gene_mean/' + sample_name + '_neighbors_gene_mean.npy')[niche_idx_to_keep, :], columns=test_adata.var_names).loc[:, selected_genes].values

                all_niche_mean_gene_mtx_ori = niche_mean_gene_mtx

                slice_spot_patch = np.load(self.process_path + "/spot/patches/" + sample_name + "_patch.npy")[niche_idx_list]
                all_spot_patch_list.append(slice_spot_patch)
                slice_niche_patch = np.load(self.process_path + "/niche/patches/" + sample_name + "_patch.npy")[niche_idx_to_keep]
                all_niche_patch_list.append(slice_niche_patch)

                if logger:
                    logger.info(f"{sample_name} loaded, origin shape: {test_adata.shape} | spot gene count_mtx shape: {all_spot_count_mtx_ori.shape} | niche gene mtx shape: {all_niche_mean_gene_mtx_ori.shape}")
                first_slide = False
                continue


            
            niche_idx_list = np.load(process_path + "/niche/idx/" + sample_name + "_idx.npy")
            slide_niche_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :]
            slide_niche_count_mtx_ori_df = pd.DataFrame(slide_niche_count_mtx_ori.values, columns=selected_genes, index=list(range(slide_niche_count_mtx_ori.shape[0])))

            # remove the spot with all NAN/zero in count mtx
            all_count_mtx_all_nan_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.isnull().all(axis=1)]
            all_count_mtx_all_zero_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.sum(axis=1) == 0]
            niche_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
            niche_idx_to_keep = list(set(slide_niche_count_mtx_ori_df.index) - set(niche_idx_to_remove))

            if logger:
                logger.info(f"{sample_name} loaded All NAN spot index: {niche_idx_list[all_count_mtx_all_nan_spot_index]}")
                logger.info(f"{sample_name} loaded All zero spot index: {niche_idx_list[all_count_mtx_all_zero_spot_index]}")

            niche_idx_list = niche_idx_list[niche_idx_to_keep]

            slice_niche_idx_list.append(niche_idx_list)
            slice_niche_idx_keep_list.append(niche_idx_to_keep)

            slice_spot_patch = np.load(self.process_path + "/spot/patches/" + sample_name + "_patch.npy")[niche_idx_list]
            all_spot_patch_list.append(slice_spot_patch)

            slice_niche_patch = np.load(self.process_path + "/niche/patches/" + sample_name + "_patch.npy")[niche_idx_to_keep]
            all_niche_patch_list.append(slice_niche_patch)


            # gene expression
            niche_mean_gene_mtx = pd.DataFrame(np.load(process_path + '/niche/neighbors_gene_mean/' + sample_name + '_neighbors_gene_mean.npy')[niche_idx_to_keep, :], columns=test_adata.var_names).loc[:, selected_genes].values

            all_niche_mean_gene_mtx_ori = np.concatenate((all_niche_mean_gene_mtx_ori, niche_mean_gene_mtx), axis=0)


            slide_niche_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :].values
            all_spot_count_mtx_ori = np.concatenate((all_spot_count_mtx_ori, slide_niche_count_mtx_ori), axis=0)
    
            if logger:
                logger.info(f"{sample_name} loaded, origin shape: {test_adata.shape} | spot gene count_mtx shape: {all_spot_count_mtx_ori.shape} | niche gene mtx shape: {all_niche_mean_gene_mtx_ori.shape}")
                logger.info(f"{sample_name} loaded, spot img ebd shape: {slice_spot_patch.shape}, all spot img ebd shape: {len(all_spot_patch_list)}")
                logger.info(f"{sample_name} loaded, niche img ebd shape: {slice_niche_patch.shape}, all spot img ebd shape: {len(all_niche_patch_list)}")
        


        all_spot_gene_count_mtx = all_spot_count_mtx_ori
        all_niche_gene_count_mtx = all_niche_mean_gene_mtx_ori

        if logger:
            logger.info(f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final spot gene_count_mtx shape: {all_spot_gene_count_mtx.shape} | final niche gene_count_mtx  shape: {all_niche_gene_count_mtx.shape}")


        # only normalized by log2(+1)
        all_spot_count_mtx_selected_genes = np.log2(all_spot_gene_count_mtx + 1).copy()
        all_niche_count_mtx_selected_genes = np.log2(all_niche_gene_count_mtx + 1).copy()

        if logger:
            logger.info(f"Selected spot genes count matrix shape: {all_spot_count_mtx_selected_genes.shape} niche genes count matrix shape: {all_niche_count_mtx_selected_genes.shape}" )

        self.sample_lengths = [len(i) for i in slice_niche_idx_list]
        self.cumlen = np.cumsum(self.sample_lengths )
        self.slice_niche_idx_list = slice_niche_idx_list

        self.all_spot_count_mtx_selected_genes = torch.from_numpy(all_spot_count_mtx_selected_genes).float().contiguous()
        self.all_niche_count_mtx_selected_genes = torch.from_numpy(all_niche_count_mtx_selected_genes).float().contiguous()

        # idx list
        self.sample_lengths = [len(i) for i in slice_niche_idx_list]
        self.cumlen = np.cumsum(self.sample_lengths )
        self.slice_niche_idx_list = slice_niche_idx_list
        self.slice_niche_idx_keep_list = slice_niche_idx_keep_list
        self.all_spot_patch_list = all_spot_patch_list
        self.all_niche_patch_list = all_niche_patch_list


    def transform(self, patch, preprocess, base_width:int):
        return preprocess(patch.resize((base_width, base_width), Image.Resampling.LANCZOS))
    
    def __len__(self):
        return len(self.all_spot_count_mtx_selected_genes)
    
    def __getitem__(self, index):
        data = {}

        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]

        sample_name = self.slidename_lst[i]
        spot_patch_idx = self.slice_niche_idx_list[i][idx]

        aug_idx = np.random.randint(0, self.random_num)

        data['label'] = self.all_spot_count_mtx_selected_genes[index].clone()

        spot_image = Image.fromarray(self.all_spot_patch_list[i][idx])
        niche_image = Image.fromarray(self.all_niche_patch_list[i][idx])
        if self.phase == 'train':
            if aug_idx > 0:
                data['spot_img'] = self.transform(spot_image.transpose(aug_idx - 1), self.preprocess, self.base_width)
                data['niche_image'] = self.transform(niche_image.transpose(aug_idx - 1), self.preprocess, self.base_width)
            else:
                data['spot_img'] = self.transform(spot_image, self.preprocess, self.base_width)
                data['niche_image'] = self.transform(niche_image, self.preprocess, self.base_width)
        else:
            data['spot_img'] = self.transform(spot_image, self.preprocess, self.base_width)
            data['niche_image'] = self.transform(niche_image, self.preprocess, self.base_width)
        data['spot_gene_ebd'] = self.all_spot_count_mtx_selected_genes[index]
        data['niche_gene_ebd'] = self.all_niche_count_mtx_selected_genes[index]
        return data
    
class HyperSTGeneADataset(Dataset):
    def __init__(
        self,
        slidename_lst:list,
        selected_genes:list,
        data_path:str,
        process_path:str,
        num_aug_ratio:int=7,
        phase:str='train',
        img_pretrained_model:str='uni',
        logger=None
    ):
        self.img_pretrained_model = img_pretrained_model
        self.num_aug_ratio = num_aug_ratio
        self.random_num = num_aug_ratio + 1
        self.phase = phase
        self.process_path = process_path
        self.slidename_lst = slidename_lst 

        # load original patches
        first_slide = True

        all_spot_count_mtx_ori = None


        slice_niche_idx_list = []

        slice_niche_idx_keep_list = []

        if logger:
            logger.info("Loading original data...")
        for sni in range(len(slidename_lst)):
            sample_name = slidename_lst[sni]
            test_adata = anndata.read_h5ad(process_path + "/" + sample_name + "_filter.h5ad")
            test_count_mtx = pd.DataFrame(test_adata[:, selected_genes].X.toarray(), 
                                            columns=selected_genes, 
                                            index=[i for i in range(test_adata.shape[0])])
            
            if first_slide:

                niche_idx_list = np.load(process_path + "/niche/idx/" + sample_name + "_idx.npy")
                slide_niche_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :]
                slide_niche_count_mtx_ori_df = pd.DataFrame(slide_niche_count_mtx_ori.values, columns=selected_genes, index=list(range(slide_niche_count_mtx_ori.shape[0])))

                # remove the spot with all NAN/zero in count mtx
                all_count_mtx_all_nan_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.isnull().all(axis=1)]
                all_count_mtx_all_zero_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.sum(axis=1) == 0]
                niche_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
                niche_idx_to_keep = list(set(slide_niche_count_mtx_ori_df.index) - set(niche_idx_to_remove))
                if logger:
                    logger.info(f"{sample_name} loaded All NAN spot index: {niche_idx_list[all_count_mtx_all_nan_spot_index]}")
                    logger.info(f"{sample_name} loaded All zero spot index: {niche_idx_list[all_count_mtx_all_zero_spot_index]}")
                niche_idx_list = niche_idx_list[niche_idx_to_keep]

                slice_niche_idx_list.append(niche_idx_list)
                slice_niche_idx_keep_list.append(niche_idx_to_keep)
                all_spot_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :].values



                # gene expression
                niche_mean_gene_mtx = pd.DataFrame(np.load(process_path + '/niche/neighbors_gene_mean/' + sample_name + '_neighbors_gene_mean.npy')[niche_idx_to_keep, :], columns=test_adata.var_names).loc[:, selected_genes].values

                all_niche_mean_gene_mtx_ori = niche_mean_gene_mtx

                
                if logger:
                    logger.info(f"{sample_name} loaded, origin shape: {test_adata.shape} | spot gene count_mtx shape: {all_spot_count_mtx_ori.shape} | niche gene mtx shape: {all_niche_mean_gene_mtx_ori.shape}")
                first_slide = False
                continue
            
            niche_idx_list = np.load(process_path + "/niche/idx/" + sample_name + "_idx.npy")
            slide_niche_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :]
            slide_niche_count_mtx_ori_df = pd.DataFrame(slide_niche_count_mtx_ori.values, columns=selected_genes, index=list(range(slide_niche_count_mtx_ori.shape[0])))

            # remove the spot with all NAN/zero in count mtx
            all_count_mtx_all_nan_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.isnull().all(axis=1)]
            all_count_mtx_all_zero_spot_index = slide_niche_count_mtx_ori_df.index[slide_niche_count_mtx_ori_df.sum(axis=1) == 0]
            niche_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
            niche_idx_to_keep = list(set(slide_niche_count_mtx_ori_df.index) - set(niche_idx_to_remove))

            if logger:
                logger.info(f"{sample_name} loaded All NAN spot index: {niche_idx_list[all_count_mtx_all_nan_spot_index]}")
                logger.info(f"{sample_name} loaded All zero spot index: {niche_idx_list[all_count_mtx_all_zero_spot_index]}")

            niche_idx_list = niche_idx_list[niche_idx_to_keep]

            slice_niche_idx_list.append(niche_idx_list)
            slice_niche_idx_keep_list.append(niche_idx_to_keep)


            # gene expression
            niche_mean_gene_mtx = pd.DataFrame(np.load(process_path + '/niche/neighbors_gene_mean/' + sample_name + '_neighbors_gene_mean.npy')[niche_idx_to_keep, :], columns=test_adata.var_names).loc[:, selected_genes].values

            all_niche_mean_gene_mtx_ori = np.concatenate((all_niche_mean_gene_mtx_ori, niche_mean_gene_mtx), axis=0)


            slide_niche_count_mtx_ori = test_count_mtx.loc[niche_idx_list, :].values
            all_spot_count_mtx_ori = np.concatenate((all_spot_count_mtx_ori, slide_niche_count_mtx_ori), axis=0)
    
            if logger:
                logger.info(f"{sample_name} loaded, origin shape: {test_adata.shape} | spot gene count_mtx shape: {all_spot_count_mtx_ori.shape} | niche gene mtx shape: {all_niche_mean_gene_mtx_ori.shape}")




        all_spot_gene_count_mtx = all_spot_count_mtx_ori
        all_niche_gene_count_mtx = all_niche_mean_gene_mtx_ori

        if logger:
            logger.info(f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final spot gene_count_mtx shape: {all_spot_gene_count_mtx.shape} | final niche gene_count_mtx  shape: {all_niche_gene_count_mtx.shape}")


        # only normalized by log2(+1)
        all_spot_count_mtx_selected_genes = np.log2(all_spot_gene_count_mtx + 1).copy()
        all_niche_count_mtx_selected_genes = np.log2(all_niche_gene_count_mtx + 1).copy()

        if logger:
            logger.info(f"Selected spot genes count matrix shape: {all_spot_count_mtx_selected_genes.shape} niche genes count matrix shape: {all_niche_count_mtx_selected_genes.shape}" )

        self.sample_lengths = [len(i) for i in slice_niche_idx_list]
        self.cumlen = np.cumsum(self.sample_lengths )
        self.slice_niche_idx_list = slice_niche_idx_list

        self.all_spot_count_mtx_selected_genes = torch.from_numpy(all_spot_count_mtx_selected_genes).float().contiguous()
        self.all_niche_count_mtx_selected_genes = torch.from_numpy(all_niche_count_mtx_selected_genes).float().contiguous()

        # idx list
        self.sample_lengths = [len(i) for i in slice_niche_idx_list]
        self.cumlen = np.cumsum(self.sample_lengths )
        self.slice_niche_idx_list = slice_niche_idx_list
        self.slice_niche_idx_keep_list = slice_niche_idx_keep_list

    def __len__(self):
        return len(self.all_spot_count_mtx_selected_genes)
    
    def __getitem__(self, index):
        data = {}

        # i = 0
        # while index >= self.cumlen[i]:
        #     i += 1
        # idx = index
        # if i > 0:
        #     idx = index - self.cumlen[i-1]

        # sample_name = self.slidename_lst[i]
        # spot_patch_idx = self.slice_niche_idx_list[i][idx]
        data['label'] = self.all_spot_count_mtx_selected_genes[index].clone()
        data['spot_gene_ebd'] = self.all_spot_count_mtx_selected_genes[index]
        data['niche_gene_ebd'] = self.all_niche_count_mtx_selected_genes[index]
        return data