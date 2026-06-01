import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import scanpy as sc
import anndata
from scipy.sparse import csr_matrix


from tqdm  import tqdm
import time

import json
from shapely import Polygon
import geopandas as gpd

from pretrained import scgpt_spatial


def get_img_patch_gene_expressions(img, 
                                adata,
                                adata_gene_emb,
                                samplename,
                                tissue_contours,
                                adata_metadata, 
                                target_patch_size,
                                target_pixel_size,
                                threshold=0.15,
                                save_path=None,
                                niche_size = 7,
                                ):

    # KDTree for obtaining niche patch
    from sklearn.neighbors import KDTree




    def valid_wsi(adata, radius, width, height):
        # if in the wsi
        patch_size_src = 2 * radius
        coords_center = adata.obsm['spatial']
        coords_topleft = coords_center - radius
        len_tmp = len(coords_topleft)
        in_slide_mask = (0 <= coords_topleft[:, 0]) & (coords_topleft[:, 0] + patch_size_src < width) & (0 <= coords_topleft[:, 1]) & (coords_topleft[:, 1] + patch_size_src < height)
        return in_slide_mask


    def valid_segmentation(adata, radius, tissue_contours, threshold=0.15):

        # if in the wsi
        patch_size_src = 2 * radius
        coords_center = adata.obsm['spatial']
        coords_topleft = coords_center - radius
        # check the segmentation
        bounding_boxes = tissue_contours.geometry.bounds
        #  if in the segmentation
        bbox_masks = []
        for _, bbox in bounding_boxes.iterrows():
            bbox_mask = (
                (coords_topleft[:, 0] >= bbox['minx'] - patch_size_src) & (coords_topleft[:, 0] <= bbox['maxx'] + patch_size_src) & 
                (coords_topleft[:, 1] >= bbox['miny'] - patch_size_src) & (coords_topleft[:, 1] <= bbox['maxy'] + patch_size_src)
            )
            bbox_masks.append(bbox_mask)
        # 
        if len(bbox_masks) > 0:
            bbox_mask = np.vstack(bbox_masks).any(axis=0)
        else:
            bbox_mask = np.zeros(len(coords_topleft), dtype=bool)


        squares = [
            Polygon([
                (xy[0], xy[1]), 
                (xy[0] + patch_size_src, xy[1]), 
                (xy[0] + patch_size_src, xy[1] + patch_size_src), 
                (xy[0], xy[1] + patch_size_src)]) 
            for xy in coords_topleft[bbox_mask]
        ]

        union_mask = tissue_contours.union_all()
        if threshold == 0:
            valid_mask = gpd.GeoSeries(squares).intersects(union_mask).values
        else:
            gdf = gpd.GeoSeries(squares)
            areas = gdf.area
            valid_mask = gdf.intersection(union_mask).area >= threshold * areas
        full_mask = bbox_mask
        full_mask[bbox_mask] &= valid_mask 
        return full_mask
    

    # process spot

    # pixel size of the patches in um/px after rescaling
    pixel_size = adata_metadata['pixel_size_um_estimated']
    print('Pixel_size (um/px): ', pixel_size)
    spot_target_patch_size = target_patch_size * (target_pixel_size / pixel_size)
    print('Spot_target_patch_size: ', spot_target_patch_size)
    neighbor_target_patch_size = 5 * spot_target_patch_size
    print('Neighbor_target_patch_size: ', neighbor_target_patch_size)
    downsample = target_pixel_size / pixel_size
    print("Downsample: ", downsample)
    # bounding_boxes = tissue_contours.geometry.bounds


    spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    print("Spot diameter: ", spot_diameter)  # Spot diameter for Visium
    if spot_diameter < spot_target_patch_size: 
        radius_spot = int(spot_target_patch_size // 2)                         # minimum patch size: 224 by 224
    else:
        radius_spot = int(spot_diameter // 2)
    print("Radius_spot: ", radius_spot)
    radius_neighbor = int(neighbor_target_patch_size // 2)
    print("Radius_neighbor: ", radius_neighbor)
    spot_radius = int(spot_diameter // 2)
    x = adata.obsm["spatial"][:, 0]          # x coordinate in H&E image
    y = adata.obsm["spatial"][:, 1]          # y coordinate in H&E image

    width, height = img.size

    
    # create kdtree for KNN
    kdtree = KDTree(adata.obsm["spatial"])
    distances, _ = kdtree.query(adata.obsm["spatial"], niche_size)
    radius_ap = np.mean(np.ceil(distances[:, -1]))
    print('Radius_ap: ', radius_ap)

    niche_idx_list = []
    niche_spatial_list = []


    combined_expression_sum_list = []
    combined_expression_mean_list = []
    combined_expression_list = []

    combined_expression_sum_emb_list = []
    combined_expression_mean_emb_list = []
    combined_expression_emb_list = []

    spot_idx_list = []
    spot_spatial_list = []
    spot_neighbors_idx_list = []
    spot_neighbors_spatial_list = []

    first_spot = True
    first = True

    neighbors_idx_list = []
    niche_x_marge_list = []
    niche_y_marge_list = []
    niche_radius_list = []
    is_neighbors_list = []

    spot_image_np_list = []
    niche_image_np_list = []
    neighbor_image_np_list = []

    neighbors_spatials = kdtree.query_radius(adata.obsm["spatial"], r=radius_ap)
    for spot_idx, neighbors_idx in enumerate(neighbors_spatials):
        neighbors_location = adata.obsm["spatial"][neighbors_idx]
        niche_x_marge = np.max(neighbors_location[:,0]) - x[spot_idx]
        niche_y_marge = np.max(neighbors_location[:,1]) - y[spot_idx]
        niche_radius = int(max(niche_x_marge, niche_y_marge)) + spot_radius
        neighbors_idx_list.append(neighbors_idx)
        niche_x_marge_list.append(niche_x_marge)
        niche_y_marge_list.append(niche_y_marge)
        niche_radius_list.append(niche_radius)
        is_neighbors_list.append(len(neighbors_location) == niche_size)
    niche_radius_list = np.array(niche_radius_list)
    niche_radius = round(niche_radius_list[is_neighbors_list].mean())
    print(len(is_neighbors_list))
    print('Niche_radius: ', niche_radius)


    wsi_mask_spot = valid_wsi(adata, radius_spot, width=width, height=height)
    wsi_mask_niche = valid_wsi(adata, radius=niche_radius, width=width, height=height)
    segment_mask_spot = valid_segmentation(adata, radius_spot, tissue_contours, threshold=threshold)
    segment_mask_niche = valid_segmentation(adata, niche_radius, tissue_contours, threshold=threshold)

    inner_loop =  tqdm(range(len(x)), desc=samplename, total=len(x) , position=1, leave=True)
    for spot_idx in inner_loop:

        neighbors_idx = neighbors_idx_list[spot_idx]
        niche_x_marge = niche_x_marge_list[spot_idx]
        niche_y_marge = niche_y_marge_list[spot_idx]
        spot_image = img.crop((x[spot_idx]-radius_spot, y[spot_idx]-radius_spot, 
            x[spot_idx]+radius_spot, y[spot_idx]+radius_spot))
        inner_loop.set_postfix({
        "x": f"{x[spot_idx]:.2f}",
        "y": f"{y[spot_idx]:.2f}",
        "radius_spot": f"{radius_spot:.2f}",
        "radius_niche": f"{niche_radius:.2f}",
        "spot_radius": f"{spot_radius:.2f}",
        "niche_radius": f"{niche_radius:.2f}",
        "neighbors": len(neighbors_idx),
        'niche_x_marge': f"{niche_x_marge:.2f}",
        'niche_y_marge': f"{niche_y_marge:.2f}",
        'wsi_mask_spot': f"{wsi_mask_spot[spot_idx]}",
        'wsi_mask_niche': f"{wsi_mask_niche[spot_idx]}",
        'segment_mask_spot': f"{segment_mask_spot[spot_idx]}",
        'segment_mask_niche': f"{segment_mask_niche[spot_idx]}",
            })

        # spot meta data
        spot_idx_list.append(spot_idx)
        spot_spatial_list.append([x[spot_idx], y[spot_idx]])
        
        # Spot image
        spot_image = img.crop((x[spot_idx]-radius_spot, y[spot_idx]-radius_spot, 
                x[spot_idx]+radius_spot, y[spot_idx]+radius_spot))


        spot_image_np_list.append(np.array(spot_image)) # high, width , channel


        # check the segmentation
        if not wsi_mask_spot[spot_idx] or not wsi_mask_niche[spot_idx] or not segment_mask_spot[spot_idx] or not segment_mask_niche[spot_idx]:
            print(f'{spot_idx} filter out of wsi and segmentation')
            continue
        
        # without the enough neighbors drop
        if len(neighbors_idx) != niche_size:
            print('spot {} {} radius {} finds {} neighbors drop, the neighbors must be {}'.format(x[spot_idx], y[spot_idx], radius_spot, len(neighbors_idx), niche_size))
            continue

        # print('niche_x_marge {} niche_y_marge {} radius_niche {} radius_spot {} radius {}'.format(niche_x_marge, niche_y_marge, radius_niche, radius_spot, radius))
        

        # gene expression

        if isinstance(adata.X, csr_matrix):
            combined_expression = adata[neighbors_idx, :].X.toarray()
            combined_expression_emb = adata_gene_emb[neighbors_idx, :].X.toarray()

        else:
            combined_expression = adata[neighbors_idx, :].X
            combined_expression_emb = adata_gene_emb[neighbors_idx, :].X.toarray()

        combined_expression_sum  = combined_expression.sum(axis=0)
        combined_expression_mean = combined_expression.mean(axis=0)

        combined_expression_emb_sum = combined_expression_emb.sum(axis=0)
        combined_expression_emb_mean = combined_expression_emb.mean(axis=0)


        combined_expression_emb_list.append(combined_expression_emb)
        combined_expression_sum_emb_list.append(combined_expression_emb_sum)
        combined_expression_mean_emb_list.append(combined_expression_emb_mean)


        combined_expression_list.append(combined_expression)
        combined_expression_sum_list.append(combined_expression_sum)
        combined_expression_mean_list.append(combined_expression_mean)

        spot_neighbors_spatial_list.append(adata[neighbors_idx, :].obsm['spatial'])
        spot_neighbors_idx_list.append(neighbors_idx)

        # niche meta data
        niche_idx_list.append(spot_idx)
        niche_spatial_list.append([x[spot_idx], y[spot_idx], radius_spot, niche_radius, spot_radius, niche_radius, niche_x_marge, niche_y_marge, downsample, width, height])
        
        niche_image = img.crop((x[spot_idx]-niche_radius, y[spot_idx]-niche_radius, 
                    x[spot_idx]+niche_radius, y[spot_idx]+niche_radius))
        niche_image_np_list.append(np.array(niche_image))


        # neighbor image
        neighbor_image = img.crop((x[spot_idx]-radius_neighbor, y[spot_idx]-radius_neighbor, 
                        x[spot_idx]+radius_neighbor, y[spot_idx]+radius_neighbor))
        neighbor_image_np_list.append(np.array(neighbor_image))
               

    inner_loop.close()

    combined_expression_sum_list = np.array(combined_expression_sum_list, copy=True)
    combined_expression_mean_list = np.array(combined_expression_mean_list, copy=True)

    
    niche_idx_list = np.array(niche_idx_list)
    niche_spatial_list = np.array(niche_spatial_list)

    spot_idx_list = np.array(spot_idx_list, copy=True)
    spot_spatial_list = np.array(spot_spatial_list, copy=True)

    spot_neighbors_idx_list = np.array(spot_neighbors_idx_list, copy=True)
    spot_neighbors_spatial_list = np.array(spot_neighbors_spatial_list, copy=True)

    spot_image_np_list = np.array(spot_image_np_list, copy=True)
    niche_image_np_list = np.array(niche_image_np_list, copy=True)
    neighbor_image_np_list = np.array(neighbor_image_np_list, copy=True)


    print("Final spatial data size: ", 
            spot_idx_list.shape,
            spot_spatial_list.shape,
            spot_neighbors_idx_list.shape,
            spot_neighbors_spatial_list.shape,
            niche_idx_list.shape,
            niche_spatial_list.shape
        )  
    
    

    print("Final image data size: ", 
            spot_image_np_list.shape,
            niche_image_np_list.shape,
            neighbor_image_np_list.shape
        )
    

    
    if save_path != None:
        # spot and niche meta data
        np.save(save_path + "spot/idx/"   + samplename + "_idx.npy", spot_idx_list)
        np.save(save_path + "spot/spatial/"   + samplename + "_spatial.npy", spot_spatial_list)
        np.save(save_path + "spot/neighbors_idx_list/"   + samplename + "_neighbors_idx.npy", spot_neighbors_idx_list)
        np.save(save_path + "spot/neighbors_spatial_list/"   + samplename + "_neighbors_spatial.npy", spot_neighbors_spatial_list)
        np.save(save_path + "niche/idx/"   + samplename + "_idx.npy", niche_idx_list)
        np.save(save_path + "niche/spatial/"   + samplename + "_spatial.npy", niche_spatial_list)
        # gene count

        np.save(save_path + "niche/neighbors_gene/"   + samplename + "_neighbors_gene.npy", combined_expression_list)
        np.save(save_path + "niche/neighbors_gene_sum/"   + samplename + "_neighbors_gene_sum.npy", combined_expression_sum_list)
        np.save(save_path + "niche/neighbors_gene_mean/"   + samplename + "_neighbors_gene_mean.npy", combined_expression_mean_list)

        # spot images 
        np.save(save_path + "spot/patches/" + samplename + "_patch.npy", spot_image_np_list)

        # niche images
        np.save(save_path + "niche/patches/" + samplename + "_patch.npy", niche_image_np_list)

        # neighor
        np.save(save_path + "neighbor/patches/" + samplename + "_patch.npy", neighbor_image_np_list)

def main(args):
    st_path = args.st_path

    data_path = args.data_path
    tif_path = args.tif_path

    niche_size = args.niche_size
    dataset = args.dataset
    save_path = args.save_path
    meta_path = args.meta_path
    contours_path = args.contours_path

    target_patch_size = args.target_patch_size
    target_pixel_size = args.target_pixel_size

    threshold = args.threshold

    if save_path is None:
        save_path = data_path
        
    vocab = scgpt_spatial.tokenizer.GeneVocab.from_file("./vocab.json")

    # load ST adata
    adata_lst = []
    # fn_lst = os.listdir(st_path) 


    if dataset == 'kidney':
        fn_lst = ["NCBI"+str(i) for i in range(692, 715)]
    elif dataset == 'colorectum':
        fn_lst = ["ZEN"+str(i) for i in range(36, 50)]
    elif dataset == "skin":
        fn_lst = ['NCBI'+str(i) for i in range(460, 527)]
        fn_lst.remove("NCBI499")
        fn_lst.remove("NCBI500")
        fn_lst.remove("NCBI501")
        fn_lst.remove("NCBI502")
        fn_lst.remove("NCBI511")
        fn_lst.remove("NCBI512")
        fn_lst.remove("NCBI513")
        fn_lst.remove("NCBI514")
        for i in range(460, 469):
          fn_lst.remove("NCBI"+str(i)) # 10x
        for i in range(523, 527):
          fn_lst.remove("NCBI"+str(i)) # 40x
    elif dataset == 'lung':
        fn_lst = ['MISC'+str(i) for i in range(17, 33)]



    else:
        raise ValueError('dataset {} must in '.format(dataset))

    first = True
    for fn in fn_lst:
        adata = anndata.read_h5ad(st_path + fn + '.h5ad')
        adata_lst.append(adata)
        if first:
            common_genes = adata.var_names 
            first = False
            print(fn, adata.shape)
            continue
        common_genes = set(common_genes).intersection(set(adata.var_names))
        print(fn, adata.shape, end="\t")

    # keep common genes
    print("Length of common genes: ", len(common_genes))
    common_genes = sorted(list(common_genes))
    common_genes_filt = []
    for i in  common_genes:
        if i in vocab:
            common_genes_filt.append(i)
    common_genes = common_genes_filt

    for fni in range(len(fn_lst)):
        adata = adata_lst[fni].copy()
        adata_lst[fni] = adata[:, common_genes].copy()
        print(fn_lst[fni], " ", adata_lst[fni].shape)
    print("Only keep common genes across the slides.")

    os.makedirs(save_path + "processed_data/", exist_ok=True)

    os.makedirs(save_path + "processed_data/spot/")
    os.makedirs(save_path + "processed_data/spot/idx/")
    os.makedirs(save_path + "processed_data/spot/spatial/")
    os.makedirs(save_path + "processed_data/spot/neighbors_idx_list/")
    os.makedirs(save_path + "processed_data/spot/neighbors_spatial_list/")

    os.makedirs(save_path + "processed_data/spot/patches/")

    os.makedirs(save_path + "processed_data/niche/")
    os.makedirs(save_path + "processed_data/niche/idx/")
    os.makedirs(save_path + "processed_data/niche/spatial/")

    os.makedirs(save_path + "processed_data/niche/patches/")

    os.makedirs(save_path + "processed_data/neighbor/")
    os.makedirs(save_path + "processed_data/neighbor/patches/")

    os.makedirs(save_path + "processed_data/niche/neighbors_gene/")
    os.makedirs(save_path + "processed_data/niche/neighbors_gene_sum/")
    os.makedirs(save_path + "processed_data/niche/neighbors_gene_mean/")

    outer_loop = tqdm(range(len(fn_lst)), desc='sample', position=0)
    for i in outer_loop:
        fn = fn_lst[i]
        outer_loop.set_postfix({"current sample": fn})
        adata = adata_lst[i].copy()
        adata_gene_emb = adata_lst[i].copy()
        print(fn, adata.shape)
        sc.pp.filter_cells(adata, min_genes=1)
        sc.pp.filter_cells(adata_gene_emb, min_genes=1)
        sc.pp.filter_genes(adata_gene_emb, min_cells=1)
        print(fn, adata_gene_emb.shape)

        adata.write(save_path + f"processed_data/{fn}_filter.h5ad")
        with open(meta_path + fn + ".json") as f:  
            adata_metadata = json.load(f)
        with open(contours_path + fn + "_contours.geojson") as f:
            lines = f.read()
            if 'hole' in lines:
                raise ValueError("")
            else:
                tissue_contours = gpd.read_file(contours_path + fn + "_contours.geojson")
        image = Image.open(tif_path + fn + ".tif")
        get_img_patch_gene_expressions(img=image, adata=adata, adata_gene_emb=adata_gene_emb,
                        tissue_contours = tissue_contours,
                        adata_metadata=adata_metadata, 
                        target_patch_size = target_patch_size,
                        target_pixel_size = target_pixel_size,
                        threshold = threshold,
                        samplename = fn,
                        save_path=save_path + "processed_data/", 
                        niche_size=niche_size
                        )
        

        print("#" * 20)
    
    union_hvg = set()
    for fn_idx in range(len(fn_lst)):
        adata = adata_lst[fn_idx].copy()
        fn = fn_lst[fn_idx]
        
        sc.pp.filter_cells(adata, min_genes=1)
        sc.pp.filter_genes(adata, min_cells=1)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)

        union_hvg = union_hvg.union(set(adata.var_names[adata.var["highly_variable"]]))
        print(fn, len(union_hvg))

    union_hvg = sorted([gene for gene in union_hvg if not gene.startswith(("MT", "mt", "RP", "rp"))])
    print(len(union_hvg))
    

    # select union_hvg and concat all slides
    all_count_df = pd.DataFrame(adata_lst[0][:, union_hvg].X.toarray(), 
                                columns=union_hvg, 
                                index=[fn_lst[0] + "_" + str(i) for i in range(adata_lst[0].shape[0])]).T


    for fn_idx in range(1, len(fn_lst)):
        adata = adata_lst[fn_idx]
        df = pd.DataFrame(adata[:, union_hvg].X.toarray(), 
                        columns=union_hvg, 
                        index=[fn_lst[fn_idx] + "_" + str(i) for i in range(adata.shape[0])]).T
        all_count_df = pd.concat([all_count_df, df], axis=1)
        print(fn_lst[fn_idx], adata.shape, all_count_df.shape)

    all_count_df.fillna(0, inplace=True)
    all_count_df = all_count_df.T


    # order selected genes by mean and std
    all_gene_order_by_mean = all_count_df.mean(axis=0).sort_values(ascending=False).index
    all_gene_order_by_std = all_count_df.std(axis=0).sort_values(ascending=False).index


    # select top intersection of high mean and high variance genes

    num_genes = args.num_genes # to make final gene list of length 200

    selected_genes_hvg = sorted(list(set(all_gene_order_by_mean[:num_genes])))

    selected_genes = sorted(list(set(all_gene_order_by_mean[:num_genes]).intersection(set(all_gene_order_by_std[:num_genes]))))
    print(len(selected_genes))

    with open(save_path + "processed_data/selected_gene_list.txt", "w") as f:
        for gene in selected_genes:
            f.write(gene + "\n")

    with open(save_path + "processed_data/selected_hvg_gene_list.txt", "w") as f:
        for gene in selected_genes_hvg:
            f.write(gene + "\n")

    with open(save_path + "processed_data/all_slide_lst.txt", "w") as f:
        for fn in fn_lst:
            f.write(fn + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--dataset", type=str, default="PRAD", help="Dataset")
    parser.add_argument("--num_genes", type=int, default=300)
    parser.add_argument("--scGPT_sptial_weight_path", type=str)
    parser.add_argument("--niche_size", type=int, default=7)
    parser.add_argument("--target_patch_size", type=int, default=224)
    parser.add_argument("--target_pixel_size", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.15)

    parser.add_argument("--scGPT_spatial_batch_size", type=int, default=64)
    parser.add_argument('--save_path', type=str, default=None)

    
    args = parser.parse_args()
    args.data_path = os.path.join(args.data_root, args.dataset) + '/'
    print(args.data_path)
    args.tif_path = args.data_path + 'wsis/' 
    args.st_path = args.data_path + 'st/'
    args.meta_path = args.data_path + 'metadata/'
    args.contours_path = args.data_path + 'tissue_seg/'
    main(args=args)