"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Function set 3.
"""

################### Function set 3 ###################
##### When importing packages, avoid circular imports. #####
import numpy as np
import random
from functions1 import *
from functions2 import *
from model import *


def random_split_data(data, barcode, matrix, num=1500, n=200, seed=42, showInfo=True):
    """
    Randomly divide the data into small datasets of NUM cells per dataset, with N (first N) identical cells between each small dataset.

    :param data: Dataset, dimensions are (number of cells, number of SNPs, 2);
    :param barcode: Barcode List of Cells;;
    :param matrix: Cell-SNP matrix;
    :param num: The number of cells in each sub dataset;
    :param n: The same number of cells in each dataset;
    :param seed: random seed;
    :param showInfo: Do you want to output detailed information;
    :return index: Randomly shuffled idnex;
    :return subset_datas: Collection of sub datasets;
    :return subset_barcodes: Collection of barcode for sub datasets;
    :return subset_mtxs: The set of Cell SNP matrices in the sub dataset.
    """
    data_len = len(barcode)
    # random index
    index = list(range(data_len))
    random.seed(seed)
    random.shuffle(index)
    # Take the first n as the common dataset for each sub dataset
    common_index = index[:n]
    # The approximate data volume of each sub dataset (excluding n)
    near_datasize = num - n
    # Split the data into several parts
    subset_num = int((data_len - n)/near_datasize)
    # Remaining data volume (to be spread evenly across all sub datasets)
    remain_num = (data_len - n) - subset_num*near_datasize
    # The amount of data divided into multiple parts for each dataset
    per_add_num = int(remain_num/subset_num)
    # The data volume of each dataset at the end (with slight differences in the last one)
    datasize = near_datasize + per_add_num
    # showInfo = False
    if showInfo:
        print("big dataset splitting:")
        print("  data size: ", data_len)
        print("  public data size:", n)
        print("  subset approximate size: ", near_datasize)
        print("  subset number: ", subset_num)
        print("  remaining size: ", remain_num)
        print("  subset additional size: ", per_add_num)
        print("  subset final size", datasize)
        print("  last subset final size: ", datasize + remain_num - per_add_num*subset_num)
        print("  check: ", datasize*subset_num + 200 + remain_num - per_add_num*subset_num)
    # subset
    subset_datas = []
    subset_barcodes = []
    subset_mtxs = []
    for i in range(subset_num):
        # last subset
        if i == subset_num - 1:
            index_temp = index[n+i*datasize:]
        else:
            index_temp = index[n+i*datasize:n+(i+1)*datasize]
        subset_temp = data[index_temp]
        barcode_temp = barcode[index_temp]
        mtx_temp = matrix[index_temp]
        subset_datas.append(np.concatenate([data[common_index], subset_temp], axis=0))
        subset_barcodes.append(np.concatenate([barcode[common_index], barcode_temp], axis=0))
        subset_mtxs.append(np.concatenate([matrix[common_index], mtx_temp], axis=0))
    return index, subset_datas, subset_barcodes, subset_mtxs

def get_split_data_group_pred(vae, data, barcode, matrix, cluster, out_path, prefix, num=1500, n=200, img_type="tSNE", seed=42, showInfo=False, saveLoss=False, saveImg=False, plot_center=False, saveZ=False):
    """
    Split the big dataset to obtain prediction results.

    :param vae: Trained VAE model;
    :param data: Dataset, dimensions are (number of cells, number of SNPs, 2);
    :param barcode: Barcode List of Cells;;
    :param matrix: Cell-SNP matrix;
    :param cluster: Number of cell cluster;
    :param out_path: Output Directory;
    :param prefix: The prefix of the output file;
    :param num: The number of cells in each sub dataset;
    :param n: The same number of cells in each dataset;
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param seed: random seed;
    :param showInfo: Do you want to output detailed information;
    :param saveLoss: Whether to save to file, default not;
    :param saveImg: Whether to save to file, default not;
    :param plot_center: Do you want to draw cluster centers;
    :param saveZ: Do you want to save the variables Z for each cell;
    :return final_group_pred: Finally predict results;
    :return result_data: Cell coordinates after tSNE/UMAP dimensionality reduction;
    :return init_center: cluster center;
    :return index: Randomly shuffled index;
    :return index_sorted: Randomly shuffled sorted index.
    """
    index, subset_datas, subset_barcodes, subset_mtxs = random_split_data(data, barcode, matrix, num, n, seed, showInfo)
    subset_preds = []
    subset_losses = [[],[],[]]
    subset_result_datas = []
    # Store the cluster centers of the first subset
    init_centers = None
    
    # Reconstruction and clustering of each sub dataset
    for i in range(len(subset_datas)):
        z, reconX, loss = data_reconX(vae, subset_datas[i])
        group_pred, centers, result_data = get_group_pred(z, cluster, None, img_type, saveImg=False, showInfo=showInfo, plot_center=False)
        # The first subset
        if i == 0:
            init_centers = centers
        subset_preds.append(group_pred)
        subset_result_datas.append(result_data)
        subset_losses[0].append(loss.loss)
        subset_losses[1].append(loss.reconLoss)
        subset_losses[2].append(loss.KLLoss)

    final_group_pred = None
    final_result_data = None
    final_losses = [[], [], []]
    
    # Match the predicted results of the subsets (based on the first n)
    for i in range(len(subset_datas)):
        new_group_pred, comb = class2class(subset_preds[i][:n], subset_preds[0][:n], cluster, showInfo=showInfo)
        new_group_pred = conv_code(subset_preds[i], comb, cluster)
        if i == 0:
            final_group_pred = new_group_pred
            result_data = subset_result_datas[0]
        else:
            final_group_pred = np.concatenate((final_group_pred, new_group_pred[n:]))
            result_data = subset_result_datas[i][n:]
        
        # Modify the centers of each subsequent class to be the same as init_centers, to facilitate the subsequent drawing of clustering maps
        for g in range(1, cluster+1):
            if i == 0:
                result_data_g = result_data[new_group_pred == g]
            else:
                result_data_g = result_data[new_group_pred[n:] == g]
            # Find the current cluster center of this class
            center = np.mean(result_data_g, axis=0)
            # The clustering center of the first subset of this class
            init_center = init_centers[g-1]
            gap = init_center - center
            # Change to the same cluster center
            result_data_g = result_data_g + gap
            # Assign back to the original data
            if i == 0:
                result_data[new_group_pred == g] = result_data_g
            else:
                result_data[new_group_pred[n:] == g] = result_data_g
            
        if i == 0:
            subset_result_datas[i] = result_data
            final_result_data = subset_result_datas[i]
            final_losses[0] = subset_losses[0][i]
            final_losses[1] = subset_losses[1][i]
            final_losses[2] = subset_losses[2][i]
        else:
            subset_result_datas[i][n:] = result_data
            final_result_data = np.concatenate((final_result_data, subset_result_datas[i][n:]))
            final_losses[0] = final_losses[0] + subset_losses[0][i][n:] 
            final_losses[1] = final_losses[1] + subset_losses[1][i][n:] 
            final_losses[2] = final_losses[2] + subset_losses[2][i][n:] 
    # Sort, adjust the random sequence to its original order
    index_dict = {k:v for k,v in zip(list(range(len(index))), index)}
    index_sorted = sorted(index_dict.items(), key=lambda x:x[1])
    index_sorted = [x[0] for x in index_sorted]
    final_group_pred = final_group_pred[index_sorted]
    final_result_data = final_result_data[index_sorted]
    
    # if saveImg:
    #     plot_cluster_img(final_group_pred, final_result_data, init_centers, f"{out_path}/imgs/{prefix}{img_type}_pred.png", img_type, saveImg, showInfo, plot_center)

    # if saveZ:
    #     np.savetxt(f"{out_path}/results/{prefix}{img_type}_result_data.txt", final_result_data, delimiter="\t")
    
    # Create loss storage object
    final_loss = LossHistory()
    final_loss.loss = np.array(final_losses[0])[index_sorted].tolist()
    final_loss.reconLoss = np.array(final_losses[1])[index_sorted].tolist()
    final_loss.KLLoss = np.array(final_losses[2])[index_sorted].tolist()
    # Output loss information
    print_loss(final_loss, out_path+"/results/", prefix=prefix, saveLoss=saveLoss, showInfo=showInfo)
    # Draw a line chart of losses
    plot_loss(final_loss, out_path+"/imgs/", prefix=prefix, saveImg=saveLoss)
    return final_group_pred, final_result_data, init_center, index, index_sorted
        
def get_all_data_group_pred(vae, data, barcode, matrix, cluster, out_path, prefix, img_type="tSNE", showInfo=False, saveLoss=False, saveImg=False, plot_center=True, saveZ=False):
    """
    The small dataset to obtain prediction results.

    :param vae: Trained VAE model;
    :param data: Dataset, dimensions are (number of cells, number of SNPs, 2);
    :param barcode: Barcode List of Cells;;
    :param matrix: Cell-SNP matrix;
    :param cluster: Number of cell cluster;
    :param out_path: Output Directory;
    :param prefix: The prefix of the output file;
    :param num: The number of cells in each sub dataset;
    :param n: The same number of cells in each dataset;
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param showInfo: Do you want to output detailed information;
    :param saveLoss: Whether to save to file, default not;
    :param saveImg: Whether to save to file, default not;
    :param plot_center: Do you want to draw cluster centers;
    :param saveZ: Do you want to save the variables Z for each cell;
    :return group_pred: finally predict results;
    :return result_data: Cell coordinates after tSNE/UMAP dimensionality reduction;
    :return center: cluster center.
    """
    z, reconX, loss = data_reconX(vae, data)
    # Output loss information
    print_loss(loss, out_path+"/results/", prefix=prefix, saveLoss=saveLoss, showInfo=showInfo)
    # Draw a line chart of losses
    plot_loss(loss, out_path+"/imgs/", prefix=prefix, saveImg=saveLoss)
    group_pred, centers, result_data = get_group_pred(z, cluster, f"{out_path}/imgs/{prefix}", img_type, saveImg=False, plot_center=plot_center)

    # if saveZ:
    #     np.savetxt(f"{out_path}/results/{prefix}{img_type}_result_data.txt", result_data, delimiter="\t")
        
    return group_pred, result_data, centers


def result(vae, data, barcode, matrix, cluster, out_path, prefix, num=1500, n=200, img_type="tSNE", seed=42, showInfo=False, saveLoss=False, saveImg=False, plot_center=True, saveZ=False):
    """
    Apply the model to a specified dataset for prediction.

    :param vae: Trained VAE model;
    :param data: Dataset, dimensions are (number of cells, number of SNPs, 2);
    :param barcode: Barcode List of Cells;;
    :param matrix: Cell-SNP matrix;
    :param cluster: Number of cell cluster;
    :param out_path: Output Directory;
    :param prefix: The prefix of the output file;
    :param num: The number of cells in each sub dataset;
    :param n: The same number of cells in each dataset;
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param seed: random seed;
    :param showInfo: Do you want to output detailed information;
    :param saveLoss: Whether to save to file, default not;
    :param saveImg: Whether to save to file, default not;
    :param plot_center: Do you want to draw cluster centers;
    :param saveZ: Do you want to save the variables Z for each cell;
    :return group_pred: finally predict results;
    :return group_pred_plus: The result of using Cell-SNP matrix for re prediction.
    """
    # the big dataset
    if len(barcode) >= num*2 - n:
        # print("############# Split data.")
        group_pred, result_data, centers, _ = get_split_data_group_pred(vae, data, barcode, matrix, cluster, out_path, prefix, num, n, img_type, seed, showInfo, saveLoss, saveImg, plot_center, saveZ)
    else: # the small dataset
        # print("############# Do not split data.")
        group_pred, result_data, centers = get_all_data_group_pred(vae, data, barcode, matrix, cluster, out_path, prefix, img_type, showInfo, saveLoss, saveImg, plot_center, saveZ)

    ############################
    ## Save prediction results
    # save_pred(barcode, group_pred, f"{out_path}/results/{prefix}{img_type}_pred.txt")

    ## using Cell-SNP matrix for re prediction
    #group_pred_plus = get_group_pred_plus(matrix, group_pred, 2)
    ## save the result of using Cell-SNP matrix for re prediction
    #save_pred_plus(barcode, group_pred, group_pred_plus, f"{out_path}/results/{prefix}{img_type}_pred_regroup.txt")
    ############################

    ############################
    # Save prediction results
    group_pred_plus = get_group_pred_plus(matrix, group_pred, 5)
    save_pred(barcode, group_pred_plus, f"{out_path}/results/{prefix}{img_type}_pred.txt")

    new_result_data = modify_map(group_pred, group_pred_plus, result_data, cluster)
    if saveImg:
        plot_cluster_img(group_pred_plus, new_result_data, centers, f"{out_path}/imgs/{prefix}{img_type}_pred.png", img_type, saveImg, showInfo, plot_center)

    if saveZ:
        np.savetxt(f"{out_path}/results/{prefix}{img_type}_result_data.txt", new_result_data, delimiter="\t")
    ############################

    #return group_pred, group_pred_plus
    return group_pred_plus, None

