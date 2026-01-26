"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Function set 4 (doublet).
"""

################### Function set 4 ###################
##### When importing packages, avoid circular imports. #####
import numpy as np
import matplotlib.pyplot as plt
from functions1 import plot_cluster_img
import numbers


def deal_doublet(data, group_pred, doublet_log, combination, barcodes, is_percent=True, doublet_num=None, saveInfo=True, append=False):
    """
    Input the Cell-SNP matrix and (optionally) a predefined doublet ratio to obtain the list of doublet barcodes and the prediction results with doublets labeled as 0.

    :param data: Cell-SNP matrix;
    :param group_pred: The cluster labels of each sample (predicted results);
    :param doublet_log: Log file of doublet prediction;
    :param combination: Array of pairwise combinations of each cluster;
    :param barcodes: Barcode List of Cells;
    :param is_percent: Is the percentage of doublets provided;
    :param doublet_num: Number of doublets;
    :param saveInfo: Is the log file saved;
    :param append: Is the result file appended;
    :return new_group_pred: The cluster labels of each sample (predicted results, including doublet);
    :return doublet_barcodes: Doublet barcode List of Cells;
    :return doublet_combs: Source sample of the doublets.
    """
    cluster = len(np.unique(group_pred))
    # Doublet barcode List of Cells
    doublet_barcodes = set()
    # Source sample of the doublets
    doublet_combs = group_pred.copy().tolist()
    # Store the similarity of each cell to a single cluster and to pairs of clusters (the first 'cluster' entries correspond to singlets, and the remaining entries correspond to doublets).
    similarity_mtx = np.zeros((len(group_pred), cluster+len(combination)),dtype=np.float64)
    new_group_pred = group_pred.copy()
    # Record the runtime log
    doublet_logs = []
    n = 0
    while True:
        n += 1
        log = "Round  {}:".format(n)
        doublet_logs.append(log)
        print(log)
        # Store the barcodes of doublets identified in this round
        temp_doublet_barcodes = set()
        # Skip cells previously identified as doublets
        temp_single = new_group_pred != 0
        _, percent_mtx, common = common_seq(data[temp_single,:], new_group_pred[temp_single])
        mask = cluster_mask(common)
        # Generate simulated doublet classes based on cluster combinations
        percent_groups = get_percent_groups(percent_mtx, combination)
        for i in range(len(new_group_pred)):
            # In non-first rounds, if the value is 0, it indicates the cell was previously identified as a doublet and should be skipped
            if new_group_pred[i] == 0:
                continue
            # Compute the similarity between the current cell and each class (including simulated doublet classes)
            for j in range(len(percent_groups)):
                similarity_mtx[i,j] = similarity_mask(data[i,:], percent_groups[j], mask)
        # Extract the similarity vectors of singlet cells belonging to each class
        single_percent = similarity_mtx[np.arange(len(new_group_pred)), group_pred-1]
        # Calculate the lower bound of 3σ (three standard deviations below the mean)
        right_boundary = right_three_sigma(single_percent[temp_single])
        for i in range(len(new_group_pred)):
            # Determine whether it is a double cell
            if temp_single[i] and single_percent[i] <= right_boundary:
                # Find out if the double cell is composed of two classes (the previous cluster is a single cell class)
                doublet_groups = similarity_mtx[i, cluster:]
                max_index = np.argmax(doublet_groups)
                # What are the two types of cells that make up this cell
                doublet_combs[i] = combination[max_index]+1
                # Record the barcode
                temp_doublet_barcodes.add(barcodes[i])
                new_group_pred[i] = 0
        temp_num = np.sum(single_percent[temp_single] <= right_boundary)
        
        # If the ratio of double cells has been set and there are no cells exceeding 3 σ or meeting the required quantity, which one should be taken directly
        if is_percent and temp_num == 0 and len(doublet_barcodes) < doublet_num:  
            need_num = int(doublet_num) - len(doublet_barcodes)
            # Take out the cells that have not yet been determined as double cells, arrange them in ascending order, and take the previous need_num as double cells
            right_boundary = np.sort(single_percent[temp_single])[need_num-1]
            for i in range(len(new_group_pred)):
                # Determine whether it is a double cell
                if temp_single[i] and single_percent[i] <= right_boundary:
                    # Find out if the double cell is composed of two classes (the previous cluster is a single cell class)
                    doublet_groups = similarity_mtx[i, cluster:]
                    max_index = np.argmax(doublet_groups)
                    # What are the two types of cells that make up this cell
                    doublet_combs[i] = combination[max_index]+1
                    # Record the barcode
                    temp_doublet_barcodes.add(barcodes[i])
                    new_group_pred[i] = 0
        doublet_barcodes = doublet_barcodes.union(temp_doublet_barcodes)
        log = "Total number of doublets: {}\tNewly identified in this round: {}".format(len(doublet_barcodes), len(temp_doublet_barcodes))
        doublet_logs.append(log)
        print(log)
        # The number of double cells has reached the required level, exit
        if is_percent and len(doublet_barcodes) >= doublet_num:
            break
            
        # Do not set a percentage, use 3 σ directly for judgment
        if not is_percent and temp_num == 0:
            break
    
    doublet_barcodes = np.array(list(doublet_barcodes))
    log = "Final number of identified doublets: {}\nNumber of prediction rounds: {}".format(len(doublet_barcodes), n)
    doublet_logs.append(log)
    print(log)
    if saveInfo:
        # Output the running logs to a file
        if append:
            with open(doublet_log, "a") as log:
                for i in range(len(doublet_logs)):
                    log.write(doublet_logs[i] + "\n")
                # log.write("\n")
        else:
             with open(doublet_log, "w") as log:
                for i in range(len(doublet_logs)):
                    log.write(doublet_logs[i] + "\n")
                # log.write("\n")

    return new_group_pred, doublet_barcodes, doublet_combs



def plot_cluster_doublet_img(group_label, dim_data, centers, path="", img_type="tSNE", saveImg=False, showInfo=True, plot_center=True):
    """
    Draw a clustering diagram with two-dimensional data obtained from tSNE/UMAP(doublet is red).

    :param group_label: The cluster labels of each sample (predicted results);
    :param dim_data: Reduced dimensional data;
    :param centers: cluster center;
    :param path: The save path of the file (including file name);
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param saveImg: Whether to save to file;
    :param showInfo: Do you want to output detailed information;
    :param plot_center: Do you want to draw cluster centers.
    """
    img = None
    if showInfo | True:
        img, ax = plt.subplots()
        # Cancel the upper and right border lines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel(img_type + "_1")
        plt.ylabel(img_type + "_2")
        plt.title(img_type)

        cols = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628","#F781BF"]
        # Add legend (at this time, each category needs to be drawn separately)
        cluster = len(np.unique(group_label))
        # have doublet
        if np.any(group_label==0): 
            cluster = cluster
        else:
            cluster += 1
        for c in range(1, cluster):
            temp_data = dim_data[group_label==c,:]
            plt.scatter(temp_data[:, 0], temp_data[:, 1], c=cols[c], label="cluster"+str(c), s=10)
        # doublet is red
        doublet_data = dim_data[group_label==0,:]
        plt.scatter(doublet_data[:, 0], doublet_data[:, 1], c="#FF0000", label="doublet", s=10)
        plt.legend(title="Cluster", loc="center right", bbox_to_anchor=(1.2, 0.5), frameon=False)
        if plot_center:
            plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100)
        # plt.show()
    if saveImg:
        img.savefig(path, bbox_inches='tight')
        img.savefig(path.rstrip("png") + "pdf", bbox_inches='tight')

def plot_doublet_img(group_label, dim_data, centers, path="", img_type="tSNE", saveImg=False, showInfo=True, plot_center=True):
    """
    Draw a clustering diagram with two-dimensional data obtained from tSNE/UMAP(doublet is red, other is grey).
    
    :param group_label: The cluster labels of each sample (predicted results);
    :param dim_data: Reduced dimensional data;
    :param centers: cluster center;
    :param path: The save path of the file (including file name);
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param saveImg: Whether to save to file;
    :param showInfo: Do you want to output detailed information;
    :param plot_center: Do you want to draw cluster centers.
    """
    img = None
    if showInfo | True:
        img, ax = plt.subplots()
        # Cancel the upper and right border lines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel(img_type + "_1")
        plt.ylabel(img_type + "_2")
        plt.title(img_type)
        # Add legend (at this time, each category needs to be drawn separately)
        # Double cell data (zero)
        doublet_data = dim_data[group_label==0,:]
        # Single cell data (non-zero data)
        other_data = dim_data[group_label!=0,:]
        plt.scatter(other_data[:, 0], other_data[:, 1], c="#7F7F7F", label="singlet", s=10)
        plt.scatter(doublet_data[:, 0], doublet_data[:, 1], c="#FF0000", label="doublet", s=10)
        plt.legend(title="Cluster", loc="center right", bbox_to_anchor=(1.2, 0.5), frameon=False)
        if plot_center:
            plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100)
        # plt.show()
    if saveImg:
        img.savefig(path, bbox_inches='tight')
        img.savefig(path.rstrip("png") + "pdf", bbox_inches='tight')



def get_percent_groups(percent_mtx, combination):
    """
    According to the combination, obtain a simulated doublet cell class.

    :param percent_mtx: The similarity of cells to each cluster;
    :param combination: Array of pairwise combinations of each cluster;
    :return percent_groups: Increased the similarity between cells and two cell classes.
    """
    doublet_groups = []
    for i in range(len(combination)):
        comb = combination[i]
        doublet_group = (percent_mtx[comb[0]] + percent_mtx[comb[1]])/2
        doublet_groups.append(doublet_group)

    doublet_groups = np.stack(doublet_groups)
    percent_groups = np.concatenate([percent_mtx, doublet_groups], axis=0)
    return percent_groups

def common_seq(data, group_label):
    """
    Input data and arrays of each cluster to obtain the common pattern of that cluster (ignoring 0).

    :param data: Cell-SNP matrix;
    :param group_label: The cluster labels of each sample (predicted results);
    :return common_count: Genotyping counting matrix for each cell cluster;
    :return common_percent: Genotyping percentage matrix for each cell cluster;
    :return common: the common pattern of clusters.
    """
    cluster = len(np.unique(group_label))
    snp = data.shape[1]
    counts = []
    for g in range(1, cluster+1):
        per_snp_count = np.zeros((snp, 3), dtype=np.int_)
        temp_data = data[group_label == g,:]
        for s in range(snp):
            uniq, temp_counts = np.unique(temp_data[:,s], return_counts=True)
            # uniq like [0 1 2 3]、[0 1 2]、[0 1 3]
            count = np.zeros(3, dtype=np.int_)
            for i in range(len(uniq)): # 1 2 3
                if uniq[i] != 0:
                    count[uniq[i]-1] = temp_counts[i]
            per_snp_count[s,:] = count
        counts.append(per_snp_count)
    common_count = np.stack(counts)
    common = np.argmax(common_count, axis=2) + 1
    percents = []
    for i in range(cluster):
        temp_counts = common_count[i]
        temp_percents = np.sum(temp_counts, axis=1)
        # Replace zero with 1 to avoid division by zero errors
        temp_percents[temp_percents == 0] = 1  
        percents.append(temp_counts/temp_percents[:, np.newaxis])
    common_percent = np.stack(percents)
    return common_count, common_percent, common

def cluster_mask(common):
    """
    Identify SNP loci that are identical across all clusters.

    :param common: The common pattern of clusters;
    :return mask: Is the SNP site the same across all clusters.
    """
    mask = np.zeros(common.shape[1], dtype=np.int_)
    # Traverse each SNP (column)
    for i in range(common.shape[1]):
        snp = set(common[:,i])
        if len(snp) == 1:
            mask[i] = 1
    return mask == 1

def similarity_mask(cell, percent_mtx, mask):
    """
    Add a mask parameter to filter out SNPs that are the same across all clusters.

    :param cell: The genotype of the cell;
    :param percent_mtx: The similarity of cells to each cluster;
    :param mask: Is the SNP site the same across all clusters;
    :return similarity: Filtered similarity.
    """
    temp_mtx = np.insert(percent_mtx, 0, 0, axis=1)
    cell[mask] = 0
    return np.sum(temp_mtx[np.arange(len(cell)), cell])/np.sum(cell != 0)


def right_three_sigma(samples):
    """
    Calculate the lower bound of 3σ (three standard deviations below the mean).

    :param samples: Single cell similarity of each cell;
    :return right_boundary: The lower bound of 3σ.
    """
    # Calculate the sample mean and sample standard deviation
    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)
    # Obtain the boundary value on the right side
    right_boundary = sample_mean - 2.8*sample_std
    return right_boundary



def save_pred(barcode, group_pred, new_group_pred, doublet_combs, path):
    """
    Save doublet prediction results.

    :param barcode: Barcode List of Cells;
    :param group_pred: The cluster labels of each sample (predicted results);
    :param new_group_pred: The cluster labels of each sample (predicted results, including doublet);
    :return doublet_combs: Source sample of the doublets.
    :param path: The save path of the file (including file name).
    """
    with open(path, "w") as pred:
        pred.write("barcode\tprediction\tbest_single\tdoublet\n")
        for i in range(len(barcode)):
            comb = "--"
            if not isinstance(doublet_combs[i], numbers.Number):
                comb = f"{doublet_combs[i][0]},{doublet_combs[i][1]}"
            temp = str(barcode[i]) + "\t" + str(group_pred[i]) + "\n"
            temp = f"{barcode[i]}\t{new_group_pred[i]}\t{group_pred[i]}\t{comb}\n"
            pred.write(temp)



def result(data, group_pred, doublet_log, combination, barcodes, result_data, prefix, out_path="", is_percent=True, img_type="tSNE", doublet_num=None, saveInfo=True, append=False):
    """
    Based on the previous step's prediction results for each cell, further predict which of them are doublets.

    :param data: Cell-SNP matrix;
    :param group_pred: The cluster labels of each sample (predicted results);
    :param doublet_log: Log file of doublet prediction;
    :param combination: Array of pairwise combinations of each cluster;
    :param barcodes: Barcode List of Cells;
    :param dim_data: Reduced dimensional data;
    :param prefix: The prefix of the output file;
    :param out_path: Output Directory;
    :param is_percent: Is the percentage of doublets provided;
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param doublet_num: Number of doublets;
    :param saveInfo: Is the log file saved;
    :param append: Is the result file appended;
    """
    # doublet prediction
    new_group_pred, doublet_barcodes, doublet_combs = deal_doublet(data, group_pred, doublet_log, combination, barcodes, is_percent, doublet_num, saveInfo=True)

    # plot tSNE/UMAP
    # remove doublet
    img_path = f"{out_path}/imgs/{prefix}{img_type}_pred_rmdoublet.png"
    plot_cluster_img(group_pred[new_group_pred!=0], result_data[new_group_pred!=0], None, img_path, img_type, True, True, False)
    # cluster and doublet
    img_path = f"{out_path}/imgs/{prefix}{img_type}_pred.png"
    plot_cluster_doublet_img(new_group_pred, result_data, None, img_path, img_type, True, True, False)
    # doublet and other
    img_path = f"{out_path}/imgs/{prefix}{img_type}_pred_doublet.png"
    plot_doublet_img(new_group_pred, result_data, None, img_path, img_type, True, True, False)

    # Save prediction results
    save_pred(barcodes, group_pred, new_group_pred, doublet_combs, f"{out_path}/results/{prefix}{img_type}_pred.txt")
    # Save doublet barcode
    np.savetxt(f"{out_path}/results/doublet_barcodes.list", doublet_barcodes, fmt="%s")

